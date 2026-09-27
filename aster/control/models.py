"""Models under test, plus the controls that make the test meaningful.

The dual encoder is the hypothesis. Everything else in this file exists to try
to explain away its results:

  TaskIDModel     -- replaces the text encoder with a lookup table. If it ties
                     the dual encoder on seen questions, the text encoder is
                     not earning its parameters. It CANNOT handle an unseen
                     question, so it is also the floor for transfer claims.
  FrozenEntity    -- entity encoder is a fixed random projection. Tells us how
                     much of the score comes from fine-tuning the entity side.
  QuestionOnly    -- never looks at the entity. Any accuracy it gets is label
                     prior, not biology. This is the shortcut detector.
  EntityOnly      -- never looks at the question. Catches the mirror shortcut.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aster.control.synthetic import ALPHABET


# --------------------------------------------------------------------- vocab

class Vocab:
    def __init__(self, tokens):
        self.itos = ["<unk>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, 0) for t in toks]


def build_vocabs(train_examples):
    """Vocabularies are built from TRAIN ONLY.

    Peeking at test tokens here would quietly hand unseen questions a trained
    embedding and manufacture transfer that is not there.
    """
    qt, opts, qids = [], [], []
    for e in train_examples:
        qt.extend(e.question_tokens)
        opts.extend(e.options)
        qids.append(e.question_id)
    return Vocab(qt), Vocab(opts), Vocab(qids)


# ------------------------------------------------------------------ encoders

class EntityEncoder(nn.Module):
    """Stand-in for ESM-2. Deliberately small: this rig tests the harness and
    the training/eval contract, not representation quality."""

    def __init__(self, d=64, h=96, frozen=False):
        super().__init__()
        self.emb = nn.Embedding(len(ALPHABET) + 1, d, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, h)
        )
        self.frozen = frozen
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, ids, mask):
        x = self.emb(ids)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.net(x)


class TextEncoder(nn.Module):
    """Question tokens + one candidate answer -> a vector in scoring space."""

    def __init__(self, n_q, n_opt, d=64, h=96):
        super().__init__()
        self.q = nn.Embedding(n_q, d)
        self.a = nn.Embedding(n_opt, d)
        self.net = nn.Sequential(
            nn.Linear(2 * d, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, h)
        )

    def forward(self, q_ids, q_mask, a_ids):
        q = self.q(q_ids)
        q = (q * q_mask.unsqueeze(-1)).sum(1) / q_mask.sum(1, keepdim=True).clamp(min=1)
        a = self.a(a_ids)
        q = q.unsqueeze(1).expand(-1, a.size(1), -1)
        return self.net(torch.cat([q, a], dim=-1))


class TaskIDText(nn.Module):
    """The control: question identity as an opaque row in a table.

    Index 0 is <unk> and is never trained, so an unseen question scores at
    chance -- which is exactly the honest answer for a lookup table.
    """

    def __init__(self, n_qid, n_opt, h=96):
        super().__init__()
        self.q = nn.Embedding(n_qid, h)
        self.a = nn.Embedding(n_opt, h)
        nn.init.zeros_(self.q.weight[0])

    def forward(self, qid, a_ids):
        q = self.q(qid).unsqueeze(1)
        return q * self.a(a_ids)


# -------------------------------------------------------------------- scorer

class ControlScorer(nn.Module):
    """entity -> u ; (question, answer) -> v ; score = <u, v> / sqrt(h)."""

    def __init__(self, vq, vopt, vqid, mode="dual", h=96):
        super().__init__()
        self.mode = mode
        self.h = h
        self.entity = EntityEncoder(h=h, frozen=(mode == "frozen_entity"))
        if mode == "task_id":
            self.text = TaskIDText(len(vqid), len(vopt), h=h)
        elif mode == "entity_only":
            # A learned row per option and nothing else. Setting v to a constant
            # instead makes every option score identical, so the model emits one
            # answer for every row in the dataset -- that measures the split's
            # label prior, not an entity shortcut, and it does so while looking
            # like a working control.
            self.opt = nn.Embedding(len(vopt), h)
        else:
            self.text = TextEncoder(len(vq), len(vopt), h=h)
        self.scale = h ** -0.5

    def forward(self, batch):
        if self.mode == "task_id":
            v = self.text(batch["qid"], batch["opt_ids"])
        elif self.mode == "entity_only":
            # Which option is which, with no question content at all.
            v = self.opt(batch["opt_ids"])
        else:
            v = self.text(batch["q_ids"], batch["q_mask"], batch["opt_ids"])

        if self.mode == "question_only":
            # No entity signal at all: a constant probe. Whatever this scores
            # is the label prior, and must be subtracted from any claim.
            u = torch.ones(
                v.size(0), self.h, device=v.device, dtype=v.dtype
            ) * self.scale
        else:
            u = self.entity(batch["e_ids"], batch["e_mask"])

        logits = torch.einsum("bh,boh->bo", u, v) * self.scale
        logits = logits.masked_fill(~batch["opt_mask"], float("-inf"))
        return logits


MODES = ("dual", "task_id", "frozen_entity", "question_only", "entity_only")

MODE_NOTES = {
    "dual": "the hypothesis: text encoder reads the question",
    "task_id": "control: question as lookup id; chance on unseen questions",
    "frozen_entity": "ablation: entity encoder not fine-tuned",
    "question_only": "shortcut detector: label prior without the entity",
    "entity_only": "shortcut detector: entity prior without the question",
}
