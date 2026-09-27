"""Models over precomputed embeddings, with the same controls as the rig.

Modes mirror aster/control/models.py so results are read the same way:

  dual           the hypothesis -- ESM-2 vector vs frozen text vector
  task_id        control -- question as a lookup row; chance on an unseen task
  composition    control -- amino-acid frequencies instead of ESM-2, so we can
                 tell whether the protein model earns its cost
  question_only  shortcut detector -- label prior, never sees the peptide
  entity_only    shortcut detector -- never sees the question
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MODES = ("dual", "task_id", "composition", "question_only", "entity_only")

MODE_NOTES = {
    "dual": "ESM-2 + frozen text encoder (the hypothesis)",
    "task_id": "question as lookup id -- chance on an unseen task",
    "composition": "amino-acid frequencies instead of ESM-2",
    "question_only": "label prior; never sees the peptide",
    "entity_only": "peptide prior; never sees the question",
}


class Tower(nn.Module):
    def __init__(self, d_in, h, depth=2, p=0.1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth - 1):
            layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, h)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RealAster(nn.Module):
    """entity -> u ; (question, answer) -> v ; score = <u, v> / sqrt(h)."""

    def __init__(self, d_entity, d_text, n_tasks, h=256, mode="dual", p=0.1):
        super().__init__()
        self.mode, self.h = mode, h

        d_in = d_entity
        self.entity = Tower(d_in, h, p=p)

        if mode == "task_id":
            # Row 0 is <unk>, zero-initialized and never trained, so an unseen
            # task yields v = 0 -> uniform logits -> chance. Structural, not
            # empirical: the floor cannot be accidentally beaten.
            self.q = nn.Embedding(n_tasks + 1, h)
            nn.init.zeros_(self.q.weight[0])
            self.a = nn.Embedding(2, h)
        elif mode == "entity_only":
            # A learned row per option, carrying no question content. Setting v
            # to a constant instead would make every option score identical, so
            # the model could only ever emit one answer for every row in the
            # dataset -- that measures the global label prior, which is what
            # question_only already measures, and leaves the entity shortcut
            # this control is named after unmeasured.
            self.a = nn.Embedding(2, h)

        if mode not in ("task_id", "entity_only"):
            self.text = Tower(d_text * 2, h, p=p)

    def forward(self, batch):
        n_opt = batch["a_emb"].size(1)
        if n_opt != 2:
            raise ValueError(
                f"RealAster is built for 2 options, got {n_opt}; "
                "use CrossAttentionAster for larger option sets"
            )
        device = batch["x"].device
        if self.mode == "task_id":
            q = self.q(batch["task_id"]).unsqueeze(1)
            v = q * self.a(torch.arange(n_opt, device=device)).unsqueeze(0)
        elif self.mode == "entity_only":
            v = self.a(torch.arange(n_opt, device=device)).unsqueeze(0).expand(
                batch["x"].size(0), -1, -1
            )
        else:
            qt = batch["q_emb"].unsqueeze(1).expand(-1, n_opt, -1)
            v = self.text(torch.cat([qt, batch["a_emb"]], dim=-1))

        if self.mode == "question_only":
            u = torch.ones(v.size(0), self.h, device=device) * (self.h ** -0.5)
        else:
            u = self.entity(batch["x"])

        return torch.einsum("bh,boh->bo", u, v) * (self.h ** -0.5)


class CrossAttentionAster(nn.Module):
    """Question and candidate answer jointly query the entity representation.

    One query per candidate answer; the entity is projected into `n_slots`
    key/value slots; the scored output is a scalar per option. So the option set
    is free -- k options in, k logits out -- which is what "typed decision" is
    supposed to mean, and what `aster/model.py` does with a dot product.

    Architecture version 2. Version 1 built a single query from the question
    alone and ended in a fixed 2-way classification head, which meant the answer
    text was never read and the option count was frozen at two. Every published
    v0.2 number was produced by version 1; they are not comparable to anything
    this class produces now and must be re-run.

    Mode semantics, all preserved under option scoring:

      dual           query = f(question, answer_i). The hypothesis.
      task_id        query = task_row * answer_row_i. An unseen task uses the
                     zeroed row 0, so every option gets an identical query and
                     therefore an identical score -> uniform. Structurally
                     chance; it cannot be beaten by luck.
      question_only  entity slots zeroed, so attention returns nothing and the
                     score depends on (question, answer) alone. Label prior.
      entity_only    query is a learned per-option row carrying no question
                     content, so scores vary with the entity and with which
                     option is which, but never with what the question asks.
                     The mirror shortcut.
    """

    ARCH_VERSION = 2

    def __init__(self, d_entity, d_text, n_tasks, h=256, n_heads=4, mode="dual",
                 p=0.1, n_slots=8, max_options=8):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        self.mode, self.h, self.n_slots = mode, h, n_slots
        self.max_options = max_options

        self.entity_proj = nn.Sequential(
            nn.Linear(d_entity, h * n_slots),
            nn.GELU(),
            nn.LayerNorm(h * n_slots),
            nn.Dropout(p),
        )

        # A per-option row, created only for the two modes whose query must
        # distinguish options without reading their text. In every other mode
        # the option is identified by its own embedding, so carrying this would
        # leave a parameter that never receives a gradient.
        if mode in ("task_id", "entity_only"):
            self.option_row = nn.Embedding(max_options, h)

        if mode == "task_id":
            # Row 0 is <unk>, zero-initialized and never trained: an unseen task
            # yields identical queries across options -> uniform logits.
            self.task_row = nn.Embedding(n_tasks + 1, h)
            nn.init.zeros_(self.task_row.weight[0])
        elif mode != "entity_only":
            # Query is built from the question AND the candidate answer.
            self.q_proj = nn.Sequential(
                nn.Linear(d_text * 2, h),
                nn.GELU(),
                nn.LayerNorm(h),
                nn.Dropout(p),
            )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h, num_heads=n_heads, dropout=p, batch_first=True
        )
        self.norm = nn.LayerNorm(h)
        self.score = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h // 2, 1),
        )

    def queries(self, batch, b, k):
        """[b, k, h] -- one query per candidate answer."""
        device = batch["x"].device
        if self.mode in ("task_id", "entity_only"):
            rows = self.option_row(torch.arange(k, device=device)).unsqueeze(0)
            if self.mode == "entity_only":
                return rows.expand(b, -1, -1)
            return self.task_row(batch["task_id"]).unsqueeze(1) * rows
        q = batch["q_emb"].unsqueeze(1).expand(-1, k, -1)
        return self.q_proj(torch.cat([q, batch["a_emb"]], dim=-1))

    def forward(self, batch):
        b = batch["x"].size(0)
        k = batch["a_emb"].size(1)
        if k > self.max_options:
            raise ValueError(f"{k} options exceeds max_options={self.max_options}")

        if self.mode == "question_only":
            # No entity signal: attention returns zeros, so whatever this scores
            # is the label prior and must be subtracted from any claim.
            kv = torch.zeros(b, self.n_slots, self.h, device=batch["x"].device)
        else:
            kv = self.entity_proj(batch["x"]).view(b, self.n_slots, self.h)

        q = self.queries(batch, b, k)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        out = self.norm(q + attn_out)
        logits = self.score(out).squeeze(-1)

        if "option_mask" in batch:
            mask = batch["option_mask"].bool()
            if not mask.any(1).all():
                raise ValueError("Every row needs at least one valid option")
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits
