"""Fine-tuning the protein encoder inside the AMP benchmark.

Every AMP result so far was produced over *cached* ESM-2 vectors: the encoder
ran once, offline, and the trainable parameters were all downstream of it. That
makes the comparison between model variants exact and cheap, and it also means
the protein encoder never learned anything about peptides. This module puts the
encoder back in the graph with the last `trainable_blocks` transformer blocks
unfrozen, so the entity representation can move.

Two things are deliberately preserved from the frozen path.

**The heads are the same classes.** `LiveAster` computes `x` from tokens
and then delegates to `RealAster` / `CrossAttentionAster` unchanged, so a
frozen run and a fine-tuned run differ in exactly one factor. Pooling matches
`embed_sequences` (mask-weighted mean over the final hidden state, special
tokens included), so a frozen `LiveAster` reproduces the cached vectors.

**The text side is a separate switch.** Questions and answers are precomputed
embeddings by default, and `--unfreeze-text` puts the text encoder in the graph
too. Keeping them separate is what makes the ladder readable: frozen, then
protein-live, then both-live. A run that unfreezes both at once and gains cannot
say which tower did it, so the flag exists but the default does not use it.

Unfreezing text is much cheaper than it looks. There are as many unique
questions as tasks (~21) and two answer strings in the whole benchmark, so a
step forwards ~23 short sequences and gathers per row, rather than forwarding a
sequence per row.

Which modes get a live encoder matters for the reading. `entity_only` is the
control that detects "answers without reading the question", and it has a
protein encoder too. Giving trainable blocks to the hypothesis model but not to
`entity_only` inflates the hypothesis by exactly the capacity the control was
denied, so both get them. `task_id` and `question_only` have no entity path
worth unfreezing -- `question_only` zeroes the entity slots by construction --
and stay frozen, which is why they are reported as reference rows rather than
as a matched comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aster.model import pooled, unfreeze_suffix
from aster.real.embed import ESM_SIZES

# Modes whose entity path is real, so unfreezing the protein encoder means
# something. question_only zeroes the entity slots by construction.
LIVE_PROTEIN_MODES = ("dual", "entity_only")

# Modes that read text, so unfreezing the text tower means something. The same
# symmetry argument as entity_only applies to question_only: it is the control
# for "answers from the prompt alone", and starving it of capacity the
# hypothesis model gets would inflate the hypothesis by exactly that much.
LIVE_TEXT_MODES = ("dual", "question_only")


def load_protein_encoder(model: str = "8M", trainable_blocks: int = 2,
                         device: str = "cpu", revision: str | None = None):
    """ESM-2 with only the last `trainable_blocks` blocks (plus final norm) trainable."""
    from transformers import AutoModel, AutoTokenizer

    name = ESM_SIZES.get(model, model)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    encoder = AutoModel.from_pretrained(name, revision=revision,
                                        attn_implementation="eager")
    unfreeze_suffix(encoder, trainable_blocks)
    return encoder.to(device), tokenizer


def load_text_encoder(model: str, trainable_blocks: int = 1, device: str = "cpu",
                      revision: str | None = None):
    """The sentence encoder with only its last `trainable_blocks` blocks trainable."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    encoder = AutoModel.from_pretrained(model, revision=revision,
                                        attn_implementation="eager")
    unfreeze_suffix(encoder, trainable_blocks)
    return encoder.to(device), tokenizer


def tokenize_sequences(seqs, tokenizer, max_len: int = 64):
    """Padded token tensors for a list of sequences, in the given order."""
    enc = tokenizer(list(seqs), return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_len)
    return enc["input_ids"], enc["attention_mask"]


def trainable_report(encoder) -> dict:
    """What is actually receiving gradients, for the run report."""
    train_n = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in encoder.parameters())
    names = sorted({
        n.split(".layer.")[0] + ".layer." + n.split(".layer.")[1].split(".")[0]
        if ".layer." in n else n
        for n, p in encoder.named_parameters() if p.requires_grad
    })
    return {
        "trainable_params": train_n,
        "total_params": total_n,
        "trainable_fraction": round(train_n / max(total_n, 1), 5),
        "trainable_modules": names,
    }


class TextTable(nn.Module):
    """Tokenized unique texts, forwarded once per step and gathered per row.

    The benchmark has one question per task and two answer strings in total, so
    the text encoder never needs a forward per row. Registered as buffers so the
    tokens move with `.to(device)` and are saved with the module.
    """

    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.register_buffer("input_ids", input_ids, persistent=False)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def embed(self, encoder):
        return pooled(encoder, {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        })


class LiveAster(nn.Module):
    """A frozen-path head, with its inputs produced by live encoders.

    The head is used as written: this class only replaces cached vectors with
    ones computed in the forward pass, so whatever the head does with them is
    unchanged and the runs stay comparable.

    `text_encoder` is optional. Without it, `q_emb` and `a_emb` come from the
    batch as before, and only the protein tower is live -- which is the default,
    because a run that unfreezes both towers at once cannot say which one moved
    the number.
    """

    def __init__(self, head: nn.Module, encoder: nn.Module,
                 text_encoder: nn.Module | None = None,
                 questions: TextTable | None = None,
                 answers: TextTable | None = None):
        super().__init__()
        self.head = head
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.questions = questions
        self.answers = answers
        if text_encoder is not None and (questions is None or answers is None):
            raise ValueError("A live text encoder needs question and answer tables")

    @property
    def text_is_live(self) -> bool:
        return self.text_encoder is not None

    def encode(self, batch) -> torch.Tensor:
        return pooled(self.encoder, {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        })

    def encode_text(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """(q_emb [b, d], a_emb [b, k, d]) from the live text encoder."""
        if "q_idx" not in batch or "a_idx" not in batch:
            raise KeyError(
                "A live text encoder needs q_idx and a_idx in the batch; the "
                "cached q_emb/a_emb cannot carry gradients back to the encoder"
            )
        q_all = self.questions.embed(self.text_encoder)
        a_all = self.answers.embed(self.text_encoder)
        return q_all[batch["q_idx"]], a_all[batch["a_idx"]]

    def forward(self, batch):
        out = {**batch, "x": self.encode(batch)}
        if self.text_is_live:
            out["q_emb"], out["a_emb"] = self.encode_text(batch)
        return self.head(out)


def param_groups(model, head_lr: float = 1e-3, encoder_lr: float = 2e-5,
                 text_lr: float | None = None, weight_decay: float = 1e-2):
    """Two learning rates: the head is new, the encoder is pretrained.

    One rate for both is the usual way a fine-tune quietly fails -- 1e-3 walks
    the pretrained weights off a cliff in the first few hundred steps, and 2e-5
    leaves a randomly initialised head barely trained.
    """
    # Also accepts a bare head, so the frozen reference run in the same script
    # goes through one training loop rather than a second, divergent one.
    def trainable(module):
        return [p for p in module.parameters() if p.requires_grad] if module is not None else []

    encoder = getattr(model, "encoder", None)
    text = getattr(model, "text_encoder", None)
    enc, txt = trainable(encoder), trainable(text)

    head_module = getattr(model, "head", model)
    taken = {id(p) for p in enc} | {id(p) for p in txt}
    head = [p for p in trainable(head_module) if id(p) not in taken]

    groups = [{"params": head, "lr": head_lr, "weight_decay": weight_decay}]
    if enc:
        groups.append({"params": enc, "lr": encoder_lr, "weight_decay": weight_decay})
    if txt:
        groups.append({"params": txt,
                       "lr": encoder_lr if text_lr is None else text_lr,
                       "weight_decay": weight_decay})
    return groups


def encoder_drift(model, reference: dict, tower: str = "encoder") -> float:
    """Mean absolute change in the trainable encoder weights since `reference`.

    A fine-tune that reports a gain while this is ~0 did not fine-tune anything;
    a gain alongside a huge drift is usually the encoder being destroyed and the
    head compensating. Both are worth seeing next to the accuracy.
    """
    module = getattr(model, tower, None)
    if module is None:
        return 0.0
    total, n = 0.0, 0
    for name, p in module.named_parameters():
        if not p.requires_grad or name not in reference:
            continue
        total += (p.detach().cpu() - reference[name]).abs().sum().item()
        n += p.numel()
    return total / max(n, 1)


def snapshot_encoder(model, tower: str = "encoder") -> dict:
    module = getattr(model, tower, None)
    if module is None:
        return {}
    return {
        name: p.detach().cpu().clone()
        for name, p in module.named_parameters() if p.requires_grad
    }
