"""Fine-tuning the protein encoder inside the AMP benchmark.

Every AMP result so far was produced over *cached* ESM-2 vectors: the encoder
ran once, offline, and the trainable parameters were all downstream of it. That
makes the comparison between model variants exact and cheap, and it also means
the protein encoder never learned anything about peptides. This module puts the
encoder back in the graph with the last `trainable_blocks` transformer blocks
unfrozen, so the entity representation can move.

Two things are deliberately preserved from the frozen path.

**The heads are the same classes.** `LiveEntityAster` computes `x` from tokens
and then delegates to `RealAster` / `CrossAttentionAster` unchanged, so a
frozen run and a fine-tuned run differ in exactly one factor. Pooling matches
`embed_sequences` (mask-weighted mean over the final hidden state, special
tokens included), so a frozen `LiveEntityAster` reproduces the cached vectors.

**The text side stays frozen.** Questions and answers remain precomputed
embeddings. Unfreezing the text tower as well is a separate, larger experiment;
mixing it in here would leave the result unattributable.

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

# Modes whose entity path is real, so unfreezing the encoder means something.
LIVE_ENCODER_MODES = ("dual", "entity_only")


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


class LiveEntityAster(nn.Module):
    """A frozen-path head, with `x` produced by a live protein encoder.

    The head is used as written: this class only replaces the cached entity
    vector with one computed from tokens in the forward pass, so whatever the
    head does with `x` is unchanged and the two runs stay comparable.
    """

    def __init__(self, head: nn.Module, encoder: nn.Module):
        super().__init__()
        self.head = head
        self.encoder = encoder

    def encode(self, batch) -> torch.Tensor:
        return pooled(self.encoder, {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        })

    def forward(self, batch):
        x = self.encode(batch)
        return self.head({**batch, "x": x})


def param_groups(model: LiveEntityAster, head_lr: float = 1e-3,
                 encoder_lr: float = 2e-5, weight_decay: float = 1e-2):
    """Two learning rates: the head is new, the encoder is pretrained.

    One rate for both is the usual way a fine-tune quietly fails -- 1e-3 walks
    the pretrained weights off a cliff in the first few hundred steps, and 2e-5
    leaves a randomly initialised head barely trained.
    """
    # Also accepts a bare head, so the frozen reference run in the same script
    # goes through one training loop rather than a second, divergent one.
    encoder = getattr(model, "encoder", None)
    enc = [p for p in encoder.parameters() if p.requires_grad] if encoder is not None else []
    head_module = getattr(model, "head", model)
    head = [p for p in head_module.parameters() if p.requires_grad]
    if encoder is not None:
        enc_ids = {id(p) for p in enc}
        head = [p for p in head if id(p) not in enc_ids]
    groups = [{"params": head, "lr": head_lr, "weight_decay": weight_decay}]
    if enc:
        groups.append({"params": enc, "lr": encoder_lr, "weight_decay": weight_decay})
    return groups


def encoder_drift(model: LiveEntityAster, reference: dict) -> float:
    """Mean absolute change in the trainable encoder weights since `reference`.

    A fine-tune that reports a gain while this is ~0 did not fine-tune anything;
    a gain alongside a huge drift is usually the encoder being destroyed and the
    head compensating. Both are worth seeing next to the accuracy.
    """
    total, n = 0.0, 0
    for name, p in model.encoder.named_parameters():
        if not p.requires_grad or name not in reference:
            continue
        total += (p.detach().cpu() - reference[name]).abs().sum().item()
        n += p.numel()
    return total / max(n, 1)


def snapshot_encoder(model: LiveEntityAster) -> dict:
    return {
        name: p.detach().cpu().clone()
        for name, p in model.encoder.named_parameters() if p.requires_grad
    }
