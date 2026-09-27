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
        else:
            self.text = Tower(d_text * 2, h, p=p)

    def forward(self, batch):
        n_opt = 2
        if self.mode == "task_id":
            q = self.q(batch["task_id"]).unsqueeze(1)
            v = q * self.a(torch.arange(n_opt, device=q.device)).unsqueeze(0)
        else:
            qt = batch["q_emb"].unsqueeze(1).expand(-1, n_opt, -1)
            v = self.text(torch.cat([qt, batch["a_emb"]], dim=-1))

        if self.mode == "question_only":
            u = torch.ones(v.size(0), self.h, device=v.device) * (self.h ** -0.5)
        else:
            u = self.entity(batch["x"])

        if self.mode == "entity_only":
            v = torch.ones_like(v) * (self.h ** -0.5)

        return torch.einsum("bh,boh->bo", u, v) * (self.h ** -0.5)


class CrossAttentionAster(nn.Module):
    """Question conditions entity representation via Multi-Head Cross-Attention.

    Instead of a rigid pooled dot product u^T v, the question queries the entity
    representation across multi-channel feature subspaces, allowing language tokens
    (cationic, hydrophobic, lipid A, peptidoglycan) to dynamically select and weight
    relevant biophysical features.
    """

    def __init__(self, d_entity, d_text, n_tasks, h=256, n_heads=4, mode="dual", p=0.1, n_slots=8):
        super().__init__()
        self.mode = mode
        self.h = h
        self.n_slots = n_slots

        # Project entity into feature slots (or residue subspaces)
        self.entity_proj = nn.Sequential(
            nn.Linear(d_entity, h * n_slots),
            nn.GELU(),
            nn.LayerNorm(h * n_slots),
            nn.Dropout(p),
        )

        if mode == "task_id":
            self.q = nn.Embedding(n_tasks + 1, h)
            nn.init.zeros_(self.q.weight[0])  # unseen = 0 -> chance
        else:
            self.q_proj = nn.Sequential(
                nn.Linear(d_text, h),
                nn.GELU(),
                nn.LayerNorm(h),
                nn.Dropout(p),
            )

        # Cross attention: Query = Question, Key/Value = Entity Slots
        self.cross_attn = nn.MultiheadAttention(embed_dim=h, num_heads=n_heads, dropout=p, batch_first=True)
        self.norm = nn.LayerNorm(h)

        # Classification head for binary choice (0 = inactive/negative, 1 = active/positive)
        self.head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h // 2, 2),
        )

    def forward(self, batch):
        b = batch["x"].size(0)

        # 1. Prepare Key & Value from Entity
        if self.mode == "question_only":
            # Zero out entity so model must rely only on question prior
            kv = torch.zeros(b, self.n_slots, self.h, device=batch["x"].device)
        else:
            slots = self.entity_proj(batch["x"])  # [B, n_slots * H]
            kv = slots.view(b, self.n_slots, self.h)  # [B, n_slots, H]

        # 2. Prepare Query from Question
        if self.mode == "task_id":
            q = self.q(batch["task_id"]).unsqueeze(1)  # [B, 1, H]
        else:
            q_raw = batch["q_emb"]
            if self.mode == "entity_only":
                q = torch.ones(b, 1, self.h, device=q_raw.device) * (self.h ** -0.5)
            else:
                q = self.q_proj(q_raw).unsqueeze(1)  # [B, 1, H]

        # 3. Cross-Attention
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)  # [B, 1, H]
        out = self.norm(q + attn_out).squeeze(1)  # [B, H]

        # 4. Predict logits
        return self.head(out)
