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
