"""Tensorization.

Encoded once per split rather than per batch -- the rig gets run repeatedly and
a slow harness is a harness people skip.
"""

from __future__ import annotations

import numpy as np
import torch

from aster.control.synthetic import ALPHABET

_RES = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 == pad


class TensorSet:
    """Whole-dataset tensors; batching is then just indexing."""

    def __init__(self, examples, vq, vopt, vqid, max_len=64, device="cpu"):
        b = len(examples)
        max_q = max(len(e.question_tokens) for e in examples)
        max_o = max(len(e.options) for e in examples)
        L = min(max_len, max(len(e.entity) for e in examples))

        e_ids = np.zeros((b, L), dtype=np.int64)
        e_mask = np.zeros((b, L), dtype=np.float32)
        q_ids = np.zeros((b, max_q), dtype=np.int64)
        q_mask = np.zeros((b, max_q), dtype=np.float32)
        opt_ids = np.zeros((b, max_o), dtype=np.int64)
        opt_mask = np.zeros((b, max_o), dtype=bool)
        qid = np.zeros(b, dtype=np.int64)
        y = np.zeros(b, dtype=np.int64)

        for i, e in enumerate(examples):
            seq = e.entity[:L]
            n = len(seq)
            e_ids[i, :n] = [_RES.get(c, 0) for c in seq]
            e_mask[i, :n] = 1.0
            qt = vq.encode(e.question_tokens)
            q_ids[i, :len(qt)] = qt
            q_mask[i, :len(qt)] = 1.0
            ot = vopt.encode(e.options)
            opt_ids[i, :len(ot)] = ot
            opt_mask[i, :len(ot)] = True
            qid[i] = vqid.stoi.get(e.question_id, 0)
            y[i] = e.label

        t = lambda a: torch.from_numpy(a).to(device)
        self.d = {"e_ids": t(e_ids), "e_mask": t(e_mask), "q_ids": t(q_ids),
                  "q_mask": t(q_mask), "opt_ids": t(opt_ids),
                  "opt_mask": t(opt_mask), "qid": t(qid), "y": t(y)}
        self.n = b

    def batch(self, idx):
        return {k: v[idx] for k, v in self.d.items()}

    def iter_batches(self, bs, shuffle=True, seed=0):
        idx = torch.arange(self.n, device=self.d["y"].device)
        if shuffle:
            g = torch.Generator().manual_seed(int(seed))
            idx = idx[torch.randperm(self.n, generator=g).to(idx.device)]
        for i in range(0, self.n, bs):
            yield self.batch(idx[i:i + bs])
