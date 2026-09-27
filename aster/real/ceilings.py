"""Shortcut ceilings: how much of a split is answerable without the model.

The control rig's reading rule is that the shortcut ceiling, not chance and not
the lookup floor, is the real zero. This module is the one place that ceiling is
computed, so the benchmark and any later re-analysis cannot drift apart.

Two things are measured per held-out task:

  majority          always answer the more common label
  composition       a linear probe on 20-dim amino-acid frequencies

The composition probe is scored **out of fold**. An in-sample probe is fitted on
the same rows it is scored on, so it reports the probe's capacity as well as the
shortcut's strength -- which biases the ceiling in whichever direction the
sample happens to favour and makes "lift over ceiling" unreadable. The in-sample
number is still returned as `composition_insample` so the difference stays
visible.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AA_LIST = sorted("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA_LIST)}


def composition_matrix(seqs: list[str]) -> np.ndarray:
    """20-dim amino-acid frequency vector per sequence."""
    mat = np.zeros((len(seqs), 20), dtype=np.float64)
    for i, s in enumerate(seqs):
        for a in s:
            j = AA_INDEX.get(a)
            if j is not None:
                mat[i, j] += 1.0
        mat[i] /= max(1, len(s))
    return mat


def binomial_ci95(acc: float, n: int) -> float:
    """Half-width of the normal-approximation 95% interval on an accuracy."""
    if n <= 0:
        return float("nan")
    return 1.96 * math.sqrt(max(acc * (1.0 - acc), 1e-12) / n)


def composition_ceiling(
    seqs: list[str],
    labels: np.ndarray,
    folds: int = 5,
    seed: int = 0,
) -> dict:
    """Shortcut ceiling for one task: max(majority, out-of-fold composition probe)."""
    y = np.asarray(labels).astype(int)
    n = len(y)
    X = composition_matrix(seqs)

    p1 = float(y.mean()) if n else 0.0
    majority = max(p1, 1.0 - p1)

    def probe():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )

    insample = float("nan")
    oof = float("nan")
    if n >= 2 * folds and 0 < y.sum() < n:
        fitted = probe().fit(X, y)
        insample = float((fitted.predict(X) == y).mean())

        preds = np.zeros(n, dtype=int)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(X, y):
            preds[test_idx] = probe().fit(X[train_idx], y[train_idx]).predict(X[test_idx])
        oof = float((preds == y).mean())

    ceiling = max(majority, oof) if not math.isnan(oof) else majority
    return {
        "n": int(n),
        "majority": float(majority),
        "composition_oof": None if math.isnan(oof) else float(oof),
        "composition_insample": None if math.isnan(insample) else float(insample),
        "ceiling": float(ceiling),
        "ceiling_ci95": float(binomial_ci95(ceiling, n)),
        "estimator": f"logreg(standardized 20-dim AA freq), {folds}-fold out-of-fold",
    }
