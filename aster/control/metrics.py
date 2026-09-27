"""Accuracy is the least interesting number here.

A model that is right 80% of the time and confident 99% of the time is worse
than useless for deciding which experiment to run -- it spends the researcher's
bench time on its own overconfidence. So calibration is reported alongside
accuracy, never after it, and never as an afterthought.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Metrics:
    n: int
    accuracy: float
    chance: float
    lift_over_chance: float   # 0 == no better than guessing
    nll: float
    brier: float
    ece: float
    mce: float
    mean_confidence: float
    overconfidence: float     # mean_confidence - accuracy; >0 means too sure

    def as_dict(self):
        return asdict(self)


def _bin_stats(conf, correct, n_bins=12):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        out.append((m.sum(), conf[m].mean(), correct[m].mean()))
    return out


def compute(probs: np.ndarray, labels: np.ndarray, n_bins: int = 12) -> Metrics:
    """probs: (N, K) rows sum to 1. labels: (N,) index of the true option."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, k = probs.shape

    pred = probs.argmax(1)
    correct = (pred == labels).astype(np.float64)
    conf = probs.max(1)
    acc = float(correct.mean())
    chance = 1.0 / k

    p_true = np.clip(probs[np.arange(n), labels], 1e-12, 1.0)
    nll = float(-np.log(p_true).mean())

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    brier = float(((probs - onehot) ** 2).sum(1).mean())

    bins = _bin_stats(conf, correct, n_bins)
    ece = float(sum(c * abs(a - m) for c, m, a in bins) / max(n, 1))
    mce = float(max((abs(a - m) for _, m, a in bins), default=0.0))

    return Metrics(
        n=n,
        accuracy=acc,
        chance=chance,
        lift_over_chance=float((acc - chance) / max(1.0 - chance, 1e-9)),
        nll=nll,
        brier=brier,
        ece=ece,
        mce=mce,
        mean_confidence=float(conf.mean()),
        overconfidence=float(conf.mean() - acc),
    )


def reliability_table(probs, labels, n_bins=12):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    rows = []
    for count, mean_conf, mean_acc in _bin_stats(conf, correct, n_bins):
        rows.append({"n": int(count), "confidence": float(mean_conf),
                     "accuracy": float(mean_acc),
                     "gap": float(mean_acc - mean_conf)})
    return rows


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """1-D temperature search on a held-out calibration set.

    Deliberately fit on val, never test. Temperature scaling cannot fix a model
    that is wrong; it can only stop a model from lying about how sure it is.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    best_t, best_nll = 1.0, np.inf
    for t in np.geomspace(0.25, 8.0, 80):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        nll = float((lse - z[np.arange(len(labels)), labels]).mean())
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
