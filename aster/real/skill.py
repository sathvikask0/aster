"""Skill scores: loss measured against a reference that knows the answer distribution.

Raw log loss is not a safe selection metric on a held-out-question protocol.
Among constant predictors, the one minimising log loss on a set of labels is
exactly that set's base rate, so on assays that are 95% negative a model can
reach a low log loss by predicting the prevailing class and never reading the
molecule at all. Selection then prefers giving up: in the strict ToxCast run the
`question_only` control scored a *better* macro log loss than the real model
while sitting at AUROC 0.5000 and MCC 0.

A skill score removes that option by dividing through:

    skill = 1 - loss(model) / loss(reference)

The reference is the held-out question's own base rate (classification) or its
own median (regression), so it is deliberately strong -- it is handed the label
distribution the model has to infer. The consequences are what make it usable
for selection:

  * skill = 1 is perfect, skill = 0 is no better than knowing the distribution,
    skill < 0 is worse than that.
  * Any constant predictor scores at most 0, and exactly 0 only if the constant
    equals the base rate. Predicting the prevailing class is no longer a way to
    win, which is the whole point.
  * It is dimensionless, so a classification skill and a regression skill can be
    macro-averaged together. Raw log loss and raw MAE cannot: they are in
    different units and the larger-magnitude family silently dominates.

Skill is undefined where the reference loss is zero -- a single-class assay, or
a regression target with no spread -- and those questions return None rather
than a fabricated number.

What skill is NOT good for: selection, on its own. Log loss is a proper scoring
rule, so it answers discrimination and calibration at once, and when a model is
badly miscalibrated the calibration term dominates and hides the discrimination
entirely. Measured on the strict ToxCast split, `question_only` scores skill
-0.2545 at AUROC 0.5000 while `question` scores -0.4082 at AUROC 0.6762: the
model that ranks molecules correctly is penalised for being overconfident about
it, and the one that ranks nothing still comes out ahead. Held-out questions make
this unavoidable, because calibrating to an unseen assay's prior would need
labels from that assay.

So the two metrics have separate jobs, and both are reported:

  * macro AUROC selects. It is invariant to any monotone rescaling of the
    predictions, so miscalibration cannot mask discrimination, and a constant
    predictor is pinned at exactly 0.5 and can never win.
  * macro skill gates the claim. Skill above 0 is what licenses calling the
    outputs probabilities; below 0 the ranking may be real while the numbers
    are not yet usable, which is a result worth stating rather than hiding.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def log_loss_of(labels, probabilities) -> float:
    """Mean binary cross-entropy in nats, clipped so a confident miss stays finite."""
    y = np.asarray(labels, dtype=float)
    p = _clip(probabilities)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def classification_skill(labels, probabilities) -> dict:
    """Log loss against the base-rate reference for these same labels."""
    y = np.asarray(labels, dtype=float)
    model = log_loss_of(y, probabilities)
    rate = float(np.mean(y))
    reference = log_loss_of(y, np.full(len(y), rate))
    # A single-class set has zero reference loss; no finite skill exists.
    defined = len(y) > 0 and 0 < rate < 1
    return {"log_loss": model, "base_rate": round(rate, 6),
            "reference_log_loss": reference,
            "log_loss_skill": float(1 - model / reference) if defined else None}


def regression_skill(values, predictions) -> dict:
    """Absolute error against the median reference for these same values.

    Median, not mean, because it is the constant that minimises absolute error,
    which makes the reference the best a distribution-only predictor can do.
    """
    y = np.asarray(values, dtype=float)
    model = float(np.mean(np.abs(y - np.asarray(predictions, dtype=float))))
    median = float(np.median(y))
    reference = float(np.mean(np.abs(y - median)))
    defined = len(y) > 0 and reference > 0
    return {"mae": model, "reference_median": median, "reference_mae": reference,
            "mae_skill": float(1 - model / reference) if defined else None}


def macro_skill(values) -> float | None:
    """Mean over the questions where skill is defined, or None if none are."""
    usable = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(usable)) if usable else None


def rank_skill(kind: str, targets, predictions) -> float | None:
    """Discrimination on a common scale: 0 is chance, 1 is a perfect ordering.

    This is what selection runs on, because it is invariant to any monotone
    rescaling of the predictions and so cannot be fooled by the overconfidence
    that dominates log loss on held-out questions.

    Classification uses Somers' D (2*AUROC - 1) rather than AUROC so that chance
    sits at 0 on both families and the two can be macro-averaged. Regression uses
    Spearman, the same quantity for a continuous target.

    A prediction with no spread orders nothing, so it scores 0 rather than None --
    a constant predictor must be scored, not skipped. None is reserved for a
    target with no spread, where no ordering exists to recover.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    if len(y) < 2 or np.ptp(y) == 0:
        return None
    if np.ptp(p) == 0:
        return 0.
    if kind == "classification":
        return float(2 * roc_auc_score(y, p) - 1)
    return float(spearmanr(y, p).statistic)
