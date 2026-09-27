"""Control rig: synthetic tasks with known-ground-truth transfer structure.

This package does not model biology and is not part of Aster's modelling path.
It exists to test the *evaluation* before the evaluation is pointed at real
data.

Aster's central claim is that a text encoder reading the question buys
generalization to questions the model was not trained on. The failure mode is
that the text encoder quietly degenerates into task identity -- an expensive
one-hot -- which produces transfer-shaped numbers with no transfer behind them.
That failure is invisible if you only ever look at accuracy on a random split.

So: two regimes are generated whose answers we already know.

    semantic   the question text compositionally determines the decision rule,
               so a model that genuinely reads it SHOULD generalize to unseen
               questions built from familiar parts.

    arbitrary  the question text is an opaque identifier and the rule is drawn
               independently of it. Nothing can generalize. Chance is the
               ceiling, by construction.

A harness worth trusting reports transfer in the first and none in the second.
If it reports transfer in `arbitrary`, the harness is broken, and no result it
later produces on real data means anything.

Entry point: `scripts/run_control.py`.
"""

from aster.control import harness, metrics, splits, synthetic  # noqa: F401

__all__ = ["synthetic", "splits", "metrics", "harness"]
