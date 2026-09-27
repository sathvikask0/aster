"""The control rig, as tests.

`test_arbitrary_regime_shows_no_transfer` is the load-bearing one. It fails if
the harness ever reports generalization on data constructed so that
generalization is impossible -- which is the exact way a joint-encoder result
can look good and mean nothing.
"""

from __future__ import annotations

import pytest

import torch

from aster.control import splits as S
from aster.control.harness import make_sets, run_grid, train_one, verdict
from aster.control.synthetic import build

SMALL = dict(examples_per_question=120, n_paired_questions=32)
EPOCHS = 8
TOL = 0.08


def _lift(results, mode):
    return {r.mode: r for r in results}[mode].test_calibrated.lift_over_chance


@pytest.fixture(scope="module")
def semantic_heldout():
    ex, _ = build("semantic", seed=7, **SMALL)
    split = S.held_out_question(ex, seed=3)
    return split, run_grid(split, seed=1, epochs=EPOCHS)


@pytest.fixture(scope="module")
def arbitrary_heldout():
    ex, _ = build("arbitrary", seed=7, **SMALL)
    split = S.held_out_question(ex, seed=3)
    return split, run_grid(split, seed=1, epochs=EPOCHS)


def test_splits_are_disjoint_in_the_dimension_they_claim():
    ex, _ = build("semantic", seed=0, **SMALL)

    sp = S.held_out_question(ex, seed=1)
    assert not ({e.question_id for e in sp.train} & {e.question_id for e in sp.test})

    sp = S.held_out_entity_family(ex, seed=1)
    assert not ({e.entity_family for e in sp.train} & {e.entity_family for e in sp.test})

    sp = S.held_out_both(ex, seed=1)
    assert not ({e.question_id for e in sp.train} & {e.question_id for e in sp.test})
    assert not ({e.entity_family for e in sp.train} & {e.entity_family for e in sp.test})


def test_calibration_set_never_overlaps_test():
    ex, _ = build("semantic", seed=0, **SMALL)
    for name, fn in S.ALL_SPLITS.items():
        sp = fn(ex, seed=1)
        assert not ({id(e) for e in sp.val} & {id(e) for e in sp.test}), name


def test_semantic_regime_transfers(semantic_heldout):
    _, res = semantic_heldout
    assert _lift(res, "dual") > 0.25


def test_arbitrary_regime_shows_no_transfer(arbitrary_heldout):
    """If this fails, every transfer number the harness reports is suspect."""
    _, res = arbitrary_heldout
    assert _lift(res, "dual") < TOL


def test_task_id_is_at_chance_on_unseen_questions(semantic_heldout, arbitrary_heldout):
    for _, res in (semantic_heldout, arbitrary_heldout):
        assert _lift(res, "task_id") < TOL


def _floor(res):
    """The measured shortcut ceiling: how well you do without using both inputs."""
    return max(0.0, _lift(res, "question_only"), _lift(res, "entity_only"))


def test_claims_are_margins_over_the_measured_shortcut(semantic_heldout, arbitrary_heldout):
    """This test used to assert the detectors sit at chance. That expectation
    was wrong twice over: `harness.shortcut_ceiling` already documents why a
    shortcut is a measured property of the split, and the assertion only ever
    passed because `entity_only` was a constant predictor that could not express
    a shortcut at all. What has to hold is the margin, not the absolute.
    """
    _, sem = semantic_heldout
    assert _lift(sem, "dual") - _floor(sem) > 0.25

    _, arb = arbitrary_heldout
    assert _lift(arb, "dual") - _floor(arb) < TOL


def test_entity_only_is_not_a_constant_predictor(semantic_heldout):
    """A control that emits one answer for every row measures the label prior
    and calls it an entity shortcut. Pinned so it cannot come back.
    """
    split, _ = semantic_heldout
    sets = make_sets(split)
    model = train_one(sets, "entity_only", seed=1, epochs=EPOCHS)
    with torch.no_grad():
        logits = model(sets["test"].batch(slice(None)))
    spread = (logits - logits[:, :1]).abs().max().item()
    assert spread > 1e-4, "entity_only scores every option identically"
    assert logits.argmax(1).float().std().item() > 0, "entity_only emits one answer for every row"


def test_family_shift_creates_a_real_entity_only_shortcut():
    """Pins a finding, so it cannot regress into a silent assumption.

    Under covariate shift, an entity-only baseline is NOT at chance. A held-out
    family far from the origin in latent space gets the same answer to most
    questions, so "ignore the question" is a genuine strategy. Any claim on a
    family-held-out split has to be a margin over this, not an absolute.
    """
    ex, _ = build("arbitrary", seed=7, **SMALL)
    sp = S.held_out_entity_family(ex, seed=3)
    res = run_grid(sp, modes=("dual", "entity_only"), seed=1, epochs=EPOCHS)
    floor = _lift(res, "entity_only")
    assert floor > TOL, "expected a real shortcut under family shift"
    assert _lift(res, "dual") > floor + 0.30, "dual must clear the shortcut"


def test_verdict_warns_when_a_shortcut_is_large():
    ex, _ = build("arbitrary", seed=7, **SMALL)
    sp = S.held_out_entity_family(ex, seed=3)
    res = run_grid(sp, seed=1, epochs=EPOCHS)
    lines = verdict("arbitrary", sp.name, res)
    assert any(l.startswith("WARN") for l in lines)
    assert any(l.startswith("INFO  shortcut ceiling") for l in lines)


def test_label_prior_skew_is_detected_by_question_only():
    """Sanity check on the detector itself: skew the labels and it must fire.

    A detector that never fires is not evidence of a clean dataset.
    """
    ex, _ = build("semantic", seed=5, label_prior_skew=0.85, **SMALL)
    sp = S.random_split(ex, seed=2)
    res = run_grid(sp, modes=("question_only",), seed=1, epochs=EPOCHS)
    assert _lift(res, "question_only") > 0.2


def test_verdict_reports_pass_fail(semantic_heldout):
    sp, res = semantic_heldout
    lines = verdict("semantic", sp.name, res)
    assert lines
    assert all(l.startswith(("PASS", "FAIL", "INFO", "WARN")) for l in lines)
    assert any(l.startswith(("PASS", "FAIL")) for l in lines)
    assert not any(l.startswith("FAIL") for l in lines)
