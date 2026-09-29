"""The properties that make skill safe to select on.

The point of these tests is the constant-predictor bound. Raw log loss did not
have it, which is why the ToxCast question_only control could outscore the real
model while discriminating nothing.
"""
import numpy as np
import pytest

from aster.real.skill import (classification_skill, log_loss_of, macro_skill, rank_skill,
                              regression_skill)

LOPSIDED = [0] * 95 + [1] * 5


def test_base_rate_predictor_scores_exactly_zero():
    assert classification_skill(LOPSIDED, [0.05] * 100)["log_loss_skill"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("constant", [0.0001, 0.01, 0.02, 0.1, 0.3, 0.5, 0.9, 0.9999])
def test_no_constant_predictor_beats_the_base_rate(constant):
    """The whole guarantee: predicting the prevailing class can never win."""
    assert classification_skill(LOPSIDED, [constant] * 100)["log_loss_skill"] <= 0


def test_a_constant_can_beat_the_base_rate_on_raw_log_loss():
    """Documents the bug being fixed: raw loss prefers the confident constant."""
    balanced = [0] * 50 + [1] * 50
    informative = [0.2] * 50 + [0.8] * 50  # reads the entity, ranks perfectly
    assert log_loss_of(balanced, informative) < log_loss_of(balanced, [0.5] * 100)
    # ...but on a lopsided assay the ignorant constant wins on raw loss,
    lopsided_guess = [0.05] * 100
    weak = [0.04] * 95 + [0.06] * 5  # also ranks perfectly, just timidly
    assert log_loss_of(LOPSIDED, lopsided_guess) < log_loss_of(LOPSIDED, [0.5] * 100)
    # while skill correctly puts the discriminating model above the constant.
    assert classification_skill(LOPSIDED, weak)["log_loss_skill"] > \
        classification_skill(LOPSIDED, lopsided_guess)["log_loss_skill"]


def test_perfect_prediction_approaches_one():
    skill = classification_skill(LOPSIDED, [float(v) for v in LOPSIDED])["log_loss_skill"]
    assert 0.99 < skill <= 1


def test_ranking_without_calibration_can_still_score_below_zero():
    """Skill is not AUROC: a perfectly ranked but wildly overconfident model loses."""
    over = [0.5] * 95 + [0.51] * 5
    assert classification_skill(LOPSIDED, over)["log_loss_skill"] < 0


def test_single_class_assay_has_no_skill():
    assert classification_skill([0] * 10, [0.1] * 10)["log_loss_skill"] is None
    assert classification_skill([1] * 10, [0.9] * 10)["log_loss_skill"] is None


def test_classification_skill_reports_its_reference():
    result = classification_skill(LOPSIDED, [0.5] * 100)
    assert result["base_rate"] == pytest.approx(0.05)
    assert result["reference_log_loss"] == pytest.approx(log_loss_of(LOPSIDED, [0.05] * 100))


def test_regression_median_reference_scores_zero():
    values = [1., 2., 3., 10.]
    assert regression_skill(values, [np.median(values)] * 4)["mae_skill"] == pytest.approx(0)


@pytest.mark.parametrize("constant", [0., 1., 2.5, 4., 10., 100.])
def test_no_constant_regressor_beats_the_median(constant):
    values = [1., 2., 3., 10.]
    assert regression_skill(values, [constant] * 4)["mae_skill"] <= 1e-12


def test_perfect_regression_scores_one():
    values = [1., 2., 3., 10.]
    assert regression_skill(values, values)["mae_skill"] == pytest.approx(1)


def test_regression_without_spread_has_no_skill():
    assert regression_skill([3., 3., 3.], [3., 3., 3.])["mae_skill"] is None


def test_skill_is_dimensionless_so_rescaling_the_target_does_not_change_it():
    """Why classification and regression skills can be macro-averaged together."""
    values = np.array([1., 2., 3., 10.])
    predictions = np.array([1.5, 2.5, 2.5, 8.])
    base = regression_skill(values, predictions)["mae_skill"]
    assert regression_skill(values * 1000, predictions * 1000)["mae_skill"] == pytest.approx(base)


def test_macro_skill_skips_undefined_questions():
    assert macro_skill([0.2, None, 0.4]) == pytest.approx(0.3)
    assert macro_skill([None, None]) is None
    assert macro_skill([]) is None


def test_a_constant_predictor_scores_exactly_zero_rank_skill():
    """The guarantee skill alone could not give: no credit for ordering nothing."""
    assert rank_skill("classification", LOPSIDED, [0.05] * 100) == 0.
    assert rank_skill("regression", [1., 2., 3., 10.], [4.] * 4) == 0.


def test_rank_skill_is_invariant_to_overconfidence():
    """Why selection uses it: monotone rescaling cannot change the ordering."""
    labels = [0] * 50 + [1] * 50
    timid = [0.49] * 50 + [0.51] * 50
    brash = [0.001] * 50 + [0.999] * 50
    assert rank_skill("classification", labels, timid) == rank_skill("classification", labels, brash) == 1.


def test_rank_and_loss_skill_disagree_for_an_overconfident_model():
    """The inversion found on strict ToxCast, reproduced in miniature.

    An overconfident but perfectly ranking model loses to an ignorant constant on
    loss based skill, and beats it on rank skill. Selection must use the latter.
    """
    labels = [0] * 95 + [1] * 5
    overconfident = [0.55] * 95 + [0.6] * 5   # ranks perfectly, calibrated terribly
    constant = [0.05] * 100                   # the base rate, orders nothing
    assert classification_skill(labels, overconfident)["log_loss_skill"] < \
        classification_skill(labels, constant)["log_loss_skill"]
    assert rank_skill("classification", labels, overconfident) > \
        rank_skill("classification", labels, constant)


def test_perfect_and_inverted_orderings_bracket_the_rank_scale():
    assert rank_skill("classification", LOPSIDED, [float(v) for v in LOPSIDED]) == 1.
    assert rank_skill("classification", LOPSIDED, [1 - float(v) for v in LOPSIDED]) == -1.


def test_rank_skill_is_none_when_the_target_has_no_spread():
    assert rank_skill("classification", [1] * 10, list(range(10))) is None
    assert rank_skill("regression", [3.] * 5, [1., 2., 3., 4., 5.]) is None


def test_rank_skill_puts_both_families_on_one_scale():
    """Chance is 0 for each, which is what makes the macro average meaningful."""
    rng = np.random.default_rng(0)
    labels = rng.permutation([0] * 400 + [1] * 400)
    unrelated = np.linspace(0, 1, 800)  # ordering carries no information about labels
    assert abs(rank_skill("classification", labels, unrelated)) < 0.1
    values = rng.permutation(np.arange(800.))
    assert abs(rank_skill("regression", values, unrelated)) < 0.1
    # ...and a reversed ordering is the opposite end of the same scale, not chance.
    assert rank_skill("regression", [1., 2., 3., 4.], [4., 3., 2., 1.]) == -1.
