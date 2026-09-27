"""The ceiling is the zero every claim is measured against, so it gets tests."""

from __future__ import annotations

import numpy as np
import pytest

from aster.real.amp import match_negatives
from aster.real.ceilings import binomial_ci95, composition_ceiling, composition_matrix


def _rng():
    return np.random.default_rng(0)


def _seqs(n, alphabet, rng, length=20):
    return ["".join(rng.choice(list(alphabet), size=length)) for _ in range(n)]


def test_composition_matrix_rows_are_frequencies():
    mat = composition_matrix(["AAAA", "ACDE"])
    assert np.allclose(mat.sum(1), 1.0)
    assert mat[0, composition_matrix(["A"]).argmax()] == 1.0


def test_out_of_fold_ceiling_is_at_chance_on_unlearnable_labels():
    """Labels independent of composition: an honest ceiling stays near majority."""
    rng = _rng()
    seqs = _seqs(400, "ACDEFGHIKLMNPQRSTVWY", rng)
    labels = rng.integers(0, 2, size=400)
    result = composition_ceiling(seqs, labels, seed=0)
    assert result["composition_oof"] < 0.60
    assert result["ceiling"] < 0.60


def test_in_sample_probe_overstates_the_shortcut():
    """The reason the ceiling is scored out of fold, pinned as a test."""
    rng = _rng()
    seqs = _seqs(200, "ACDEFGHIKLMNPQRSTVWY", rng)
    labels = rng.integers(0, 2, size=200)
    result = composition_ceiling(seqs, labels, seed=0)
    assert result["composition_insample"] > result["composition_oof"]


def test_ceiling_detects_a_real_composition_shortcut():
    rng = _rng()
    cationic = _seqs(200, "KKRRKRAL", rng)
    anionic = _seqs(200, "DDEEDESG", rng)
    labels = np.array([1] * 200 + [0] * 200)
    result = composition_ceiling(cationic + anionic, labels, seed=0)
    assert result["composition_oof"] > 0.95


def test_matching_collapses_a_composition_shortcut():
    """Matched negatives must leave the classes compositionally inseparable."""
    rng = _rng()
    positives = _seqs(150, "KKRRKRAL", rng)
    candidates = _seqs(300, "KKRRKRAL", rng) + _seqs(300, "DDEEDESG", rng)
    matched = match_negatives(positives, candidates)
    assert matched, "matching returned nothing"
    labels = np.array([1] * len(positives) + [0] * len(matched))
    result = composition_ceiling(positives + matched, labels, seed=0)
    assert result["composition_oof"] < 0.65


def test_match_negatives_respects_the_caliper_and_does_not_reuse():
    rng = _rng()
    positives = _seqs(50, "KKRR", rng)
    far = _seqs(50, "DDEE", rng)
    assert match_negatives(positives, far, caliper=0.01) == []
    matched = match_negatives(positives, far, caliper=10.0)
    assert len(matched) == len(set(matched))


def test_binomial_ci_shrinks_with_n():
    assert binomial_ci95(0.5, 100) > binomial_ci95(0.5, 10_000)
