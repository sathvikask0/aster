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


def _data_available():
    from aster.real.amp import DATA_DIR
    return (DATA_DIR / "peptide_pathogen_triple.csv").exists()


needs_data = pytest.mark.skipif(not _data_available(), reason="AMP triples not fetched")


@needs_data
def test_loader_reports_the_label_prior_it_actually_built():
    """A skewed training prior must be visible in metadata, not inferred later.

    E. coli has more positives than the library has peptides without an E. coli
    record, so its negatives are exhausted and its rows cannot be 1:1.
    """
    from collections import Counter

    from aster.real.amp import load_amp_benchmark

    examples, meta = load_amp_benchmark(min_samples=400, seed=42)
    for task, m in meta.items():
        counts = Counter(e.label for e in examples if e.task == task)
        assert m["n_positive"] == counts[1]
        assert m["n_negative"] == counts[0]
        assert m["label_prior"] == pytest.approx(counts[1] / (counts[1] + counts[0]))
        assert m["negatives_exhausted"] == (counts[0] < counts[1])

    assert meta["E.coli"]["negatives_exhausted"], "expected E.coli to run out of negatives"
    assert meta["E.coli"]["label_prior"] > 0.7


@needs_data
@pytest.mark.parametrize("policy", ["random", "covered", "matched"])
def test_balance_tasks_pairs_every_task(policy):
    """With balancing on, no task may arrive skewed -- under any policy."""
    from aster.real.amp import load_amp_benchmark

    _, meta = load_amp_benchmark(
        min_samples=400, seed=42, negative_policy=policy, balance_tasks=True
    )
    for task, m in meta.items():
        assert m["n_positive"] == m["n_negative"], f"{task} unbalanced under {policy}"
        assert not m["negatives_exhausted"]
