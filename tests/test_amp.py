"""AMP source activity must select the corresponding answer text in training."""

import numpy as np
import pytest

from aster.real.amp import load_amp_benchmark
from aster.real.evaluate import build_tensor_dict


@pytest.fixture
def amp_source(tmp_path):
    # Each held-out peptide has the same composition as a training peptide,
    # so this fixture also supplies matches under the matched-negative policy.
    records = {
        "training": {"ACDEFG", "CDEFGH"},
        "held_out": {"GFEDCA", "HGFEDC"},
    }
    rows = ["sequence,pathogen"]
    for task, sequences in records.items():
        rows.extend(f"{sequence},{task}" for sequence in sorted(sequences))
    (tmp_path / "peptide_pathogen_triple.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "pathogen_description.csv").write_text(
        "pathogen,types,description\n"
        "training,Gram-negative,Training pathogen.\n"
        "held_out,Gram-positive,Held-out pathogen.\n"
    )
    return tmp_path, records


@pytest.mark.parametrize("policy", ["random", "covered", "matched", "scrambled"])
@pytest.mark.parametrize("balanced", [False, True])
def test_activity_selects_correct_answer_through_training_tensors(amp_source, policy, balanced):
    data_dir, records = amp_source
    examples, meta = load_amp_benchmark(
        data_dir=data_dir,
        min_samples=1,
        test_tasks=("held_out",),
        negative_policy=policy,
        min_coverage=1,
        balance_tasks=balanced,
    )
    assert len(examples) == 8
    assert {e.label for e in examples} == {0, 1}

    positive_answer = "it potently disrupts the membrane barrier and inhibits growth"
    negative_answer = "it fails to disrupt the microbial membrane"
    answer_vectors = {
        negative_answer: np.array([1.0, 0.0], dtype=np.float32),
        positive_answer: np.array([0.0, 1.0], dtype=np.float32),
    }
    q = {task: np.zeros(2, dtype=np.float32) for task in meta}
    a = {
        (task, index): answer_vectors[answer]
        for task, spec in meta.items()
        for index, answer in enumerate(spec["options"])
    }
    batch = build_tensor_dict(
        examples, np.zeros((len(examples), 2), dtype=np.float32), q, a,
        {"training": 1}, "cpu",
    )
    for row, example in enumerate(examples):
        active = example.sequence in records[example.task]
        expected_answer = positive_answer if active else negative_answer
        assert example.label == int(active)
        assert example.options[example.label] == expected_answer
        target = batch["y"][row].item()
        np.testing.assert_array_equal(
            batch["a_emb"][row, target].numpy(), answer_vectors[expected_answer]
        )


@pytest.mark.parametrize("balanced", [False, True])
def test_scrambled_negatives_are_decoys_not_library_peptides(amp_source, balanced):
    """The positive control's guarantees, asserted rather than assumed."""
    data_dir, records = amp_source
    real = records["training"] | records["held_out"]
    examples, meta = load_amp_benchmark(
        data_dir=data_dir, min_samples=1, test_tasks=("held_out",),
        negative_policy="scrambled", min_coverage=1, balance_tasks=balanced,
    )
    positives = [e.sequence for e in examples if e.label == 1]
    negatives = [e.sequence for e in examples if e.label == 0]
    assert positives and negatives

    # A decoy is never a real peptide, so it cannot silently be a true positive.
    assert not set(negatives) & real
    # No sequence carries both labels, which is what makes the matched policy
    # unlearnable; see the negative_policy docstring.
    assert not set(negatives) & set(positives)
    assert len(set(negatives)) == len(negatives)
    # Composition and length are preserved exactly, so both shortcut ceilings
    # are 0.5 by construction rather than by empirical hope.
    assert sorted(sorted(s) for s in negatives) == sorted(sorted(s) for s in positives)
    for spec in meta.values():
        assert spec["negatives_are_synthetic_decoys"] is True
        assert spec["negatives_are_presumed"] is False


def test_scrambled_negatives_skip_sequences_with_no_valid_shuffle():
    """A homopolymer has no distinct permutation, so it contributes no decoy."""
    import numpy as np
    from aster.real.amp import scrambled_negatives

    rng = np.random.default_rng(0)
    assert scrambled_negatives(["AAAA"], rng, {"AAAA"}) == []
    out = scrambled_negatives(["ACDEFG"], rng, {"ACDEFG"})
    assert len(out) == 1 and sorted(out[0]) == sorted("ACDEFG") and out[0] != "ACDEFG"
