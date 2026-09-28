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


@pytest.mark.parametrize("policy", ["random", "covered", "matched"])
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
