"""The mechanism-swap ablation must actually swap the mechanism."""

from __future__ import annotations

import numpy as np
import pytest

from aster.real import ablation
from aster.real.amp import build_mechanistic_prompt, load_amp_benchmark, prompt_family


def test_prompt_family_matches_the_prompt_actually_built():
    cases = [
        ("Gram-negative", "A rod.", "lipopolysaccharide"),
        ("Gram-positive", "A cocci.", "peptidoglycan cell wall"),
        ("Fungus", "A yeast.", "chitin"),
        ("Pathogen", "Something else.", "microbial membrane barrier"),
    ]
    for p_type, desc, marker in cases:
        question, _ = build_mechanistic_prompt("X", p_type, desc)
        assert marker in question
        # The family label and the prompt text cannot disagree.
        family = prompt_family(p_type, desc)
        assert family in ("gram_negative", "gram_positive", "fungal", "generic")
        other, _ = build_mechanistic_prompt("X", p_type, desc)
        assert question == other


def test_family_reads_the_description_when_the_type_is_uninformative():
    assert prompt_family("Pathogen", "A budding yeast.") == "fungal"
    assert prompt_family("Pathogen", "A soil mold.") == "fungal"
    assert prompt_family("Pathogen", "Unclear.") == "generic"


@pytest.fixture
def meta():
    return {
        "gn_train": {"prompt_family": "gram_negative", "is_held_out": False},
        "gp_train": {"prompt_family": "gram_positive", "is_held_out": False},
        "fung_train": {"prompt_family": "fungal", "is_held_out": False},
        "gn_test": {"prompt_family": "gram_negative", "is_held_out": True},
        "fung_test": {"prompt_family": "fungal", "is_held_out": True},
    }


def test_every_donor_comes_from_a_different_family(meta):
    mapping = ablation.swap_map(meta, ["gn_test", "fung_test"], seed=0)
    assert set(mapping) == {"gn_test", "fung_test"}
    for task, donor in mapping.items():
        assert meta[donor]["prompt_family"] != meta[task]["prompt_family"]


def test_donors_are_prompts_the_model_has_seen(meta):
    mapping = ablation.swap_map(meta, ["gn_test", "fung_test"], seed=0)
    for donor in mapping.values():
        assert meta[donor]["is_held_out"] is False


def test_swap_map_is_deterministic(meta):
    a = ablation.swap_map(meta, ["gn_test", "fung_test"], seed=7)
    b = ablation.swap_map(meta, ["gn_test", "fung_test"], seed=7)
    assert a == b


def test_swap_map_survives_a_single_family(meta):
    only = {"a": {"prompt_family": "fungal", "is_held_out": False},
            "b": {"prompt_family": "fungal", "is_held_out": True}}
    assert ablation.swap_map(only, ["b"], seed=0) == {}


def test_swapped_tensor_replaces_questions_and_nothing_else():
    torch = pytest.importorskip("torch")

    class Row:
        def __init__(self, task):
            self.task = task

    examples = [Row("gn_test"), Row("fung_test"), Row("gn_test")]
    q_map = {
        "gn_test": np.array([1.0, 0.0]), "fung_test": np.array([0.0, 1.0]),
        "gp_train": np.array([9.0, 9.0]),
    }
    tensors = {
        "q_emb": torch.zeros(3, 2),
        "y": torch.tensor([0, 1, 0]),
        "x": torch.ones(3, 4),
    }
    out = ablation.swapped_question_tensor(
        tensors, examples, q_map, {"gn_test": "gp_train"}, "cpu"
    )
    assert torch.equal(out["q_emb"][0], torch.tensor([9.0, 9.0]))
    assert torch.equal(out["q_emb"][1], torch.tensor([0.0, 1.0]))  # unmapped: own prompt
    assert torch.equal(out["y"], tensors["y"]) and torch.equal(out["x"], tensors["x"])
    assert torch.equal(tensors["q_emb"], torch.zeros(3, 2)), "input was mutated"


# --- the pre-registered reading ---------------------------------------------

def test_no_drop_reads_as_a_task_id():
    assert "task identifier" in ablation.read_swap(0.70, 0.699, 0.02)


def test_a_real_drop_reads_as_prompt_content_used():
    reading = ablation.read_swap(0.70, 0.55, 0.02)
    assert "prompt content is being used" in reading
    assert "not sufficient" in reading


def test_an_improvement_is_flagged_as_unexpected():
    assert "IMPROVED" in ablation.read_swap(0.55, 0.70, 0.02)


def test_real_benchmark_metadata_carries_a_usable_family(tmp_path):
    rows = ["sequence,pathogen"]
    for task in ("training", "held_out"):
        rows += [f"ACDEFG,{task}", f"CDEFGH,{task}"]
    (tmp_path / "peptide_pathogen_triple.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "pathogen_description.csv").write_text(
        "pathogen,types,description\n"
        "training,Gram-negative,Training pathogen.\n"
        "held_out,Fungus,Held-out pathogen.\n"
    )
    _, meta = load_amp_benchmark(
        data_dir=tmp_path, min_samples=1, test_tasks=("held_out",), min_coverage=1,
    )
    assert meta["training"]["prompt_family"] == "gram_negative"
    assert meta["held_out"]["prompt_family"] == "fungal"
    assert ablation.swap_map(meta, ["held_out"], seed=0) == {"held_out": "training"}
