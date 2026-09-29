"""Held-out questions must stay unseen by gradients, IDs, and target scalers."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("rdkit")
from aster.real import peptiverse as pv
from aster.real import peptiverse_transfer as transfer
from aster.real.peptiverse_evaluation import train_baselines


@pytest.fixture
def rows():
    records = []
    for task_index, (task, spec) in enumerate(pv.TASKS.items()):
        data = []
        for i in range(12):
            length = task_index * 12 + i + 1
            item = {spec["smiles"]: "C" * length + "O",
                    spec["label"]: i % 2 if spec["kind"] == "classification" else (i + 1.) / 3}
            if "sequence" in spec:
                item[spec["sequence"]] = "A" * length
            if "target" in spec:
                item[spec["target"]] = "MAKW"
            data.append(item)
        normalized, _ = pv.normalize_frame(task, pd.DataFrame(data))
        records.extend(normalized)
    return pv.assign_splits(records)


def test_question_splits_are_exhaustive_and_disjoint(rows):
    assigned, audit = transfer.hold_out_questions(rows, transfer.DEFAULT_PLAN)
    assert audit["excluded_rows"] == 0
    for split, tasks in transfer.DEFAULT_PLAN.items():
        assert {r["task"] for r in assigned if r["split"] == split} == set(tasks)
    assert audit["checks"]["cross_split_identity_overlap"] == 0
    with pytest.raises(ValueError, match="Held-out question"):
        broken = copy.deepcopy(assigned)
        broken[0]["split"] = "test" if broken[0]["split"] != "test" else "train"
        pv.validate_benchmark(broken, task_splits=transfer.task_assignment(transfer.DEFAULT_PLAN))


def test_shared_identity_is_removed_from_lower_priority_questions(rows):
    shared = [next(r for r in rows if r["task"] == t) for t in ("solubility", "hemolysis", "toxicity")]
    for r in shared:
        r["group_id"] = "same-group"
    assigned, audit = transfer.hold_out_questions(rows, transfer.DEFAULT_PLAN)
    assert {r["task"] for r in assigned if r["group_id"] == "same-group"} == {"toxicity"}
    assert audit["excluded_shared_identity_rows"] == {"hemolysis": 1, "solubility": 1}
    mutated = [{**r, "label": r["label"] + 1} if r["kind"] == "regression" else r for r in rows]
    again, other = transfer.hold_out_questions(mutated, transfer.DEFAULT_PLAN)
    assert {r["id"] for r in again} == {r["id"] for r in assigned}


def test_question_plan_rejects_duplicates_and_missing_questions():
    bad = copy.deepcopy(transfer.DEFAULT_PLAN)
    bad["train"].append("toxicity")
    with pytest.raises(ValueError, match="exactly one"):
        transfer.task_assignment(bad)
    bad = copy.deepcopy(transfer.DEFAULT_PLAN)
    bad["test"] = []
    with pytest.raises(ValueError, match="nonempty"):
        transfer.task_assignment(bad)


def test_target_scaler_cannot_learn_test_or_validation_values(rows):
    rows, _ = transfer.hold_out_questions(rows, transfer.DEFAULT_PLAN)
    scale = transfer.fit_regression_scale(rows)
    changed = [{**r, "label": 1e12} if r["split"] != "train" else r for r in rows]
    assert scale == transfer.fit_regression_scale(changed)


@pytest.mark.parametrize("mode", transfer.MODES)
def test_model_controls_read_only_allowed_inputs(mode):
    torch.manual_seed(0)
    model = transfer.TransferModel(8, 6, 4, mode, 8).eval()
    x, q, ids = torch.randn(3, 8), torch.randn(3, 6), torch.zeros(3, dtype=torch.long)
    original = model(x, q, ids)
    if mode in ("entity_only", "task_id"):
        torch.testing.assert_close(original, model(x, q + 2, ids))
    else:
        assert not torch.allclose(original, model(x, q + 2, ids))
    if mode == "question_only":
        torch.testing.assert_close(original, model(x + 2, q, ids))
    else:
        assert not torch.allclose(original, model(x + 2, q, ids))
    assert torch.equal(model.task.weight[0], torch.zeros(8))


def test_prepare_train_evaluate_without_using_test_for_selection(tmp_path, monkeypatch, rows):
    original = {"version": pv.VERSION, "tasks": pv.TASKS, "files": {"fixture": "test"},
                "split_policy": "global_identity", "limitations": []}
    with monkeypatch.context() as m:
        m.setattr(transfer, "load_benchmark", lambda _: (rows, original))
        transfer.build_transfer(tmp_path / "source", tmp_path / "data")
    data, manifest = pv.load_benchmark(tmp_path / "data")
    assert set(manifest["task_splits"]) == set(pv.TASKS)
    with pytest.raises(ValueError, match="unseen questions"):
        train_baselines(tmp_path / "data", tmp_path / "bad", tmp_path / "cache")
    calls = []

    def fake_text(names, revision=None):
        calls.append(names)
        vectors = np.array([[i == list(pv.TASKS).index(t) for i in range(8)] for t in names], dtype=np.float32)
        return vectors, "fixture-revision"

    monkeypatch.setattr(transfer, "encode_questions", fake_text)
    run = tmp_path / "run"
    result = transfer.train_transfer(tmp_path / "data", run, tmp_path / "cache", seeds=(42,), epochs=2, batch_size=8, hidden=8)
    assert not (set(calls[0]) & set(transfer.DEFAULT_PLAN["test"]))
    assert result["protocol"]["test_evaluated"] is False
    assert len(result["runs"]) == 4
    for record in result["runs"].values():
        assert set(record["validation"]["tasks"]) == set(transfer.DEFAULT_PLAN["validation"])
    assert not (run / "test_evaluation.json").exists()
    evaluated = transfer.evaluate_transfer(tmp_path / "data", run, tmp_path / "cache")
    for record in evaluated["runs"].values():
        assert set(record["tasks"]) == set(transfer.DEFAULT_PLAN["test"])
    with pytest.raises(FileExistsError, match="already evaluated"):
        transfer.evaluate_transfer(tmp_path / "data", run, tmp_path / "cache")
