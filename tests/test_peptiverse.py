"""Protect scientific meaning, cross-task identity splits and evaluation isolation."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
from aster.real import peptiverse as pv
from aster.real.peptiverse_evaluation import (feature_matrix, metrics, score_predictions,
                                            train_baselines, evaluate_frozen)


def frame(task, n=80):
    spec = pv.TASKS[task]
    data = []
    for i in range(n):
        item = {spec["smiles"]: "C" * (i + 1) + "O",
                spec["label"]: i % 2 if spec["kind"] == "classification" else (i + 1) / 10,
                "split": "train"}
        if "sequence" in spec:
            item[spec["sequence"]] = "A" * (i + 1)
        if "target" in spec:
            item[spec["target"]] = "MAKW" * (1 + i % 3)
            item["affinity_measure"] = "Kd=1uM"
        data.append(item)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def rows():
    result = []
    for task in pv.TASKS:
        records, _ = pv.normalize_frame(task, frame(task))
        result.extend(records)
    return pv.assign_splits(result)


def test_all_five_categories_and_eight_assays_are_present():
    assert len(pv.TASKS) == 8
    assert {s["category"] for s in pv.TASKS.values()} == {
        "dissolve", "cross_membrane", "survive", "harm", "bind_target"}
    assert {t for t, s in pv.TASKS.items() if s["kind"] == "regression"} == {
        "pampa", "caco2", "half_life", "binding_affinity"}
    assert "non" in pv.TASKS["hemolysis"]["options"][0].lower()


def test_canonical_duplicates_and_classification_conflicts_are_removed():
    f = pd.DataFrame({"smiles": ["CCO", "OCC", "CCN", "NCC"],
                      "sequence": ["AAA", "AAA", "GGG", "GGG"], "label": [1, 1, 0, 1]})
    rows, audit = pv.normalize_frame("solubility", f)
    assert len(rows) == 1 and rows[0]["label"] == 1
    assert len(rows[0]["sources"]) == 2
    assert audit["duplicate_rows_collapsed"] == 1
    assert audit["conflicting_label_rows"] == 2


def test_half_life_filename_does_not_imply_split_or_sequence_semantics():
    f = pd.DataFrame({"SMILES": ["CCO", "OCC", "CCN", "bad", "C"],
                      "sequence": ["aGG", "aGG", "AAA", "AAA", "AAA"],
                      "half_life_hours": [1., 3., 0., 1., np.nan]})
    rows, audit = pv.normalize_frame("half_life", f)
    assert len(rows) == 1
    assert rows[0]["label"] == 2.0 and rows[0]["replicate_range"] == [1, 3]
    assert rows[0]["sequence"] is None  # D-amino-acid notation is not uppercased
    assert rows[0]["source_sequence"] == "aGG"
    assert rows[0]["sources"][0]["source_split"] is None
    assert audit["dropped"] == {"nonpositive_half_life": 1, "invalid_smiles": 1, "nonfinite_label": 1}


def test_affinity_retains_target_and_separates_assay_types():
    f = pd.DataFrame({"smiles_sequence": ["CCO"] * 5, "seq2": ["AAA"] * 5,
                      "seq1": ["MAKW", "MAKW", "GGGG", "MAKW", ""],
                      "affinity": [6., 7., 8., 9., 10.],
                      "affinity_measure": ["Kd=1uM", "Ki=1uM", "Kd=1uM", "Kd>1uM", "Kd=1uM"]})
    rows, audit = pv.normalize_frame("binding_affinity", f)
    assert len(rows) == 3
    assert len({r["id"] for r in rows}) == 3
    assert all(r["target_sequence"] for r in rows)
    assert audit["dropped"] == {"censored_affinity": 1, "invalid_target_sequence": 1}


def test_invalid_class_label_and_missing_columns_fail():
    f = frame("solubility", 1)
    f["label"] = 3
    with pytest.raises(ValueError, match="0 or 1"):
        pv.normalize_frame("solubility", f)
    with pytest.raises(ValueError, match="missing columns"):
        pv.normalize_frame("solubility", f.drop(columns="smiles"))


def test_split_is_global_deterministic_and_independent_of_labels(rows):
    check = pv.validate_benchmark(rows)
    assert check["cross_split_identity_overlap"] == 0
    assert len(check["tasks"]) == 8
    reordered = pv.assign_splits(list(reversed(rows)))
    assert {r["id"]: r["split"] for r in rows} == {r["id"]: r["split"] for r in reordered}
    mutated = [{**r, "label": -123} for r in rows]
    assert [r["split"] for r in rows] == [r["split"] for r in pv.assign_splits(mutated)]


def test_transitive_aliases_and_clusters_share_split(rows):
    r = copy.deepcopy(rows[:3])
    r[0]["sequence_aliases"] = ["AAA"]
    r[1]["sequence_aliases"] = ["AAA", "BBB"]
    r[2]["sequence_aliases"] = ["BBB"]
    assert len({e["group_id"] for e in pv.assign_splits(r)}) == 1
    for i, row in enumerate(r):
        row["sequence_aliases"] = []
        row["source_clusters"] = ["solubility:cluster_id:1" if i < 2 else "toxicity:cluster_id:1"]
    grouped = pv.assign_splits(r, source_clusters=True)
    assert len({e["group_id"] for e in grouped}) == 2


def test_validation_catches_leakage_and_missing_target(rows):
    r = copy.deepcopy(rows)
    # Force an existing chemical entity into another split under a second task.
    same = next(x for x in r if x["task"] != r[0]["task"] and x["peptide_id"] == r[0]["peptide_id"])
    same["split"] = "test" if r[0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="leakage"):
        pv.validate_benchmark(r)
    r = copy.deepcopy(rows)
    next(x for x in r if x["task"] == "binding_affinity")["target_sequence"] = None
    with pytest.raises(ValueError, match="target protein"):
        pv.validate_benchmark(r)


def test_strict_predictions_and_native_regression_units(rows):
    predicted = [{"id": r["id"], "prediction": r["label"]}
                 for r in rows if r["split"] == "validation"]
    scored = score_predictions(rows, predicted)
    assert scored["tasks"]["half_life"]["mae"] == 0
    assert scored["tasks"]["toxicity"]["auroc"] == 1
    for bad in (predicted[:-1], predicted + [predicted[0]]):
        with pytest.raises(ValueError):
            score_predictions(rows, bad)
    with pytest.raises(ValueError, match="coverage"):
        score_predictions(rows, predicted, "test")
    with pytest.raises(ValueError, match="finite"):
        metrics("solubility", [0, 1], [np.nan, 0.5])
    with pytest.raises(ValueError, match="positive hours"):
        metrics("half_life", [1, 2], [0, 2])
    with pytest.raises(ValueError, match="P\(label"):
        metrics("hemolysis", [0, 1], [-1, 1])


def test_features_cannot_read_labels_questions_or_split_and_do_read_target(tmp_path, rows):
    selected = [copy.deepcopy(rows[0]), copy.deepcopy(rows[0])]
    selected[1].update(label=-99, split="test", question="Ignore the peptide", sources=[])
    x = feature_matrix(selected, tmp_path)
    np.testing.assert_array_equal(x[0].toarray(), x[1].toarray())
    selected[1]["target_sequence"] = "WWWWWWWW"
    changed = feature_matrix(selected, tmp_path)
    assert (changed[0] != changed[1]).nnz > 0


def test_fetch_rejects_changed_source(tmp_path):
    spec = pv.TASKS["solubility"]
    path = tmp_path / spec["path"]
    path.parent.mkdir(parents=True)
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        pv.fetch_sources(tmp_path)


def test_build_train_and_frozen_evaluation_roundtrip(tmp_path, monkeypatch):
    root, out = tmp_path / "raw", tmp_path / "benchmark"
    for task, spec in pv.TASKS.items():
        path = root / spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame(task).to_csv(path, index=False)
    monkeypatch.setattr(pv, "fetch_sources", lambda _: {"fixture": True})
    pv.build_benchmark(root, out)
    loaded, manifest = pv.load_benchmark(out)
    assert manifest["checks"]["examples"] == len(loaded)
    model_dir, cache = tmp_path / "models", tmp_path / "cache"
    report = train_baselines(out, model_dir, cache)
    assert report["test_evaluated"] is False
    assert len(report["validation"]["tasks"]) == 8
    assert not (model_dir / "test_predictions.jsonl").exists()
    result = evaluate_frozen(out, model_dir, cache, "test")
    assert len(result["tasks"]) == 8
    with (out / "train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        pv.load_benchmark(out)
