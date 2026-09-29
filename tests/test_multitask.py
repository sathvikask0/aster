"""Held-out assays must stay out of training, and splits must be provable."""
import json

import numpy as np
import pytest

from aster.real import multitask as mt


def rows_for(assays, molecules=6):
    out = []
    for a, assay in enumerate(assays):
        for m in range(molecules):
            out.append({"task": assay, "kind": "classification",
                        "question": mt.assay_question(assay),
                        "options": ["inactive", "active"],
                        "label": (a + m) % 2, "units": "binary",
                        "smiles": "C" * (m + 1) + "O",
                        "molecule_id": mt.digest(f"m{m}"),
                        "source_row": m,
                        "id": mt.digest(f"{assay}:{m}")})
    return out


def test_assay_plan_is_deterministic_and_populates_every_split():
    assays = [f"A{i}" for i in range(40)]
    plan = mt.assay_plan(assays)
    assert plan == mt.assay_plan(assays)
    assert set(plan.values()) == set(mt.SPLITS)
    assert set(plan) == set(assays)
    with pytest.raises(ValueError, match="at least three"):
        mt.assay_plan(["only", "two"])
    with pytest.raises(ValueError, match="summing to one"):
        mt.assay_plan(assays, fractions=(0.5, 0.4, 0.4))


def test_no_assay_appears_in_two_splits():
    assays = [f"A{i}" for i in range(30)]
    plan = mt.assay_plan(assays)
    rows, _ = mt.split_rows(rows_for(assays), plan)
    checks = mt.validate(rows, plan)
    seen = {}
    for row in rows:
        seen.setdefault(row["task"], set()).add(row["split"])
    assert all(len(v) == 1 for v in seen.values())
    assert sum(checks[s]["assays"] for s in mt.SPLITS) == len(assays)


def test_leaked_assay_is_rejected():
    assays = [f"A{i}" for i in range(30)]
    plan = mt.assay_plan(assays)
    rows, _ = mt.split_rows(rows_for(assays), plan)
    rows[0] = {**rows[0], "split": "test" if rows[0]["split"] != "test" else "train"}
    with pytest.raises(ValueError, match="leaked"):
        mt.validate(rows, plan)


def test_molecule_disjoint_mode_separates_molecules():
    assays = [f"A{i}" for i in range(30)]
    plan = mt.assay_plan(assays)
    shared, _ = mt.split_rows(rows_for(assays, molecules=40), plan)
    strict, audit = mt.split_rows(rows_for(assays, molecules=40), plan, molecule_disjoint=True)
    assert audit["molecule_disjoint"] is True
    assert len(strict) < len(shared)
    assert mt.validate(strict, plan)["molecules_in_more_than_one_split"] == 0
    assert mt.validate(shared, plan)["molecules_in_more_than_one_split"] > 0


def test_questions_differ_per_assay_and_describe_the_assay():
    a = mt.assay_question("APR_HepG2_CellCycleArrest_24h_dn")
    b = mt.assay_question("APR_HepG2_CellCycleArrest_24h_up")
    assert a != b and "decreased" in a and "increased" in b
    assert mt.assay_question("NR-AR").startswith("Does this compound")
    assert len({mt.assay_question(f"X_{i}") for i in range(20)}) == 20


def test_scoring_reports_per_assay_and_macro_numbers():
    rows = rows_for(["A0", "A1"], molecules=8)
    probs = np.full(len(rows), 0.5)
    score = mt.score_by_assay(rows, probs)
    assert score["assays_scored"] == 2
    assert np.isclose(score["macro_log_loss"], np.log(2), atol=1e-6)
    assert np.isclose(score["macro_auroc"], 0.5, atol=1e-9)


def test_duplicate_and_conflicting_pairs_are_handled(tmp_path):
    # validate() must reject a duplicate id outright.
    assays = [f"A{i}" for i in range(30)]
    plan = mt.assay_plan(assays)
    rows, _ = mt.split_rows(rows_for(assays), plan)
    with pytest.raises(ValueError, match="Duplicate"):
        mt.validate(rows + [rows[0]], plan)


def test_round_trip_through_disk_verifies_checksums(tmp_path, monkeypatch):
    assays = [f"A{i}" for i in range(30)]
    plan = mt.assay_plan(assays)
    rows, split_audit = mt.split_rows(rows_for(assays), plan)
    out = tmp_path / "ds"
    out.mkdir()
    files = {}
    for split in mt.SPLITS:
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == split))
        files[path.name] = mt.file_digest(path)
    (out / "manifest.json").write_text(json.dumps({
        "version": mt.VERSION, "files": files, "assay_plan": plan,
        "assay_splits": {s: sorted(a for a, v in plan.items() if v == s) for s in mt.SPLITS},
        "source": "fixture"}))
    loaded, manifest = mt.load_multitask(out)
    assert len(loaded) == len(rows)
    (out / "test.jsonl").write_text("")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        mt.load_multitask(out)
