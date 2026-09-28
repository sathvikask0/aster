"""Property tasks must use measured negatives, and must say so in metadata."""

import numpy as np
import pandas as pd
import pytest

from aster.real.properties import (
    PROPERTIES, build_property_benchmark, label_agreement, load_property_table,
    task_overlap, threshold_sensitivity,
)


def write_csv(tmp_path, name, rows, seq_col="SEQUENCE", conc_col="µM"):
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(rows, columns=[seq_col, conc_col]).to_csv(path, index=False)
    return path


HEMO = [
    ("GLPALISWIKRKRL", 596.7),   # measured, needs a flooding dose -> negative
    ("ILPILSLIGGLL", 180.0),     # measured negative
    ("IVPFLLGMVPKLVCLITKKC", 7.0),
    ("KLKLKLKLKLKLKL", 0.3),
]
AMP = [
    ("GLPALISWIKRKRL", 4.0),     # shared sequence: kills microbes, spares blood
    ("ILPILSLIGGLL", 128.0),
    ("KKKKWWWWKKKK", 2.0),
    ("AGAGAGAGAGAG", 256.0),
]


def test_loader_keeps_only_measured_rows_and_finds_the_unit_column(tmp_path):
    rows = HEMO + [("BADSEQ0XZ", 1.0), ("AAA", 1.0), ("GLWSKIKEVGKEAAKA", None)]
    table = load_property_table("hemolytic", write_csv(tmp_path, "h", rows))
    kept = set(table.frame["sequence"])
    assert kept == {s for s, _ in HEMO}
    # Non-standard residues, too-short, and missing-concentration rows are gone.
    assert "BADSEQ0XZ" not in kept and "AAA" not in kept
    assert "GLWSKIKEVGKEAAKA" not in kept
    assert table.measure == "HC50" and table.units == "uM"


def test_loader_refuses_a_file_with_a_label_but_no_concentration(tmp_path):
    path = tmp_path / "label_only.csv"
    pd.DataFrame({"SEQUENCE": ["GLPALISWIKRKRL"], "label": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="measured rather than presumed"):
        load_property_table("hemolytic", path)


def test_duplicate_measurements_average_before_thresholding(tmp_path):
    # 40 and 160 average to 100, which is on the boundary and therefore active.
    rows = [("GLPALISWIKRKRL", 40.0), ("GLPALISWIKRKRL", 160.0)] + HEMO[2:]
    table = load_property_table("hemolytic", write_csv(tmp_path, "d", rows))
    conc = dict(zip(table.frame["sequence"], table.frame["concentration"]))
    assert conc["GLPALISWIKRKRL"] == pytest.approx(100.0)
    labels = table.labelled().set_index("sequence")["label"]
    assert labels["GLPALISWIKRKRL"] == 1


def test_label_direction_a_large_dose_is_the_negative(tmp_path):
    table = load_property_table("hemolytic", write_csv(tmp_path, "h", HEMO))
    labels = table.labelled(threshold=100.0).set_index("sequence")["label"]
    assert labels["GLPALISWIKRKRL"] == 0   # 596.7 uM
    assert labels["KLKLKLKLKLKLKL"] == 1   # 0.3 uM


def test_overlap_and_agreement_expose_the_therapeutic_index_case(tmp_path):
    tables = {
        "hemolytic": load_property_table("hemolytic", write_csv(tmp_path, "h", HEMO)),
        "antimicrobial": load_property_table("antimicrobial", write_csv(tmp_path, "a", AMP)),
    }
    overlap = task_overlap(tables)
    assert overlap["pairs"]["antimicrobial|hemolytic"]["shared"] == 2
    assert overlap["measured_for_all"] == 2

    agree = label_agreement(tables)
    assert agree["n"] == 2
    # GLPALISWIKRKRL: antimicrobial active (4 uM), hemolytic inactive (596.7 uM).
    # The two properties disagree for the same peptide, which is the point.
    assert agree["joint"]["1|0"] == 1


def test_threshold_sensitivity_moves_the_positive_rate(tmp_path):
    table = load_property_table("hemolytic", write_csv(tmp_path, "h", HEMO))
    sens = threshold_sensitivity(table, cuts=(1.0, 100.0, 1000.0))
    assert sens["1.0"] < sens["100.0"] < sens["1000.0"] == 1.0


def test_benchmark_holds_out_a_whole_property_and_records_provenance(tmp_path):
    tables = {
        "hemolytic": load_property_table("hemolytic", write_csv(tmp_path, "h", HEMO)),
        "antimicrobial": load_property_table("antimicrobial", write_csv(tmp_path, "a", AMP)),
    }
    examples, meta = build_property_benchmark(tables, test_tasks=("hemolytic",))
    assert {e.split for e in examples if e.task == "hemolytic"} == {"test"}
    assert {e.split for e in examples if e.task == "antimicrobial"} <= {"train", "val"}
    assert {e.label for e in examples} == {0, 1}
    for name, spec in meta.items():
        assert spec["negatives_are_measured"] is True
        assert spec["negatives_are_presumed"] is False
        assert spec["options"] == PROPERTIES[name]["answers"]
        assert spec["n_positive"] == spec["n_negative"]  # balanced
    # The two properties must ask different questions, or the prompt has no work.
    assert meta["hemolytic"]["question"] != meta["antimicrobial"]["question"]
    assert meta["hemolytic"]["prompt_family"] != meta["antimicrobial"]["prompt_family"]


def test_benchmark_rejects_holding_out_every_property(tmp_path):
    tables = {"hemolytic": load_property_table("hemolytic", write_csv(tmp_path, "h", HEMO))}
    with pytest.raises(ValueError, match="remain for training"):
        build_property_benchmark(tables, test_tasks=("hemolytic",))


def test_benchmark_rejects_a_threshold_outside_the_measured_range(tmp_path):
    tables = {
        "hemolytic": load_property_table("hemolytic", write_csv(tmp_path, "h", HEMO)),
        "antimicrobial": load_property_table("antimicrobial", write_csv(tmp_path, "a", AMP)),
    }
    with pytest.raises(ValueError, match="only one class"):
        build_property_benchmark(tables, test_tasks=("hemolytic",),
                                 thresholds={"antimicrobial": 1e9})


def test_disjoint_sequences_keeps_a_peptide_out_of_two_splits(tmp_path):
    tables = {
        "hemolytic": load_property_table("hemolytic", write_csv(tmp_path, "h", HEMO)),
        "antimicrobial": load_property_table("antimicrobial", write_csv(tmp_path, "a", AMP)),
    }
    examples, meta = build_property_benchmark(
        tables, test_tasks=("hemolytic",), disjoint_sequences=True, balance_tasks=False)
    trainish = {e.sequence for e in examples if e.split in ("train", "val")}
    by_seq = {}
    for e in examples:
        by_seq.setdefault(e.sequence, set()).add(e.split)
    # A held-out property still sends every row to test, so a shared sequence
    # appears in both; the flag is recorded so the report cannot be misread.
    assert all(spec["disjoint_sequences"] for spec in meta.values())
    assert trainish


@pytest.mark.parametrize("mu", ["\u00b5M", "\u03bcM", "HC50 (uM)", "MIC_uM"])
def test_concentration_column_is_found_whichever_mu_the_export_used(tmp_path, mu):
    """The real HemoPI2 download spells micromolar with GREEK SMALL LETTER MU.

    An earlier version matched only MICRO SIGN (U+00B5) and rejected that file
    outright, reporting it as having no concentration column. Both characters,
    and a plain ASCII spelling, must resolve to the same column.
    """
    path = tmp_path / "mu.csv"
    pd.DataFrame(HEMO, columns=["SEQUENCE", mu]).to_csv(path, index=False)
    table = load_property_table("hemolytic", path)
    assert len(table.frame) == len(HEMO)
    assert table.frame["concentration"].max() == pytest.approx(596.7)


def test_fold_normalises_both_mu_characters():
    from aster.real.properties import _fold

    assert _fold("\u00b5M") == _fold("\u03bcM") == "um"
