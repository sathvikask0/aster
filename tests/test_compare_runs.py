"""The comparison script must refuse to average incomparable runs."""

from __future__ import annotations

import json

from scripts.compare_amp_runs import comparability


def run(seed=42, policy="matched", label_version=2, esm="8M", results=None):
    return {
        "config": {
            "seed": seed, "negative_policy": policy, "balance_tasks": True,
            "esm": esm, "min_samples": 400, "label_semantics_version": label_version,
        },
        "results": results or {},
    }


def test_a_run_without_label_semantics_is_excluded():
    ok, problems = comparability([("old.json", run(label_version=None)),
                                  ("new.json", run())])
    assert [n for n, _ in ok] == ["new.json"]
    assert any("label_semantics_version" in p for p in problems)


def test_differing_negative_policy_is_flagged():
    ok, problems = comparability([("a.json", run(policy="matched")),
                                  ("b.json", run(policy="random"))])
    assert len(ok) == 2
    assert any("negative_policy" in p for p in problems)


def test_differing_encoder_is_flagged():
    _, problems = comparability([("a.json", run(esm="8M")),
                                 ("b.json", run(esm="35M"))])
    assert any("esm" in p for p in problems)


def test_matching_runs_have_no_complaints():
    ok, problems = comparability([("a.json", run(seed=42)), ("b.json", run(seed=43))])
    assert len(ok) == 2 and problems == []


def test_the_stored_v03_reports_are_rejected(tmp_path):
    """The real files in reports/ predate the label fix; this is the guard."""
    from pathlib import Path

    paths = sorted(Path("reports").glob("amp_v03_*.json"))
    if not paths:
        return
    runs = [(p.name, json.loads(p.read_text())) for p in paths]
    ok, problems = comparability(runs)
    assert ok == []
    assert len(problems) == len(paths)
