"""Compare AMP report JSONs: across seeds, across policies, frozen vs fine-tuned.

One run gives one number and no idea how much of it is the seed. This reads
several report files and prints, per model, the spread across runs and whether a
lift over the ceiling survives everywhere -- which is the difference between a
result and a draw from a distribution nobody measured.

It also refuses to compare runs that are not comparable: a report without
`label_semantics_version` predates the answer-label fix, and one produced under
a different negative policy or a different architecture version is measuring
something else. Those are reported as incomparable rather than averaged in.

  uv run python scripts/compare_amp_runs.py reports/amp_finetune_s42.json \\
      reports/amp_finetune_s43.json reports/amp_finetune_s44.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(paths):
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append((Path(p).name, json.load(f)))
    return runs


def comparability(runs):
    """Split the runs into a comparable set and a list of complaints."""
    ok, problems = [], []
    for name, run in runs:
        cfg = run.get("config", {})
        if cfg.get("label_semantics_version") is None:
            problems.append(
                f"{name}: no label_semantics_version -- predates the answer-label "
                f"fix, so its answer-reading models were trained on inverted options"
            )
            continue
        ok.append((name, run))

    if len(ok) > 1:
        for key in ("negative_policy", "balance_tasks", "esm", "min_samples",
                    "text_encoder_trainable", "text_encoder"):
            values = {name: ok_run["config"].get(key) for name, ok_run in ok}
            if len(set(values.values())) > 1:
                problems.append(f"differing {key}: {values}")
    return ok, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    runs = load(args.reports)
    ok, problems = comparability(runs)

    if problems:
        print("--- Not comparable ---")
        for p in problems:
            print(f"  ! {p}")
        print()
    if not ok:
        print("Nothing comparable to report.")
        return
    if len(ok) == 1:
        print(f"Only one comparable run ({ok[0][0]}). Seed spread is unmeasured: "
              f"re-run with --seed 43, 44 into separate --out files.")

    seeds = sorted({run["config"].get("seed") for _, run in ok})
    print(f"{len(ok)} comparable run(s), seeds {seeds}")

    models = sorted({m for _, run in ok for m in run.get("results", {})})
    summary = {}
    print(f"\n{'model':24s} {'acc mean':>9s} {'spread':>8s} {'min':>7s} {'max':>7s}"
          f"  {'lifts>0 everywhere':>19s}")
    for model in models:
        accs, all_positive, per_task_lifts = [], True, {}
        for _, run in ok:
            res = run["results"].get(model)
            if res is None:
                continue
            accs.append(res["overall_accuracy"])
            for task, pt in res.get("per_task", {}).items():
                per_task_lifts.setdefault(task, []).append(pt["lift"])
                if pt["lift"] <= 0:
                    all_positive = False
        if not accs:
            continue
        spread = (max(accs) - min(accs)) if len(accs) > 1 else 0.0
        summary[model] = {
            "n_runs": len(accs),
            "accuracy_mean": statistics.fmean(accs),
            "accuracy_spread": spread,
            "accuracy_min": min(accs),
            "accuracy_max": max(accs),
            "lift_positive_in_every_run": all_positive,
            "per_task_lift_mean": {t: statistics.fmean(v) for t, v in per_task_lifts.items()},
        }
        print(f"{model:24s} {statistics.fmean(accs):9.3f} {spread:8.3f} "
              f"{min(accs):7.3f} {max(accs):7.3f}  {str(all_positive):>19s}")

    # A lift smaller than the seed spread is not a finding about the model.
    print("\n--- Lift vs seed spread ---")
    for model, s in summary.items():
        worst = min(s["per_task_lift_mean"].values(), default=0.0)
        best = max(s["per_task_lift_mean"].values(), default=0.0)
        verdict = "OK" if best > s["accuracy_spread"] else "lift is within seed noise"
        print(f"  {model:24s} mean lift {worst:+.3f}..{best:+.3f} "
              f"| seed spread {s['accuracy_spread']:.3f}  -> {verdict}")

    # Fine-tuning deltas, when a run carries both halves.
    # Both-towers-live runs answer a different question than protein-only ones.
    towers = {name: run["config"].get("text_encoder_trainable", False)
              for name, run in ok}
    if len(set(towers.values())) > 1:
        print("\n  ! mixing protein-only and both-towers-live runs. A gain here "
              "cannot be attributed to either tower; compare them as a ladder "
              "(frozen -> protein -> both), one step at a time.")

    pairs = [("cross_attention_live", "cross_attention_frozen"),
             ("dual_live", "dual_frozen"),
             ("entity_only_live", "entity_only_frozen"),
             ("question_only_live", "question_only_frozen")]
    deltas = {live: [] for live, _ in pairs}
    for _, run in ok:
        for live, frozen in pairs:
            r = run.get("results", {})
            if live in r and frozen in r:
                deltas[live].append(r[live]["overall_accuracy"] - r[frozen]["overall_accuracy"])
    shown = {k: v for k, v in deltas.items() if v}
    if shown:
        print("\n--- Unfreezing delta (live - frozen) ---")
        for live, vals in shown.items():
            mean = statistics.fmean(vals)
            spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
            flag = "" if abs(mean) > spread else "  (within seed spread)"
            print(f"  {live:24s} {mean:+.3f} over {len(vals)} run(s), spread {spread:.3f}{flag}")
        hypothesis = shown.get("cross_attention_live") or shown.get("dual_live")
        if hypothesis:
            h = statistics.fmean(hypothesis)
            entity = shown.get("entity_only_live")
            if entity and statistics.fmean(entity) >= h - 1e-9:
                print("  ! entity_only gained at least as much as the hypothesis model: "
                      "unfreezing bought peptide classification, not question use.")
            question = shown.get("question_only_live")
            if question and statistics.fmean(question) >= h - 1e-9:
                print("  ! question_only gained at least as much as the hypothesis "
                      "model: the text tower learned the label prior, not the question.")

    # Mechanism swap, aggregated.
    swap_rows = {}
    for _, run in ok:
        for model, res in run.get("results", {}).items():
            ms = res.get("mechanism_swap")
            if ms:
                swap_rows.setdefault(model, []).append(ms["drop"])
    if swap_rows:
        print("\n--- Mechanism swap (own - wrong-family prompt) ---")
        for model, drops in swap_rows.items():
            mean = statistics.fmean(drops)
            print(f"  {model:24s} drop {mean:+.3f} over {len(drops)} run(s)"
                  f"{'   <- prompt is a task id in English' if abs(mean) < 0.01 else ''}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"runs": [n for n, _ in ok], "incomparable": problems,
                       "summary": summary}, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
