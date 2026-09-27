"""Re-derive the shortcut ceilings of an existing benchmark result, without ESM.

The v0.2 run reported lift against an *in-sample* composition probe. Model
accuracies do not depend on that baseline, so the correction can be applied to
the stored run without re-embedding 4,000 peptides: rebuild the exact held-out
split from the recorded config, recompute the ceiling out of fold, and rewrite
the lifts.

    python scripts/recompute_ceilings.py reports/amp_multitask_results.json \
        --max-peptides 4000 --min-samples 400 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aster.real.amp import load_amp_benchmark
from aster.real.ceilings import binomial_ci95, composition_ceiling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--max-peptides", type=int, default=4000)
    ap.add_argument("--min-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-policy", default="random", choices=["random", "covered"])
    args = ap.parse_args()

    examples, _ = load_amp_benchmark(
        min_samples=args.min_samples,
        seed=args.seed,
        negative_policy=args.negative_policy,
    )
    unique = sorted({e.sequence for e in examples})
    if args.max_peptides and len(unique) > args.max_peptides:
        rng = np.random.default_rng(42)
        kept = set(rng.choice(unique, size=args.max_peptides, replace=False))
        examples = [e for e in examples if e.sequence in kept]

    test = [e for e in examples if e.split == "test"]
    report = json.loads(args.results.read_text())
    held_out = report["held_out_tasks"]

    ceilings = {}
    for task in held_out:
        sub = [e for e in test if e.task == task]
        if not sub:
            raise SystemExit(f"Split mismatch: no held-out rows for {task}")
        labels = np.array([e.label for e in sub])
        ceilings[task] = composition_ceiling(
            [e.sequence for e in sub], labels, folds=5, seed=args.seed
        )
        old = report["ceilings"].get(task, {})
        c = ceilings[task]
        print(
            f"{task:16s} n={c['n']:5d} old(in-sample)={old.get('composition', float('nan')):.3f} "
            f"-> out-of-fold={c['composition_oof']:.3f} ceiling={c['ceiling']:.3f}"
        )

    for name, res in report["results"].items():
        for task, entry in res["per_task"].items():
            n = ceilings[task]["n"]
            acc = entry["acc"]
            lift = acc - ceilings[task]["ceiling"]
            entry["acc_ci95"] = binomial_ci95(acc, n)
            entry["lift"] = lift
            entry["lift_is_significant"] = bool(
                abs(lift) > entry["acc_ci95"] + ceilings[task]["ceiling_ci95"]
            )
        floor = report["results"]["task_id"]["overall_accuracy"]
        res["overall_lift_vs_task_id"] = res["overall_accuracy"] - floor
        res["mean_lift_vs_ceiling"] = float(
            np.mean([e["lift"] for e in res["per_task"].values()])
        )
        res["tasks_above_ceiling"] = int(sum(e["lift"] > 0 for e in res["per_task"].values()))

    report["ceilings"] = ceilings
    report["reporting_rule"] = (
        "Every claim is a margin over 'ceiling' (max of majority and the out-of-fold "
        "composition probe), never over task_id. A lift smaller than the combined 95% "
        "intervals is not a result."
    )
    report.setdefault("config", {}).update(
        {
            "command": "python scripts/run_amp_multitask.py --esm 8M --max-peptides 4000",
            "esm": "8M",
            "max_peptides": args.max_peptides,
            "min_samples": args.min_samples,
            "seed": args.seed,
            "negative_policy": args.negative_policy,
            "n_test": len(test),
            "ceilings_recomputed_by": "scripts/recompute_ceilings.py",
        }
    )
    args.results.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nRewrote {args.results}")


if __name__ == "__main__":
    main()
