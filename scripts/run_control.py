"""The control experiment.

Run:  python scripts/run_control.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aster.control.synthetic import build
from aster.control import splits as S
from aster.control.harness import run_grid, verdict

SPLITS = ["random", "held_out_entity_family", "held_out_question", "held_out_both"]
REGIMES = ["semantic", "arbitrary"]


def fmt_row(r):
    c = r.test_calibrated
    return (f"  {r.mode:<15} acc={c.accuracy:.3f} lift={c.lift_over_chance:+.3f} "
            f"ece={c.ece:.3f} (raw {r.test.ece:.3f}) "
            f"overconf={c.overconfidence:+.3f} T={r.temperature:.2f} "
            f"n={c.n}")


def main():
    all_results, all_checks = {}, []
    passed = failed = 0

    for regime in REGIMES:
        examples, world = build(regime, seed=7)
        print(f"\n{'=' * 78}\nREGIME: {regime}   "
              f"({len(examples)} examples, {len(world.questions)} questions)")
        print({"semantic": "  text compositionally determines the rule "
                           "-> transfer is POSSIBLE",
               "arbitrary": "  text is an opaque id -> transfer is IMPOSSIBLE"}[regime])
        print("=" * 78)

        for name in SPLITS:
            split = S.ALL_SPLITS[name](examples, seed=3)
            if not split.test or not split.train:
                print(f"\n{split.name}: empty, skipped")
                continue
            print(f"\n{split.summary()}\n  ({split.note})")

            results = run_grid(split, seed=1)
            for r in results:
                print(fmt_row(r))

            print()
            for line in verdict(regime, name, results):
                print("  " + line)
                passed += line.startswith("PASS")
                failed += line.startswith("FAIL")
                all_checks.append(f"[{regime}/{name}] {line}")

            all_results[f"{regime}/{name}"] = [
                {"mode": r.mode, "temperature": r.temperature,
                 "raw": r.test.as_dict(), "calibrated": r.test_calibrated.as_dict()}
                for r in results
            ]

    print(f"\n{'=' * 78}")
    print(f"HARNESS SELF-CHECK: {passed} passed, {failed} failed")
    print("=" * 78)
    if failed:
        print("\nThe harness cannot be trusted on real data until these pass:")
        for c in all_checks:
            if "FAIL" in c:
                print("  " + c)

    out = Path(__file__).resolve().parents[1] / "reports"
    out.mkdir(exist_ok=True)
    (out / "control_results.json").write_text(
        json.dumps({"results": all_results, "checks": all_checks,
                    "passed": passed, "failed": failed}, indent=2))
    print(f"\nwrote {out / 'control_results.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
