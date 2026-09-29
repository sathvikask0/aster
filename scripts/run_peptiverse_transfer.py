"""Build and run question-disjoint PeptiVerse transfer experiments."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aster.real.peptiverse_transfer import (DEFAULT_PLAN, MODES, build_transfer,
                                          evaluate_transfer, train_transfer)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "train", "evaluate"])
    p.add_argument("--source", type=Path, default=Path("data/peptiverse_benchmark"))
    p.add_argument("--data", type=Path, default=Path("data/peptiverse_transfer"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/peptiverse_transfer"))
    p.add_argument("--cache-dir", type=Path, default=Path("~/.cache/aster"))
    for split in ("train", "validation", "test"):
        p.add_argument(f"--{split}-tasks", nargs="+", default=DEFAULT_PLAN[split])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--models", nargs="+", choices=MODES, default=list(MODES))
    args = p.parse_args()
    if args.mode == "prepare":
        plan = {s: getattr(args, f"{s}_tasks") for s in ("train", "validation", "test")}
        manifest = build_transfer(args.source, args.data, plan)
        print("Questions:", manifest["question_plan"])
        print("Excluded overlapping labels:", manifest["transfer_audit"]["excluded_shared_identity_rows"])
        for task, split in manifest["task_splits"].items():
            print(f"{split:10s} {task:18s} {manifest['checks']['tasks'][task][split]['n']:6d}")
    elif args.mode == "train":
        train_transfer(args.data, args.out, args.cache_dir, tuple(args.seeds), args.epochs,
                       args.batch_size, args.hidden, tuple(args.models))
    else:
        evaluate_transfer(args.data, args.out, args.cache_dir)


if __name__ == "__main__":
    main()
