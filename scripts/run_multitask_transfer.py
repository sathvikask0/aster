"""Build and run held-out-assay transfer on Tox21 or ToxCast.

This tests the question axis with hundreds of questions, which the peptide
benchmark cannot do. Small molecules, not peptides.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aster.real.multitask import MODES, build_multitask, train_multitask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "train"])
    p.add_argument("--source", choices=["tox21", "toxcast"], default="toxcast")
    p.add_argument("--raw-dir", type=Path, default=Path("data/multitask_raw"))
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("~/.cache/aster"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--molecule-disjoint", action="store_true",
                   help="Also separate molecules across splits, as the peptide protocol does")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--assays-per-step", type=int, default=64)
    p.add_argument("--rows-per-assay", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--models", nargs="+", choices=MODES, default=list(MODES))
    args = p.parse_args()
    data = args.data or Path(f"data/{args.source}_transfer")
    out = args.out or Path(f"checkpoints/{args.source}_transfer")
    if args.mode == "prepare":
        manifest = build_multitask(args.source, args.raw_dir, data, args.seed,
                                   molecule_disjoint=args.molecule_disjoint)
        print(f"{args.source}: {manifest['audit']['assays']} assays, "
              f"{manifest['audit']['labels']:,} labels, "
              f"{manifest['audit']['molecules']:,} molecules")
        for split, check in manifest["checks"].items():
            if isinstance(check, dict):
                print(f"  {split:11s} assays={check['assays']:4d} "
                      f"labels={check['labels']:>9,} positive_rate={check['positive_rate']:.3f}")
        print("molecules in more than one split:",
              manifest["checks"]["molecules_in_more_than_one_split"])
    else:
        train_multitask(data, out, args.cache_dir, tuple(args.seeds), args.epochs,
                        args.assays_per_step, args.rows_per_assay, args.steps_per_epoch,
                        args.hidden, tuple(args.models))


if __name__ == "__main__":
    main()
