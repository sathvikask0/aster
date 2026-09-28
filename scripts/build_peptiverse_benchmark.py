"""Download and build the reproducible five-question PeptiVerse benchmark."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aster.real.peptiverse import REVISION, build_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/peptiverse_raw") / REVISION)
    parser.add_argument("--out", type=Path, default=Path("data/peptiverse_benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-clusters", action="store_true",
                        help="Also keep each available upstream cluster together; ratios can be uneven")
    args = parser.parse_args()
    manifest = build_benchmark(args.raw_dir, args.out, args.seed, args.source_clusters)
    print(f"Built {manifest['checks']['examples']:,} examples in {args.out}")
    for task, splits in manifest["checks"]["tasks"].items():
        print(task, {name: values["n"] for name, values in splits.items()})


if __name__ == "__main__":
    main()
