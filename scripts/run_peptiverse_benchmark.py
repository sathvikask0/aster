"""Train reference models or evaluate frozen models on the PeptiVerse benchmark."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aster.real.peptiverse import load_benchmark, write_json
from aster.real.peptiverse_evaluation import evaluate_frozen, score_predictions, train_baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "score"])
    parser.add_argument("--data", type=Path, default=Path("data/peptiverse_benchmark"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/peptiverse_baseline"))
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/aster"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--predictions", type=Path, help="For score mode: JSONL with id and prediction")
    args = parser.parse_args()
    if args.mode == "train":
        if args.split != "validation":
            parser.error("Training selects on validation only; use evaluate --split test afterward")
        train_baselines(args.data, args.out, args.cache_dir, args.seed)
    elif args.mode == "evaluate":
        evaluate_frozen(args.data, args.out, args.cache_dir, args.split)
    else:
        if args.predictions is None:
            parser.error("score requires --predictions")
        rows, manifest = load_benchmark(args.data)
        predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
        report = score_predictions(rows, predictions, args.split)
        report["benchmark_files"] = manifest["files"]
        write_json(args.out / f"{args.split}_score.json", report)
    print(f"Results saved in {args.out}")


if __name__ == "__main__":
    main()
