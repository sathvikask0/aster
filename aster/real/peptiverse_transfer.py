"""Question-disjoint, peptide-disjoint transfer protocol and shared pilot model.

The chemical encoder here is a fixed Morgan/descriptor representation, not ESM
or PeptideCLM. Frozen MiniLM supplies task semantics. This inexpensive pilot
tests the transfer setup before committing to foundation-model fine-tuning.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from aster.real.peptiverse import (SPLITS, TASKS, file_digest, load_benchmark,
                                  validate_benchmark, write_json)
from aster.real.peptiverse_evaluation import feature_matrix, score_predictions

DEFAULT_PLAN = {
    "train": ["solubility", "penetrance", "pampa", "binding_affinity"],
    "validation": ["hemolysis", "caco2"],
    "test": ["toxicity", "half_life"],
}
MODES = ("question", "entity_only", "task_id", "question_only")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def task_assignment(plan: dict) -> dict:
    if set(plan) != set(SPLITS) or any(not plan[s] for s in SPLITS):
        raise ValueError("Plan needs nonempty train, validation and test question lists")
    flat = [task for split in SPLITS for task in plan[split]]
    if len(flat) != len(set(flat)) or set(flat) != set(TASKS):
        raise ValueError("Assign each of the eight questions to exactly one split")
    return {task: split for split in SPLITS for task in plan[split]}


def hold_out_questions(rows: list[dict], plan: dict) -> tuple[list[dict], dict]:
    """Test identities take priority, then validation, then training.

    Drop lower-priority labels for a shared identity rather than moving them
    into the held-out split (which would leak the training question there).
    Decisions use identities and question assignments, never label values.
    """
    assignment = task_assignment(plan)
    priority = {"train": 0, "validation": 1, "test": 2}
    owner = {}
    for row in rows:
        group, split = row["group_id"], assignment[row["task"]]
        if group not in owner or priority[split] > priority[owner[group]]:
            owner[group] = split
    kept, dropped = [], Counter()
    for row in rows:
        split = assignment[row["task"]]
        if owner[row["group_id"]] != split:
            dropped[row["task"]] += 1
        else:
            kept.append({**row, "split": split})
    kept.sort(key=lambda r: (r["task"], r["id"]))
    checks = validate_benchmark(kept, task_splits=assignment)
    return kept, {"excluded_shared_identity_rows": dict(sorted(dropped.items())),
                  "excluded_rows": sum(dropped.values()), "checks": checks}


def build_transfer(source: Path, out: Path, plan: dict | None = None) -> dict:
    plan = copy.deepcopy(DEFAULT_PLAN if plan is None else plan)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}; choose a fresh directory")
    rows, original = load_benchmark(source)
    if original.get("task_splits"):
        raise ValueError("Build from the complete supervised benchmark, not an already filtered holdout")
    rows, audit = hold_out_questions(rows, plan)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        path = out / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        files[path.name] = file_digest(path)
    manifest = {**original, "benchmark": "peptiverse_question_transfer",
                "parent_files": original["files"], "files": files,
                "parent_split_policy": original["split_policy"],
                "split_policy": "held_out_questions_and_global_identities",
                "fractions": None, "task_splits": task_assignment(plan),
                "question_plan": plan, "transfer_audit": audit,
                "checks": audit["checks"],
                "limitations": [
                    "Exploratory new-task and new-peptide transfer; no claim of unseen homology families.",
                    "Validation questions influence model selection but never gradient updates.",
                    "Native regression scales differ; new-task regression requires extrapolating the output scale.",
                    "Question holdouts are assay endpoints, not necessarily disjoint biological categories.",
                    "Earlier supervised experiments used these task labels; transfer models must start fresh.",
                    *[s for s in original["limitations"] if not s.startswith("Supervised task coverage")],
                ]}
    write_json(out / "manifest.json", manifest)
    return manifest


class TransferModel(nn.Module):
    """One shared classification head and one shared regression head.

    No parameters are allocated for held-out tasks. The task-ID control uses a
    fixed zero vector for every unseen question. Question-only has no chemical
    or target information; entity-only has no question information.
    """
    def __init__(self, d_entity, d_text, n_train_tasks, mode="question", hidden=64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.entity = nn.Sequential(nn.Linear(d_entity, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.question = nn.Sequential(nn.Linear(d_text, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.task = nn.Embedding(n_train_tasks + 1, hidden, padding_idx=0)
        self.shared = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.1))
        self.output = nn.Linear(hidden, 2)

    def forward(self, x, q, task_ids):
        e = self.entity(x)
        if self.mode == "question_only":
            e = torch.zeros_like(e)
        if self.mode == "task_id":
            t = self.task(task_ids)
        elif self.mode == "entity_only":
            t = torch.zeros_like(e)
        else:
            t = self.question(q)
        return self.output(self.shared(torch.cat((e, t, e * t), dim=-1)))


def task_text(task):
    spec = TASKS[task]
    if spec["kind"] == "classification":
        return spec["question"] + " Negative: " + spec["options"][0] + " Positive: " + spec["options"][1]
    scale = "log10 hours" if task == "half_life" else spec["units"]
    return spec["question"] + " Predict a number on the " + scale + " scale."


def encode_questions(task_names, revision=None):
    """Frozen, revision-pinned MiniLM with attention-mask mean pooling."""
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer
    revision = revision or model_info(TEXT_MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=revision)
    encoder = AutoModel.from_pretrained(TEXT_MODEL, revision=revision).eval()
    encoded = tokenizer([task_text(t) for t in task_names], padding=True, return_tensors="pt")
    if encoded["input_ids"].shape[1] > encoder.config.max_position_embeddings:
        raise ValueError("Question exceeds text encoder length; no silent truncation")
    with torch.no_grad():
        states = encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        vectors = (states * mask).sum(1) / mask.sum(1)
        vectors = nn.functional.normalize(vectors, dim=-1)
    return vectors.numpy(), revision


def fit_regression_scale(rows):
    # One global training-only scale. Never fit a scaler to held-out task labels.
    values = [np.log10(r["label"]) if r["task"] == "half_life" else r["label"]
              for r in rows if r["split"] == "train" and r["kind"] == "regression"]
    if not values:
        raise ValueError("At least one regression question must be in training")
    return {"mean": float(np.mean(values)), "std": max(float(np.std(values)), 1e-6)}


def regression_value(row, scale):
    value = np.log10(row["label"]) if row["task"] == "half_life" else row["label"]
    return (value - scale["mean"]) / scale["std"]


def _predictions(model, rows, x, q, ids, scale, batch_size=512):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            sl = slice(start, start + batch_size)
            result = model(x[sl], q[sl], ids[sl]).cpu().numpy()
            for row, (logit, reg) in zip(rows[sl], result):
                if row["kind"] == "classification":
                    value = float(1 / (1 + np.exp(-np.clip(logit, -50, 50))))
                else:
                    value = float(reg * scale["std"] + scale["mean"])
                    if row["task"] == "half_life":
                        value = float(10. ** value)
                values.append({"id": row["id"], "prediction": value})
    return values


def _validation_loss(report):
    # Macro average two separate loss families; no test-derived normalizer.
    losses = []
    for task, score in report["tasks"].items():
        if TASKS[task]["kind"] == "classification":
            losses.append(score["log_loss"])
        else:
            losses.append(score.get("mae_log10_hours", score["mae"]))
    return float(np.mean(losses))


def _inputs(rows, features, texts, names, train_names):
    pos = {t: i for i, t in enumerate(names)}
    known = {t: i + 1 for i, t in enumerate(train_names)}
    return (torch.tensor(features.toarray(), dtype=torch.float32),
            torch.tensor(texts[[pos[r["task"]] for r in rows]], dtype=torch.float32),
            torch.tensor([known.get(r["task"], 0) for r in rows], dtype=torch.long))


def train_transfer(directory: Path, out: Path, cache_dir: Path, seeds=(42, 43, 44),
                   epochs=12, batch_size=256, hidden=64,
                   modes=("question", "entity_only", "task_id", "question_only")):
    if epochs < 1 or batch_size < 1 or hidden < 1 or not seeds or not modes or set(modes) - set(MODES):
        raise ValueError("Invalid training configuration")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}; choose a fresh run directory")
    rows, manifest = load_benchmark(directory)
    if not manifest.get("task_splits"):
        raise ValueError("Prepare a held-out-question benchmark first")
    rows = [r for r in rows if r["split"] != "test"]
    train_names = manifest["question_plan"]["train"]
    if not any(TASKS[t]["kind"] == "classification" for t in train_names):
        raise ValueError("At least one classification question must be in training")
    # Test questions and features are not encoded until explicit evaluation.
    names = sorted({r["task"] for r in rows})
    features = feature_matrix(rows, cache_dir)
    texts, revision = encode_questions(names)
    scale = fit_regression_scale(rows)
    x, q, ids = _inputs(rows, features, texts, names, train_names)
    by_task = {t: np.array([i for i, r in enumerate(rows) if r["task"] == t and r["split"] == "train"])
               for t in train_names}
    validation = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    val_rows = [rows[i] for i in validation]
    targets = torch.tensor([float(r["label"]) if r["kind"] == "classification" else regression_value(r, scale)
                            for r in rows], dtype=torch.float32)
    out.mkdir(parents=True, exist_ok=True)
    config = {"benchmark_files": manifest["files"], "question_plan": manifest["question_plan"],
              "text_model": TEXT_MODEL, "text_revision": revision,
              "feature_version": "morgan-v1", "regression_scale": scale,
              "seeds": list(seeds), "epochs": epochs, "batch_size": batch_size,
              "hidden": hidden, "modes": list(modes), "learning_rate": 0.001,
              "selection": "macro validation classification log-loss and native regression MAE (log10 for half-life)",
              "test_evaluated": False, "encoder_training": "Fixed chemistry; frozen MiniLM; shared heads trained from scratch",
              "train_task_balancing": "Round-robin tasks; reshuffled independent batches; each task contributes equally per step",
              "software": {"torch": torch.__version__, "numpy": np.__version__}}
    # Save the protocol before any optimization or score is available.
    write_json(out / "protocol.json", config)
    reports = {}
    torch.set_num_threads(4)
    steps = max(1, int(np.ceil(sum(len(v) for v in by_task.values()) / (batch_size * len(train_names)))))
    for seed in seeds:
        for mode in modes:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = TransferModel(x.shape[1], q.shape[1], len(train_names), mode, hidden)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            best, stale, history = None, 0, []
            for epoch in range(epochs):
                model.train()
                losses = []
                for _ in range(steps):
                    optimizer.zero_grad()
                    total = 0
                    for task in train_names:
                        pool = by_task[task]
                        idx = rng.choice(pool, size=min(batch_size, len(pool)), replace=False)
                        output = model(x[idx], q[idx], ids[idx])
                        if TASKS[task]["kind"] == "classification":
                            loss = nn.functional.binary_cross_entropy_with_logits(output[:, 0], targets[idx])
                        else:
                            loss = nn.functional.smooth_l1_loss(output[:, 1], targets[idx])
                        total = total + loss / len(train_names)
                    total.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    losses.append(float(total.detach()))
                prediction = _predictions(model, val_rows, x[validation], q[validation], ids[validation], scale)
                score = score_predictions(rows, prediction)
                loss = _validation_loss(score)
                history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)), "validation_selection_loss": loss})
                print(f"{mode} seed={seed} epoch={epoch+1}: validation={loss:.4f}", flush=True)
                if best is None or loss < best[0]:
                    best = loss, copy.deepcopy(model.state_dict()), prediction, score, epoch + 1
                    stale = 0
                else:
                    stale += 1
                if stale >= 3:
                    break
            name = f"{mode}_s{seed}"
            checkpoint = out / f"{name}.pt"
            torch.save({"state_dict": best[1], "mode": mode, "d_entity": x.shape[1],
                        "d_text": q.shape[1], "hidden": hidden, "n_train_tasks": len(train_names)}, checkpoint)
            reports[name] = {"mode": mode, "seed": seed, "best_epoch": best[4], "history": history,
                             "validation": best[3], "checkpoint_sha256": file_digest(checkpoint)}
            (out / f"{name}_validation_predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in best[2]))
            write_json(out / "selection.json", {"protocol": config, "runs": reports})
    return {"protocol": config, "runs": reports}


def evaluate_transfer(directory: Path, out: Path, cache_dir: Path):
    """Evaluate all predeclared models once; never select a winner using test."""
    out = Path(out)
    if (out / "test_evaluation.json").exists():
        raise FileExistsError("Test already evaluated for this run; read the saved report")
    selected = json.loads((out / "selection.json").read_text())
    config = selected["protocol"]
    expected = {f"{m}_s{s}" for m in config["modes"] for s in config["seeds"]}
    if set(selected["runs"]) != expected:
        raise ValueError("Finish every predeclared training run before opening test")
    rows, manifest = load_benchmark(directory)
    if config["benchmark_files"] != manifest["files"]:
        raise ValueError("Benchmark changed since training")
    test = [r for r in rows if r["split"] == "test"]
    # A wrong training-task prompt of the same output type is an ablation;
    # deterministic task names choose donors, never test labels or performance.
    test_names = sorted({r["task"] for r in test})
    train_names = manifest["question_plan"]["train"]
    donors = {t: sorted(n for n in train_names if TASKS[n]["kind"] == TASKS[t]["kind"])[0] for t in test_names}
    names = sorted(set(test_names) | set(donors.values()))
    texts, _ = encode_questions(names, config["text_revision"])
    x, q, ids = _inputs(test, feature_matrix(test, cache_dir), texts, names, train_names)
    pos = {t: i for i, t in enumerate(names)}
    wrong_q = torch.tensor(texts[[pos[donors[r["task"]]] for r in test]], dtype=torch.float32)
    reports = {}
    torch.set_num_threads(4)
    for name, run in selected["runs"].items():
        path = out / f"{name}.pt"
        if file_digest(path) != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after selection")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        model = TransferModel(saved["d_entity"], saved["d_text"], saved["n_train_tasks"], saved["mode"], saved["hidden"])
        model.load_state_dict(saved["state_dict"])
        predicted = _predictions(model, test, x, q, ids, config["regression_scale"])
        report = score_predictions(rows, predicted, "test")
        if run["mode"] in ("question", "question_only"):
            swapped = _predictions(model, test, x, wrong_q, ids, config["regression_scale"])
            report["wrong_question"] = score_predictions(rows, swapped, "test")
            report["wrong_question_donors"] = donors
        reports[name] = report
        (out / f"{name}_test_predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predicted))
    result = {"protocol": config, "test_evaluated": True, "runs": reports,
              "interpretation": "Exploratory transfer pilot. Compare all seeds and controls; no model selected on test. "
              "Prompt sensitivity alone does not establish correct biological reasoning."}
    write_json(out / "test_evaluation.json", result)
    return result
