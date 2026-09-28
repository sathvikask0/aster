"""Strict prediction scoring and inexpensive train/validation reference models.

These models deliberately do not read the English question: one head per assay
is the supervised baseline that a future question-conditioned Aster must beat.
No source labels, cluster IDs, PDB IDs, confidence scores or split fields are
allowed into the features.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

from aster.real.peptiverse import (AA, TASKS, VERSION, digest, file_digest,
                                  load_benchmark, write_json)


def metrics(task: str, labels, predictions) -> dict:
    y, p = np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Need a finite prediction for every example")
    result = {"n": len(y)}
    if TASKS[task]["kind"] == "classification":
        if ((p < 0) | (p > 1)).any():
            raise ValueError("Classification prediction must be P(label=1) in [0, 1]")
        if set(y) != {0, 1}:
            raise ValueError("Classification scoring requires both classes")
        hard = p >= 0.5  # fixed before evaluation, not optimized on test
        result.update(auroc=float(roc_auc_score(y, p)),
                      auprc=float(average_precision_score(y, p)),
                      accuracy=float(accuracy_score(y, hard)),
                      balanced_accuracy=float(balanced_accuracy_score(y, hard)),
                      mcc=float(matthews_corrcoef(y, hard)),
                      brier=float(brier_score_loss(y, p)),
                      log_loss=float(log_loss(y, p, labels=[0, 1])))
    else:
        result.update(mae=float(mean_absolute_error(y, p)),
                      rmse=float(np.sqrt(mean_squared_error(y, p))),
                      r2=float(r2_score(y, p)) if len(y) > 1 and np.ptp(y) > 0 else None,
                      spearman=float(spearmanr(y, p).statistic)
                      if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0 else None)
        if task == "half_life":
            if (p <= 0).any():
                raise ValueError("Half-life predictions must be positive hours")
            result["mae_log10_hours"] = float(mean_absolute_error(np.log10(y), np.log10(p)))
    return result


def score_predictions(rows: list[dict], predictions: list[dict], split="validation") -> dict:
    """Predictions contain exactly {id, prediction}; unknown or missing IDs fail."""
    if split not in ("validation", "test"):
        raise ValueError("Score validation or test only")
    expected = {r["id"]: r for r in rows if r["split"] == split}
    if not expected:
        raise ValueError(f"Empty {split} split")
    predicted = {}
    for item in predictions:
        if set(item) != {"id", "prediction"}:
            raise ValueError("Each prediction must contain exactly id and prediction")
        if item["id"] in predicted:
            raise ValueError("Duplicate prediction id")
        predicted[item["id"]] = float(item["prediction"])
    if set(expected) != set(predicted):
        raise ValueError(f"Prediction coverage mismatch: missing {len(set(expected) - set(predicted))}, "
                         f"unexpected {len(set(predicted) - set(expected))}")
    grouped = defaultdict(list)
    for identity, row in expected.items():
        grouped[row["task"]].append((row["label"], predicted[identity]))
    scored = {task: metrics(task, *zip(*values)) for task, values in sorted(grouped.items())}
    classification = [v["balanced_accuracy"] for k, v in scored.items()
                      if TASKS[k]["kind"] == "classification"]
    return {"split": split, "tasks": scored,
            "classification_macro_balanced_accuracy": float(np.mean(classification)) if classification else None,
            "regression_aggregation": "Per-task native units; no average across incompatible units."}


def feature_matrix(rows: list[dict], cache_dir: Path) -> sparse.csr_matrix:
    """Chiral Morgan bits + fixed-scale descriptors + target composition.

    Affinity additionally gets descriptor × target-composition interactions.
    Fixed scaling does not learn any statistics from validation/test data.
    Whole structures and target sequences are used without truncation.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    signatures = [r["smiles"] + "\0" + (r["target_sequence"] or "") for r in rows]
    key = digest(json.dumps(["morgan-v1", rdBase.rdkitVersion, signatures]))
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"peptiverse_{key}.npz"
    if path.exists():
        return sparse.load_npz(path)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    data, indices, indptr = [], [], [0]
    aa = sorted(AA)
    chemistry_cache = {}
    for index, row in enumerate(rows):
        smiles = row["smiles"]
        if smiles not in chemistry_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES in benchmark feature input")
            bits = list(generator.GetFingerprint(mol).GetOnBits())
            descriptors = np.array([
                np.log1p(Descriptors.MolWt(mol)) / 10,
                np.log1p(mol.GetNumAtoms()) / 10,
                np.log1p(Descriptors.NumHDonors(mol)) / 5,
                np.log1p(Descriptors.NumHAcceptors(mol)) / 5,
                np.log1p(Descriptors.NumRotatableBonds(mol)) / 5,
                np.log1p(Descriptors.TPSA(mol)) / 10,
                Descriptors.MolLogP(mol) / 20,
                Chem.GetFormalCharge(mol) / 10,
            ])
            chemistry_cache[smiles] = bits, descriptors
        bits, descriptors = chemistry_cache[smiles]
        target = row["target_sequence"] or ""
        target_features = np.array([target.count(a) / max(1, len(target)) for a in aa]
                                   + [np.log1p(len(target)) / 10])
        dense = np.concatenate([descriptors, target_features,
                                np.outer(descriptors, target_features).ravel()])
        nonzero = np.flatnonzero(dense)
        indices.extend(bits)
        data.extend([1.0] * len(bits))
        indices.extend((2048 + nonzero).tolist())
        data.extend(dense[nonzero].tolist())
        indptr.append(len(data))
        if index % 5000 == 0:
            print(f"Featurizing {index + 1}/{len(rows)}", flush=True)
    features = sparse.csr_matrix((np.asarray(data, dtype=np.float32), indices, indptr),
                                 shape=(len(rows), 2048 + 8 + 21 + 8 * 21))
    if not np.isfinite(features.data).all():
        raise ValueError("Nonfinite molecular features")
    sparse.save_npz(path, features)
    return features


def _predict(model, task, features):
    if TASKS[task]["kind"] == "classification":
        return model.predict_proba(features)[:, list(model.classes_).index(1)]
    result = model.predict(features)
    return np.power(10., result) if task == "half_life" else result


def train_baselines(directory: Path, out: Path, cache_dir: Path, seed=42) -> dict:
    """Fit on train, select regularization on validation, never score test."""
    rows, manifest = load_benchmark(directory)
    out = Path(out)
    if (out / "selection.json").exists():
        raise FileExistsError(f"{out} already contains a fitted run; choose a new output directory")
    # Do not even featurize test rows during model selection.
    rows = [r for r in rows if r["split"] != "test"]
    features = feature_matrix(rows, cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions, constants, selections, model_hashes = [], [], {}, {}
    for task, spec in TASKS.items():
        train = [i for i, row in enumerate(rows) if row["task"] == task and row["split"] == "train"]
        val = [i for i, row in enumerate(rows) if row["task"] == task and row["split"] == "validation"]
        x, xv = features[train], features[val]
        y = np.array([rows[i]["label"] for i in train])
        yv = np.array([rows[i]["label"] for i in val])
        fit_y = np.log10(y) if task == "half_life" else y
        classification = spec["kind"] == "classification"
        dummy = DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="median")
        dummy.fit(x, fit_y)
        candidates, best = [], None
        for regularization in (0.1, 1., 10.):
            model = (LogisticRegression(C=regularization, max_iter=2000, random_state=seed)
                     if classification else Ridge(alpha=regularization, solver="lsqr"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, fit_y)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f"{task}: logistic baseline did not converge")
            prediction = _predict(model, task, xv)
            score = metrics(task, yv, prediction)
            # Use proper probability loss for classification and preserve the
            # half-life log transform in validation selection.
            loss = score["log_loss"] if classification else score.get("mae_log10_hours", score["mae"])
            candidates.append({"C" if classification else "alpha": regularization,
                               "selection_loss": loss, "validation": score})
            if best is None or loss < best[0]:
                best = loss, model, prediction, regularization
        _, model, predicted, selected = best
        selections[task] = {"model": type(model).__name__, "regularization": selected,
                            "selection_metric": "log_loss" if classification else
                            "mae_log10_hours" if task == "half_life" else "mae",
                            "candidates": candidates}
        for i, prediction, constant in zip(val, predicted, _predict(dummy, task, xv)):
            predictions.append({"id": rows[i]["id"], "prediction": float(prediction)})
            constants.append({"id": rows[i]["id"], "prediction": float(constant)})
        path = out / f"{task}.joblib"
        joblib.dump({"model": model, "dummy": dummy, "task": task, "feature_version": "morgan-v1"}, path)
        model_hashes[path.name] = file_digest(path)
        print(f"{task}: selected {selected:g}, validation loss {best[0]:.4f}", flush=True)
    report = {"benchmark_version": VERSION, "benchmark_files": manifest["files"],
              "seed": seed, "training_rows_only": True, "test_evaluated": False,
              "baseline": "Per-task chiral Morgan/descriptor models; target composition and interactions for binding.",
              "reads_question_text": False, "models": model_hashes, "selection": selections,
              "validation": score_predictions(rows, predictions),
              "constant_baseline": score_predictions(rows, constants)}
    for name, values in (("validation_predictions", predictions), ("validation_constant_predictions", constants)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(p, allow_nan=False) + "\n" for p in values))
    write_json(out / "selection.json", report)
    return report


def evaluate_frozen(directory: Path, model_dir: Path, cache_dir: Path, split="test") -> dict:
    """Explicit separate evaluation of already selected, local trusted models."""
    if split not in ("validation", "test"):
        raise ValueError("Evaluate validation or test")
    rows, manifest = load_benchmark(directory)
    model_dir = Path(model_dir)
    selection = json.loads((model_dir / "selection.json").read_text())
    if selection["benchmark_files"] != manifest["files"]:
        raise ValueError("Models were selected on a different benchmark")
    selected = [r for r in rows if r["split"] == split]
    x = feature_matrix(selected, cache_dir)
    predictions = []
    for task in TASKS:
        path = model_dir / f"{task}.joblib"
        if file_digest(path) != selection["models"][path.name]:
            raise ValueError("Model changed since validation selection")
        bundle = joblib.load(path)  # Only load your own trusted local artifacts.
        idx = [i for i, row in enumerate(selected) if row["task"] == task]
        for i, value in zip(idx, _predict(bundle["model"], task, x[idx])):
            predictions.append({"id": selected[i]["id"], "prediction": float(value)})
    result = score_predictions(rows, predictions, split)
    result["benchmark_files"] = manifest["files"]
    result["model_files"] = selection["models"]
    (model_dir / f"{split}_predictions.jsonl").write_text("".join(json.dumps(p, allow_nan=False) + "\n" for p in predictions))
    write_json(model_dir / f"{split}_evaluation.json", result)
    return result
