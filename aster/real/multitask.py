"""Question-transfer at scale: hold out whole assays from Tox21 or ToxCast.

The peptide benchmark gives four or five training questions, which is too few to
learn how a question's wording should reshape a prediction. This module supplies
the same protocol with up to 617 questions, so the question axis can be tested
without the peptide data being the limiting factor.

These are small molecules, not peptides. That is deliberate: the point is to find
out whether held-out-question transfer works *at all* when questions are
plentiful. A protein encoder does not apply here, and nothing in this module is
evidence about peptide biology.

Default splits hold out assays but let a molecule appear in several splits under
different assays, which isolates the question axis and is the easier of the two
settings. `molecule_disjoint=True` additionally separates molecules, matching the
stricter peptide protocol. Numbers from the two settings are not comparable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd

from aster.real.skill import classification_skill, macro_skill

SPLITS = ("train", "validation", "test")
VERSION = 1
SOURCES = {
    "tox21": dict(
        file="tox21.csv.gz",
        sha256="45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360",
        smiles="smiles"),
    "toxcast": dict(
        file="toxcast_data.csv.gz",
        sha256="02c8b96da666884a13746e4891fc10b4f81f51fa14e0060df1db7da0223b46c6",
        smiles="smiles"),
}
BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
# Tox21's twelve codes are opaque to a text encoder as written, so they get real
# questions. ToxCast's 617 names are turned into text mechanically below.
TOX21_QUESTIONS = {
    "NR-AR": "Does this compound activate the androgen receptor?",
    "NR-AR-LBD": "Does this compound activate the androgen receptor ligand binding domain?",
    "NR-AhR": "Does this compound activate the aryl hydrocarbon receptor?",
    "NR-Aromatase": "Does this compound disrupt the aromatase enzyme?",
    "NR-ER": "Does this compound activate the estrogen receptor?",
    "NR-ER-LBD": "Does this compound activate the estrogen receptor ligand binding domain?",
    "NR-PPAR-gamma": "Does this compound activate the PPAR gamma receptor?",
    "SR-ARE": "Does this compound trigger the antioxidant response element stress pathway?",
    "SR-ATAD5": "Does this compound cause genotoxic stress reported by ATAD5?",
    "SR-HSE": "Does this compound trigger the heat shock response element pathway?",
    "SR-MMP": "Does this compound disrupt the mitochondrial membrane potential?",
    "SR-p53": "Does this compound activate the p53 stress response pathway?",
}
# Expansions for the tokens that recur across ToxCast assay names. Anything not
# listed is passed through as its own word, so an unknown code still contributes.
TOKENS = {
    "dn": "decreased", "up": "increased", "ch": "change", "hr": "hours",
    "h": "hours", "CIS": "cis-regulatory element", "TRANS": "trans-activation",
    "BLA": "beta-lactamase reporter", "LUC": "luciferase reporter",
    "Agonist": "agonist activity", "Antagonist": "antagonist activity",
    "Positive": "positive direction", "Negative": "negative direction",
    "ratio": "ratio", "viability": "cell viability", "Cytotoxicity": "cytotoxicity",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_source(name: str, root: Path) -> Path:
    """Download the pinned MoleculeNet CSV, refusing any modified copy."""
    if name not in SOURCES:
        raise ValueError(f"Unknown source: {name}")
    spec = SOURCES[name]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec["file"]
    if not path.exists():
        temporary = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{spec['file']}", timeout=120) as response, \
                    temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if file_digest(temporary) != spec["sha256"]:
                raise ValueError(f"Checksum mismatch downloading {spec['file']}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if file_digest(path) != spec["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def assay_question(name: str) -> str:
    """Readable question text for an assay code.

    ToxCast names carry the vendor, cell line, endpoint, timepoint and direction,
    which is real signal for a text encoder even though the codes are terse.
    """
    if name in TOX21_QUESTIONS:
        return TOX21_QUESTIONS[name]
    parts = [p for p in re.split(r"[_\-]", name) if p]
    words = []
    for part in parts:
        if part in TOKENS:
            words.append(TOKENS[part])
        elif re.fullmatch(r"\d+h(r|rs)?", part, re.I):
            words.append(re.sub(r"h(r|rs)?$", " hours", part, flags=re.I))
        else:
            # Split camel case so "CellCycleArrest" becomes three words.
            words.append(" ".join(re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|\d+|[a-z]+", part)) or part)
    return f"In the {name} assay, is this compound active? Assay details: {' '.join(words)}."


def load_frame(path: Path, smiles_column: str) -> tuple[pd.DataFrame, list[str]]:
    with gzip.open(path, "rt") as stream:
        frame = pd.read_csv(stream)
    assays = [c for c in frame.columns
              if c != smiles_column and frame[c].dropna().isin([0, 1, 0.0, 1.0]).all()
              and frame[c].notna().any()]
    if not assays:
        raise ValueError("No binary assay columns found")
    return frame, assays


def build_rows(name: str, root: Path) -> tuple[list[dict], dict]:
    """One row per measured (molecule, assay) pair. Unmeasured pairs are absent."""
    from rdkit import Chem, rdBase

    spec = SOURCES[name]
    frame, assays = load_frame(fetch_source(name, root), spec["smiles"])
    molecules, dropped = {}, Counter()
    for raw in frame[spec["smiles"]]:
        if not isinstance(raw, str) or raw in molecules:
            continue
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(raw)
        if mol is None or mol.GetNumAtoms() == 0:
            dropped["invalid_smiles"] += 1
            molecules[raw] = None
            continue
        molecules[raw] = Chem.MolToSmiles(mol)
    rows = []
    for position, record in enumerate(frame.to_dict("records")):
        raw = record[spec["smiles"]]
        canonical = molecules.get(raw) if isinstance(raw, str) else None
        if canonical is None:
            continue
        molecule_id = digest("smiles:" + canonical)
        for assay in assays:
            value = record[assay]
            if pd.isna(value):
                continue
            rows.append({
                "task": assay, "kind": "classification",
                "question": assay_question(assay),
                "options": ["It is inactive in this assay.", "It is active in this assay."],
                "label": int(value), "units": "binary", "smiles": canonical,
                "molecule_id": molecule_id, "source_row": position,
            })
    # One label per (molecule, assay); disagreeing replicates have no majority.
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["molecule_id"], row["task"])].append(row)
    result, conflicts, collapsed = [], 0, 0
    for key, group in sorted(buckets.items()):
        if len({r["label"] for r in group}) > 1:
            conflicts += len(group)
            continue
        row = dict(group[0])
        row["id"] = digest(json.dumps([VERSION, name, *key]))
        collapsed += len(group) - 1
        result.append(row)
    result.sort(key=lambda r: (r["task"], r["id"]))
    audit = {"source": name, "source_rows": len(frame), "assays": len(assays),
             "molecules": sum(1 for v in molecules.values() if v),
             "labels": len(result), "dropped": dict(dropped),
             "conflicting_pairs_removed": conflicts,
             "duplicate_pairs_collapsed": collapsed}
    return result, audit


def assay_plan(assays, seed: int = 42, fractions=(0.80, 0.10, 0.10)) -> dict:
    """Assign whole assays to splits by seeded hash, never by label statistics."""
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Need three positive fractions summing to one")
    ordered = sorted(set(assays))
    if len(ordered) < 3:
        raise ValueError("Need at least three assays to hold any out")
    plan = {}
    for assay in ordered:
        u = int(digest(f"{seed}:assay:{assay}")[:16], 16) / 2 ** 64
        plan[assay] = ("train" if u < fractions[0]
                       else "validation" if u < fractions[0] + fractions[1] else "test")
    # Every split must be populated; nudge deterministically if a tail is empty.
    for split in SPLITS:
        if split not in plan.values():
            for assay in ordered:
                if list(plan.values()).count(plan[assay]) > 1:
                    plan[assay] = split
                    break
    return plan


def split_rows(rows, plan: dict, molecule_disjoint: bool = False, seed: int = 42):
    """Attach splits from the assay plan, optionally separating molecules too."""
    missing = {r["task"] for r in rows} - set(plan)
    if missing:
        raise ValueError(f"Assay plan is missing {len(missing)} assays")
    out, dropped = [], Counter()
    if molecule_disjoint:
        # Give every molecule one split, then keep only rows whose assay agrees.
        molecule_split = {}
        for molecule in sorted({r["molecule_id"] for r in rows}):
            u = int(digest(f"{seed}:molecule:{molecule}")[:16], 16) / 2 ** 64
            molecule_split[molecule] = ("train" if u < 0.80
                                        else "validation" if u < 0.90 else "test")
        for row in rows:
            split = plan[row["task"]]
            if molecule_split[row["molecule_id"]] != split:
                dropped[split] += 1
                continue
            out.append({**row, "split": split, "group_id": row["molecule_id"]})
    else:
        for row in rows:
            out.append({**row, "split": plan[row["task"]], "group_id": row["molecule_id"]})
    out.sort(key=lambda r: (r["task"], r["id"]))
    return out, {"molecule_disjoint": molecule_disjoint,
                 "dropped_for_molecule_disjointness": dict(dropped)}


def validate(rows, plan: dict) -> dict:
    """Fail closed on empty splits, leaked assays, duplicate ids or bad labels."""
    if not rows:
        raise ValueError("No rows")
    seen = set()
    per_split = defaultdict(lambda: {"labels": 0, "assays": set(), "positives": 0})
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate example id")
        seen.add(row["id"])
        if row["label"] not in (0, 1):
            raise ValueError("Label must be 0 or 1")
        if row["split"] != plan[row["task"]]:
            raise ValueError("Held-out assay leaked into another split")
        bucket = per_split[row["split"]]
        bucket["labels"] += 1
        bucket["assays"].add(row["task"])
        bucket["positives"] += row["label"]
    if set(per_split) != set(SPLITS):
        raise ValueError(f"Every split must be populated; got {sorted(per_split)}")
    overlap = set.intersection(*(v["assays"] for v in per_split.values()))
    if overlap:
        raise ValueError(f"Assays appear in more than one split: {sorted(overlap)}")
    checks = {}
    for split, bucket in sorted(per_split.items()):
        single = [a for a in bucket["assays"]
                  if len({r["label"] for r in rows if r["task"] == a}) < 2]
        checks[split] = {"labels": bucket["labels"], "assays": len(bucket["assays"]),
                         "positive_rate": round(bucket["positives"] / bucket["labels"], 4),
                         "single_class_assays": len(single)}
    molecules = defaultdict(set)
    for row in rows:
        molecules[row["group_id"]].add(row["split"])
    checks["molecules_in_more_than_one_split"] = sum(1 for v in molecules.values() if len(v) > 1)
    return checks


def build_multitask(name: str, raw_dir: Path, out: Path, seed: int = 42,
                    fractions=(0.80, 0.10, 0.10), molecule_disjoint: bool = False) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}; choose a fresh directory")
    rows, audit = build_rows(name, raw_dir)
    plan = assay_plan({r["task"] for r in rows}, seed, fractions)
    rows, split_audit = split_rows(rows, plan, molecule_disjoint, seed)
    checks = validate(rows, plan)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        path = out / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        files[path.name] = file_digest(path)
    manifest = {"benchmark": f"{name}_question_transfer", "version": VERSION,
                "source": name, "source_url": f"{BASE_URL}/{SOURCES[name]['file']}",
                "source_sha256": SOURCES[name]["sha256"], "seed": seed,
                "fractions": list(fractions), "assay_plan": plan,
                "assay_splits": {s: sorted(a for a, v in plan.items() if v == s) for s in SPLITS},
                "files": files, "audit": audit, "split_audit": split_audit, "checks": checks,
                "limitations": [
                    "Small molecules, not peptides; says nothing about peptide biology.",
                    "Default splits share molecules across splits, isolating the question axis;"
                    " this is the easier setting and is not comparable to the peptide protocol.",
                    "Assay names are terse codes, so question text is weaker than plain prose.",
                    "Label sparsity differs per assay; per-assay n ranges widely.",
                ]}
    with (out / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
    return manifest


def load_multitask(directory: Path) -> tuple[list[dict], dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("Unsupported benchmark version")
    rows = []
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if file_digest(path) != manifest["files"][path.name]:
            raise ValueError(f"Checksum mismatch: {path}")
        subset = [json.loads(line) for line in path.read_text().splitlines()]
        if any(row["split"] != split for row in subset):
            raise ValueError("Row stored in the wrong split file")
        rows.extend(subset)
    validate(rows, manifest["assay_plan"])
    return rows, manifest


def molecule_features(rows, cache_dir: Path, bits: int = 2048) -> tuple[np.ndarray, dict]:
    """Chiral Morgan bits per unique molecule, cached by content.

    Featurizing molecules rather than rows matters here: ToxCast has 8,597
    molecules behind 1.5M labels.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    from scipy import sparse

    unique = sorted({r["smiles"] for r in rows})
    key = digest(json.dumps(["morgan", bits, rdBase.rdkitVersion, unique]))
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"multitask_{key}.npz"
    index = {s: i for i, s in enumerate(unique)}
    if path.exists():
        return sparse.load_npz(path).toarray().astype(np.float32), index
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits,
                                                          includeChirality=True)
    matrix = np.zeros((len(unique), bits), dtype=np.float32)
    for i, smiles in enumerate(unique):
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES reached featurization")
        for bit in generator.GetFingerprint(mol).GetOnBits():
            matrix[i, bit] = 1.
        if i % 2000 == 0:
            print(f"  featurizing {i}/{len(unique)}", flush=True)
    sparse.save_npz(path, sparse.csr_matrix(matrix))
    return matrix, index


TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODES = ("question", "entity_only", "task_id", "question_only")


def encode_texts(texts, revision=None, batch_size=256):
    """Frozen, revision-pinned MiniLM with attention-mask mean pooling.

    Frozen by construction: computed under no_grad and handed to the heads as
    fixed vectors, so no gradient can reach the text encoder.
    """
    import torch
    from huggingface_hub import model_info
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    revision = revision or model_info(TEXT_MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=revision)
    encoder = AutoModel.from_pretrained(TEXT_MODEL, revision=revision).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=128,
                                return_tensors="pt")
            states = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (states * mask).sum(1) / mask.sum(1)
            out.append(nn.functional.normalize(pooled, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32), revision


def score_by_assay(rows, probabilities) -> dict:
    """Per-assay AUROC, log loss and log-loss skill, plus macro averages.

    macro_auroc is the selection metric; the two loss figures are reported but
    not selected on. Raw macro_log_loss is minimised among constants by the base
    rate, and macro_log_loss_skill, which fixes that, is calibration dominated
    and inherits the same inversion on unseen assays. See aster.real.skill.
    """
    from sklearn.metrics import roc_auc_score

    grouped = defaultdict(lambda: ([], []))
    for row, p in zip(rows, probabilities):
        grouped[row["task"]][0].append(row["label"])
        grouped[row["task"]][1].append(float(p))
    per_assay, aurocs, losses, skills = {}, [], [], []
    for assay, (labels, probs) in sorted(grouped.items()):
        entry = {"n": len(labels), "positive_rate": round(float(np.mean(labels)), 4)}
        entry.update(classification_skill(labels, probs))
        if len(set(labels)) > 1:
            entry["auroc"] = float(roc_auc_score(labels, probs))
            aurocs.append(entry["auroc"])
        losses.append(entry["log_loss"])
        skills.append(entry["log_loss_skill"])
        per_assay[assay] = entry
    defined = [s for s in skills if s is not None]
    return {"assays": per_assay, "macro_log_loss": float(np.mean(losses)),
            "macro_log_loss_skill": macro_skill(skills),
            "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
            "assays_scored": len(per_assay), "assays_with_both_classes": len(aurocs),
            "assays_with_skill": len(defined)}


def _tensors(rows, features, index, texts, assay_order, train_assays):
    import torch

    position = {a: i for i, a in enumerate(assay_order)}
    known = {a: i + 1 for i, a in enumerate(train_assays)}
    molecule = np.array([index[r["smiles"]] for r in rows])
    return (torch.from_numpy(features), torch.from_numpy(molecule),
            torch.from_numpy(texts[[position[r["task"]] for r in rows]]),
            torch.tensor([known.get(r["task"], 0) for r in rows], dtype=torch.long),
            torch.tensor([float(r["label"]) for r in rows], dtype=torch.float32))


def _predict(model, features, molecule, q, ids, batch_size=8192):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(molecule), batch_size):
            sl = slice(start, start + batch_size)
            logits = model(features[molecule[sl]], q[sl], ids[sl])[:, 0]
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_multitask(directory: Path, out: Path, cache_dir: Path, seeds=(42, 43, 44),
                    epochs=8, assays_per_step=64, rows_per_assay=8, steps_per_epoch=400,
                    hidden=64, modes=MODES):
    """Train the same four controls with hundreds of questions instead of four."""
    import torch
    from torch import nn

    from aster.real.peptiverse_transfer import TransferModel

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}; choose a fresh run directory")
    if set(modes) - set(MODES) or not modes or not seeds:
        raise ValueError("Invalid configuration")
    rows, manifest = load_multitask(directory)
    rows = [r for r in rows if r["split"] != "test"]  # test is not read here at all
    train_assays = manifest["assay_splits"]["train"]
    assay_order = sorted({r["task"] for r in rows})
    features, index = molecule_features(rows, cache_dir)
    texts, revision = encode_texts([next(r["question"] for r in rows if r["task"] == a)
                                    for a in assay_order])
    x, molecule, q, ids, y = _tensors(rows, features, index, texts, assay_order, train_assays)
    by_assay = {a: np.array([i for i, r in enumerate(rows)
                             if r["task"] == a and r["split"] == "train"])
                for a in train_assays}
    validation = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    val_rows = [rows[i] for i in validation]
    val_index = np.array(validation)
    out.mkdir(parents=True, exist_ok=True)
    config = {"benchmark_files": manifest["files"], "source": manifest["source"],
              "assay_counts": {s: len(v) for s, v in manifest["assay_splits"].items()},
              "text_model": TEXT_MODEL, "text_revision": revision,
              "feature_version": "morgan-2048-chiral", "entity_dimension": int(features.shape[1]),
              "seeds": list(seeds), "epochs": epochs, "hidden": hidden, "modes": list(modes),
              "assays_per_step": assays_per_step, "rows_per_assay": rows_per_assay,
              "steps_per_epoch": steps_per_epoch, "learning_rate": 0.001,
              "encoder_training": "Frozen Morgan features; frozen MiniLM; heads trained from scratch",
              "selection": "macro validation AUROC over held-out assays; macro log-loss"
                           " skill reported alongside as a calibration gate",
              "selection_metric": "macro_auroc",
              "selection_note": "Neither loss metric is safe to select on here. Raw macro log"
                                " loss is minimised among constants by the base rate, so it"
                                " rewarded giving up. Skill fixes that bound but is calibration"
                                " dominated, and these models are overconfident: question_only"
                                " outscores question on skill while sitting at AUROC 0.5000."
                                " AUROC is invariant to monotone rescaling, so overconfidence"
                                " cannot hide discrimination, and a constant is pinned at 0.5.",
              "calibration_gate": "macro_log_loss_skill > 0 is required before treating any"
                                  " output as a probability rather than a ranking.",
              "sampling": "Each step samples assays uniformly then rows within them, so assays"
                          " carry equal weight regardless of label count",
              "test_evaluated": False,
              "software": {"torch": torch.__version__, "numpy": np.__version__}}
    with (out / "protocol.json").open("w") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
    reports = {}
    torch.set_num_threads(4)
    for seed in seeds:
        for mode in modes:
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = TransferModel(features.shape[1], texts.shape[1], len(train_assays), mode, hidden)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            best, stale, history = None, 0, []
            for epoch in range(epochs):
                model.train()
                running = []
                for _ in range(steps_per_epoch):
                    picked = rng.choice(len(train_assays),
                                        size=min(assays_per_step, len(train_assays)), replace=False)
                    batch = np.concatenate([rng.choice(by_assay[train_assays[p]],
                                                       size=min(rows_per_assay,
                                                                len(by_assay[train_assays[p]])),
                                                       replace=False) for p in picked])
                    optimizer.zero_grad()
                    logits = model(x[molecule[batch]], q[batch], ids[batch])[:, 0]
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y[batch])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    running.append(float(loss.detach()))
                probs = _predict(model, x, molecule[val_index], q[val_index], ids[val_index])
                score = score_by_assay(val_rows, probs)
                if score["macro_auroc"] is None:
                    raise ValueError("No validation assay has both classes; AUROC is undefined "
                                     "and there is nothing to select on")
                history.append({"epoch": epoch + 1, "training_loss": float(np.mean(running)),
                                "validation_macro_log_loss": score["macro_log_loss"],
                                "validation_macro_log_loss_skill": score["macro_log_loss_skill"],
                                "validation_macro_auroc": score["macro_auroc"]})
                print(f"{mode} seed={seed} epoch={epoch + 1}: "
                      f"skill={score['macro_log_loss_skill']:+.4f} "
                      f"loss={score['macro_log_loss']:.4f} auroc={score['macro_auroc']:.4f}",
                      flush=True)
                if best is None or score["macro_auroc"] > best[0]:
                    best, stale = (score["macro_auroc"], copy_state(model), score, epoch + 1), 0
                else:
                    stale += 1
                if stale >= 3:
                    break
            name = f"{mode}_s{seed}"
            checkpoint = out / f"{name}.pt"
            torch.save({"state_dict": best[1], "mode": mode, "d_entity": int(features.shape[1]),
                        "d_text": int(texts.shape[1]), "hidden": hidden,
                        "n_train_tasks": len(train_assays)}, checkpoint)
            reports[name] = {"mode": mode, "seed": seed, "best_epoch": best[3],
                             "history": history,
                             "validation_macro_log_loss": best[2]["macro_log_loss"],
                             "validation_macro_log_loss_skill": best[2]["macro_log_loss_skill"],
                             "validation_macro_auroc": best[2]["macro_auroc"],
                             "checkpoint_sha256": file_digest(checkpoint)}
            with (out / "selection.json").open("w") as stream:
                json.dump({"protocol": config, "runs": reports}, stream, indent=2, sort_keys=True)
    return {"protocol": config, "runs": reports}


def copy_state(model):
    import copy
    return copy.deepcopy(model.state_dict())
