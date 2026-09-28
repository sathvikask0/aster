"""Eight assay endpoints covering the five PeptiVerse drug-property questions.

This is a supervised, mixed classification/regression benchmark. It does not
reuse the AMP binary-only Example or claim zero-shot question transfer. Inputs
are chemical peptide structures and (for affinity) a target protein sequence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd

REPO = "ChatterjeeLab/PeptiVerse_data"
REVISION = "6292abb1be90b7661245dcd0c2505f37c862a750"
VERSION = 1
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
SPLITS = ("train", "validation", "test")

# SHA-256 of the CSV bytes at the pinned upstream revision. Download only
# metadata, never the 62 GB of pretrained embeddings in the dataset repository.
TASKS = {
    "solubility": dict(
        category="dissolve", kind="classification", units="binary",
        question="Does this peptide dissolve in water?",
        options=["It is not soluble in water.", "It is soluble in water."],
        path="solubility/sol_meta_with_split_with_smiles.csv",
        sha256="c3df4a2105c714b901528808a8b8e06a1e1d8f79defb956f273b7a4c05bbdaf9",
        label="label", smiles="smiles", sequence="sequence"),
    "penetrance": dict(
        category="cross_membrane", kind="classification", units="binary",
        question="Can this peptide penetrate a cell membrane?",
        options=["It is not cell penetrating.", "It is cell penetrating."],
        path="permeability_penetrance/permeability_smiles_meta_with_split.csv",
        sha256="cbece0b3b8345cae1ce6fe2e9a1a10ddd5320bae18c3a7a3f958b97b98979796",
        label="label", smiles="smiles", sequence="sequence"),
    "pampa": dict(
        category="cross_membrane", kind="regression", units="source log permeability",
        question="What is this peptide's log permeability in the PAMPA assay?",
        path="permeability_pampa/pampa_meta_with_split.csv",
        sha256="d04d3767f03a4846003f404db6d03f8392ef9ad73830546064769beded3cfa80",
        label="PAMPA", smiles="SMILES"),
    "caco2": dict(
        category="cross_membrane", kind="regression", units="source log permeability",
        question="What is this peptide's log permeability in the Caco-2 assay?",
        path="permeability_caco2/caco2_meta_with_split.csv",
        sha256="fa8f0fb32da50e69eafd3e585d68c7876710951fc54c20dd85c2501745dbb38c",
        label="Caco2", smiles="SMILES"),
    "half_life": dict(
        category="survive", kind="regression", units="hours",
        question="What is this peptide's reported half-life in hours?",
        path="half_life/halflife_meta_with_split.csv",
        sha256="68da34e700019243ae2dc3d1983508fe5e2f780927e849a8096b967013cd17cd",
        label="half_life_hours", smiles="SMILES", sequence="sequence",
        training_transform="log10"),
    "hemolysis": dict(
        category="harm", kind="classification", units="binary",
        question="Does this peptide damage red blood cells?",
        options=["It is non-hemolytic.", "It is hemolytic."],
        path="hemolysis/hemo_smiles_meta_with_split.csv",
        sha256="089f7c031410cc16b2669d41a5909d5210eb372b8ee6343d3cfab4718e66bd7d",
        label="label", smiles="SMILES", sequence="sequence"),
    "toxicity": dict(
        category="harm", kind="classification", units="binary",
        question="Does this peptide exhibit toxicity in the source toxicity dataset?",
        options=["It is labelled non-toxic.", "It is labelled toxic."],
        path="toxicity/tox_meta_with_split.csv",
        sha256="757730999cf644f543b57b01e37b0352fadc36f4f1cc0a4d6694f33559c71683",
        label="Label", smiles="SMILES"),
    "binding_affinity": dict(
        category="bind_target", kind="regression", units="source p-affinity score",
        question="What is this peptide's reported binding-affinity score for the supplied target protein?",
        path="binding_affinity/binding_affinity_smiles_meta_with_split.csv",
        sha256="3aee738ef2b17343ae69723a75473821b4188a196a55dacd0286ec47d065d531",
        label="affinity", smiles="smiles_sequence", sequence="seq2",
        target="seq1"),
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_sources(root: Path) -> dict:
    """Fetch pinned files atomically; refuse modified or partial local copies."""
    root = Path(root)
    manifest = {}
    for task, spec in TASKS.items():
        path = root / spec["path"]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{spec['path']}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".csv.part")
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if file_digest(temporary) != spec["sha256"]:
                    raise ValueError(f"Checksum mismatch downloading {spec['path']}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if file_digest(path) != spec["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}; use the pinned upstream CSV")
        manifest[task] = {"path": spec["path"], "url": url,
                          "sha256": spec["sha256"], "bytes": path.stat().st_size}
    return manifest


@lru_cache(maxsize=50000)
def canonical_smiles(value: str) -> str | None:
    from rdkit import Chem, rdBase
    # Bad source rows are reported in the audit, not printed as huge SMILES logs.
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(value) if value else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalize_frame(task: str, frame: pd.DataFrame) -> tuple[list[dict], dict]:
    """Keep input features separate from outcomes and audit every dropped row.

    Standard sequences are never obtained by uppercasing D-residue notation.
    Nonstandard source sequences remain provenance and conservative split keys;
    their actual model input is the chemical structure.
    """
    spec = TASKS[task]
    required = {spec["label"], spec["smiles"]}
    required.update(spec[k] for k in ("sequence", "target") if k in spec)
    if missing := required - set(frame):
        raise ValueError(f"{task}: missing columns {sorted(missing)}")
    dropped, rows = Counter(), []
    for position, item in enumerate(frame.to_dict("records")):
        value = pd.to_numeric(item[spec["label"]], errors="coerce")
        if not np.isfinite(value):
            dropped["nonfinite_label"] += 1
            continue
        if spec["kind"] == "classification" and value not in (0, 1):
            raise ValueError(f"{task} row {position}: label must be 0 or 1")
        if task == "half_life" and value <= 0:
            dropped["nonpositive_half_life"] += 1
            continue
        measurement = _text(item.get("affinity_measure"))
        if task == "binding_affinity" and re.search(r"[<>≤≥~≈]", measurement):
            dropped["censored_affinity"] += 1
            continue
        structure = canonical_smiles(_text(item[spec["smiles"]]))
        if structure is None:
            dropped["invalid_smiles"] += 1
            continue
        raw_sequence = _text(item.get(spec.get("sequence", "")))
        sequence = raw_sequence if raw_sequence and set(raw_sequence) <= AA else None
        target = _text(item.get(spec.get("target", ""))) or None
        if "target" in spec and (not target or set(target) - (AA | {"X", "B", "Z", "U", "O"})):
            dropped["invalid_target_sequence"] += 1
            continue
        assay_match = re.match(r"(Kd|Ki|IC50|EC50|Ka)", measurement, re.I)
        assay = assay_match.group(1).lower() if assay_match else None
        peptide_id = digest("smiles:" + structure)
        source = {"file": spec["path"], "row": position, "csv_line": position + 2,
                  "source_split": _text(item.get("split")) or None,
                  "source_id": _text(item.get("id", item.get("PDB_id"))) or None}
        rows.append({
            "task": task, "category": spec["category"], "kind": spec["kind"],
            "question": spec["question"], "options": spec.get("options"),
            "label": int(value) if spec["kind"] == "classification" else float(value),
            "units": spec["units"], "smiles": structure, "sequence": sequence,
            "source_sequence": raw_sequence or None, "target_sequence": target,
            "target_id": digest(target) if target else None,
            "peptide_id": peptide_id, "assay_type": assay,
            "source_clusters": [f"{task}:{column}:{_text(item[column])}"
                                for column in ("cluster_id", "mmseqs_cluster")
                                if column in item and _text(item[column])],
            "sources": [source],
        })
    # One endpoint per chemical entity/target/assay type. Conflicting categorical
    # labels have no defensible majority rule; exclude them. Replicate continuous
    # values use the median in original units, with range and provenance retained.
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["peptide_id"], row["target_id"], row["assay_type"])].append(row)
    result, duplicate_rows, conflicts, variable_replicates = [], 0, 0, 0
    for key, group in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        values = [r["label"] for r in group]
        if spec["kind"] == "classification" and len(set(values)) > 1:
            conflicts += len(group)
            continue
        row = group[0]
        row["sequence_aliases"] = sorted({r["source_sequence"] for r in group if r["source_sequence"]})
        row["sources"] = [s for r in group for s in r["sources"]]
        row["source_clusters"] = sorted({c for r in group for c in r["source_clusters"]})
        if spec["kind"] == "regression":
            row["label"] = float(np.median(values))
            row["replicate_range"] = [min(values), max(values)]
            variable_replicates += int(len(set(values)) > 1)
        row["id"] = digest(json.dumps([VERSION, task, *key]))
        duplicate_rows += len(group) - 1
        result.append(row)
    return result, {"source_rows": len(frame), "retained_examples": len(result),
                    "dropped": dict(dropped), "conflicting_label_rows": conflicts,
                    "duplicate_rows_collapsed": duplicate_rows,
                    "regression_groups_with_variable_replicates": variable_replicates,
                    "source_columns": list(frame),
                    "source_splits": frame["split"].value_counts().to_dict() if "split" in frame else {},
                    "sequence_available": sum(r["sequence"] is not None for r in result)}


def assign_splits(rows: list[dict], seed: int = 42,
                  fractions=(0.70, 0.15, 0.15), source_clusters: bool = False) -> list[dict]:
    """Global connected components prevent cross-task and cross-format leakage.

    Sequence aliases conservatively join differently modified forms with the
    same source notation. Optional upstream clusters are namespaced by task:
    cluster 1 in toxicity is not cluster 1 in hemolysis. Components are assigned
    by a seeded hash, independent of labels. Ratios are approximate.
    """
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Need three positive split fractions summing to one")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    keys_by_row = []
    for row in rows:
        keys = ["molecule:" + row["peptide_id"]]
        keys.extend("sequence:" + digest(s) for s in row["sequence_aliases"])
        if source_clusters:
            keys.extend("cluster:" + c for c in row["source_clusters"])
        for key in keys:
            union(keys[0], key)
        keys_by_row.append(keys)
    out = []
    for row, keys in zip(rows, keys_by_row):
        group = digest(find(keys[0]))
        u = int(digest(f"{seed}:{group}")[:16], 16) / 2**64
        split = "train" if u < fractions[0] else "validation" if u < sum(fractions[:2]) else "test"
        out.append({**row, "group_id": group, "split": split})
    return sorted(out, key=lambda r: (r["task"], r["id"]))


def validate_benchmark(rows: list[dict], tasks=None) -> dict:
    """Fail closed on missing tasks, bad targets, duplicate IDs or split leakage."""
    tasks = set(TASKS if tasks is None else tasks)
    if not rows or {r["task"] for r in rows} != tasks:
        raise ValueError("Benchmark must contain every requested task")
    seen_ids, identities = set(), defaultdict(set)
    for row in rows:
        if row["id"] in seen_ids:
            raise ValueError("Duplicate example id")
        seen_ids.add(row["id"])
        if row["split"] not in SPLITS or not math.isfinite(row["label"]):
            raise ValueError("Invalid split or nonfinite label")
        spec = TASKS[row["task"]]
        if row["kind"] != spec["kind"] or row["question"] != spec["question"]:
            raise ValueError("Task schema mismatch")
        if row["kind"] == "classification" and (row["label"] not in (0, 1) or row["options"] != spec["options"]):
            raise ValueError("Classification label/answer mismatch")
        if row["task"] == "half_life" and row["label"] <= 0:
            raise ValueError("Half-life must be positive")
        if row["task"] == "binding_affinity" and not row["target_sequence"]:
            raise ValueError("Binding affinity requires a target protein")
        if not row["smiles"] or not row["sources"]:
            raise ValueError("Missing structure or provenance")
        keys = ["group:" + row["group_id"], "peptide:" + row["peptide_id"]]
        keys.extend("seq:" + s for s in row["sequence_aliases"])
        for key in keys:
            identities[key].add(row["split"])
    if any(len(s) != 1 for s in identities.values()):
        raise ValueError("Peptide/sequence/group leakage across splits")
    counts = {}
    for task in sorted(tasks):
        counts[task] = {}
        for split in SPLITS:
            subset = [r for r in rows if r["task"] == task and r["split"] == split]
            if not subset:
                raise ValueError(f"{task}: empty {split} split")
            labels = [r["label"] for r in subset]
            summary = {"n": len(subset), "groups": len({r["group_id"] for r in subset})}
            if TASKS[task]["kind"] == "classification":
                if set(labels) != {0, 1}:
                    raise ValueError(f"{task}: {split} must contain both classes")
                summary["class_counts"] = dict(Counter(labels))
            else:
                summary.update(label_min=min(labels), label_median=float(np.median(labels)), label_max=max(labels))
            counts[task][split] = summary
    return {"cross_split_identity_overlap": 0, "tasks": counts,
            "examples": len(rows), "groups": len({r["group_id"] for r in rows})}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_benchmark(root: Path, out: Path, seed=42, source_clusters=False) -> dict:
    import rdkit
    sources = fetch_sources(root)
    rows, quality = [], {}
    for task, spec in TASKS.items():
        print(f"Normalizing {task}...", flush=True)
        examples, quality[task] = normalize_frame(task, pd.read_csv(Path(root) / spec["path"]))
        rows.extend(examples)
    rows = assign_splits(rows, seed=seed, source_clusters=source_clusters)
    checks = validate_benchmark(rows)
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already contains a benchmark; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        path = out / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        files[path.name] = file_digest(path)
    manifest = {
        "benchmark": "peptiverse_five_questions", "version": VERSION,
        "source_repo": REPO, "source_revision": REVISION, "seed": seed,
        "rdkit_version": rdkit.__version__, "fractions": [0.70, 0.15, 0.15],
        "split_policy": "global_identity_and_source_clusters" if source_clusters else "global_identity",
        "tasks": TASKS, "sources": sources, "files": files,
        "quality": quality, "checks": checks,
        "limitations": [
            "Supervised task coverage; not a held-out-question transfer experiment.",
            "Identity-disjoint is not a sequence-homology or chemical-scaffold holdout.",
            "Targets may recur across binding splits; this is not a cold-target evaluation.",
            "Upstream train/val assignments are provenance only; these are new splits.",
            "Source assay conditions are incomplete; half-life is not guaranteed serum half-life.",
            "Affinity mixes source Kd/Ki/IC50-style endpoints, retained on the source score scale.",
            "Binary source labels do not establish measured inactivity or universal safety.",
            "Solubility includes long protein sequences; do not describe all rows as short peptides.",
            "Replicate regression values are medians in original units; inspect ranges for heterogeneity.",
            "Missing property labels remain missing; no imputation from absence.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def load_benchmark(directory: Path) -> tuple[list[dict], dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != VERSION or manifest["tasks"] != TASKS:
        raise ValueError("Unsupported benchmark schema")
    rows = []
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if file_digest(path) != manifest["files"][path.name]:
            raise ValueError(f"Benchmark checksum mismatch: {path}")
        subset = [json.loads(line) for line in path.read_text().splitlines()]
        if any(row["split"] != split for row in subset):
            raise ValueError("Row stored in the wrong split file")
        rows.extend(subset)
    validate_benchmark(rows)
    return rows, manifest
