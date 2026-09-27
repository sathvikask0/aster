"""Audit a peptide property table before it is allowed near a model.

Answers the two questions that decide whether PeptiVerse can support Aster's
transfer claim at all:

  1. How many peptides convert to the 20-letter amino-acid alphabet? Anything
     that does not cannot go through ESM-2. Non-natural residues are not a
     conversion bug -- those molecules have no protein-letter form, and any
     tool that returns one for them is fabricating it.

  2. Is the label distribution usable? The Arc sample in this repo has label
     counts [1, 4997, 2]: 99.94% one class, so "always answer no change" scores
     99.94%. That is the failure this script exists to catch early.

It also reports duplicates and near-duplicates, because the same peptide under
two ids landing in both train and test is the cheapest way to manufacture a
result that is not real.

Usage:
    python scripts/audit_peptides.py hemolysis.parquet solubility.parquet
    python scripts/audit_peptides.py data/*.csv --smiles-col smiles
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:  # audit still runs, conversion section is skipped
    HAVE_RDKIT = False

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
AA1 = set(AA3.values())

SMILES_HINTS = ("smiles", "canonical_smiles", "peptide_smiles", "mol", "structure")
SEQ_HINTS = ("sequence", "seq", "peptide", "peptide_sequence", "aa_sequence")
LABEL_HINTS = ("label", "value", "target", "y", "activity", "hc50", "class",
               "solubility", "hemolysis", "toxicity", "affinity")


def pick(df, hints, exclude=()):
    for h in hints:
        for c in df.columns:
            if c.lower() == h and c not in exclude:
                return c
    for h in hints:
        for c in df.columns:
            if h in c.lower() and c not in exclude:
                return c
    return None


def load(path: Path) -> pd.DataFrame:
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv", ".txt"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    if path.suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"unsupported: {path}")


# ------------------------------------------------------------------ chemistry

def smiles_to_sequence(smi: str) -> tuple[str | None, str]:
    """Return (sequence, reason). sequence is None when there is no honest
    protein-letter form -- which is an answer, not a failure to be patched."""
    if not HAVE_RDKIT:
        return None, "rdkit_missing"
    if not isinstance(smi, str) or not smi.strip():
        return None, "empty"
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "unparseable"
    try:
        seq = Chem.MolToSequence(mol)
    except Exception:
        return None, "rdkit_error"
    if not seq:
        return None, "no_peptide_backbone"
    if any(c not in AA1 for c in seq):
        return None, "non_natural_residue"
    return seq, "ok"


def is_plain_sequence(s) -> bool:
    return (isinstance(s, str) and len(s) > 1
            and all(c in AA1 for c in s.strip().upper()))


# --------------------------------------------------------------------- audit

def audit(path: Path, smiles_col=None, seq_col=None, label_col=None) -> dict:
    df = load(path)
    out: dict = {"file": path.name, "rows": int(len(df)),
                 "columns": list(map(str, df.columns))[:60]}

    smiles_col = smiles_col or pick(df, SMILES_HINTS)
    seq_col = seq_col or pick(df, SEQ_HINTS, exclude={smiles_col})
    label_col = label_col or pick(df, LABEL_HINTS)
    out["detected"] = {"smiles": smiles_col, "sequence": seq_col, "label": label_col}

    # ---- labels: the Arc failure mode ----
    if label_col is not None:
        s = df[label_col]
        lab = {"column": label_col, "dtype": str(s.dtype),
               "missing": int(s.isna().sum())}
        v = s.dropna()
        if pd.api.types.is_numeric_dtype(v) and v.nunique() > 12:
            q = v.quantile([0, .05, .25, .5, .75, .95, 1]).round(4)
            lab.update(kind="continuous", unique=int(v.nunique()),
                       quantiles={str(k): float(x) for k, x in q.items()})
            # A cutoff is a scientific decision. Report what the median split
            # WOULD give, explicitly as a diagnostic, not as a recommendation.
            lab["median_split_balance"] = round(float((v > v.median()).mean()), 4)
        else:
            counts = Counter(v.tolist())
            n = sum(counts.values())
            major = max(counts.values()) / n if n else 0.0
            lab.update(kind="categorical",
                       counts={str(k): int(c) for k, c in counts.most_common(12)},
                       majority_class_fraction=round(major, 4),
                       always_majority_accuracy=round(major, 4))
            lab["USABLE"] = major < 0.90
            if major >= 0.90:
                lab["verdict"] = (f"{major:.2%} one class -- always predicting it "
                                  "scores that. Not usable as-is.")
        out["label"] = lab

    # ---- convertibility: can ESM-2 read these at all? ----
    if seq_col is not None and df[seq_col].map(is_plain_sequence).mean() > 0.5:
        seqs = df[seq_col].astype(str).str.strip().str.upper()
        out["sequences"] = {"source": "already amino-acid", "column": seq_col,
                            "usable": int(seqs.map(is_plain_sequence).sum())}
    elif smiles_col is not None:
        reasons, seqs = Counter(), []
        col = df[smiles_col].tolist()
        for smi in col:
            seq, why = smiles_to_sequence(smi)
            reasons[why] += 1
            seqs.append(seq)
        ok = reasons["ok"]
        out["sequences"] = {
            "source": f"converted from {smiles_col}",
            "convertible": ok,
            "convertible_fraction": round(ok / max(len(col), 1), 4),
            "reasons": dict(reasons.most_common()),
            "note": ("non_natural_residue means the molecule has no "
                     "protein-letter form; ESM-2 cannot represent it"),
        }
        seqs = [s for s in seqs if s]
    else:
        seqs = []
        out["sequences"] = {"source": None, "note": "no sequence or smiles column"}

    if seq_col is not None and "seqs" not in dir():
        seqs = [s for s in df[seq_col].astype(str).str.strip().str.upper()
                if is_plain_sequence(s)]

    # ---- duplicates: the cheapest way to fake a result ----
    if seqs:
        c = Counter(seqs)
        lens = np.array([len(s) for s in seqs])
        out["duplicates"] = {
            "n_sequences": len(seqs), "unique": len(c),
            "exact_duplicate_rows": len(seqs) - len(c),
            "duplicate_fraction": round(1 - len(c) / len(seqs), 4),
            "most_repeated": [[s[:30], n] for s, n in c.most_common(3) if n > 1],
        }
        out["length"] = {
            "min": int(lens.min()), "max": int(lens.max()),
            "median": float(np.median(lens)), "mean": round(float(lens.mean()), 2),
            "over_1024": int((lens > 1024).sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--smiles-col"), ap.add_argument("--seq-col")
    ap.add_argument("--label-col")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/peptide_audit.json"))
    a = ap.parse_args()

    if not HAVE_RDKIT:
        print("WARNING: rdkit missing -- conversion check skipped "
              "(pip install rdkit)\n", file=sys.stderr)

    reports, blockers = [], []
    for f in a.files:
        if not f.exists():
            print(f"missing: {f}", file=sys.stderr)
            continue
        r = audit(f, a.smiles_col, a.seq_col, a.label_col)
        reports.append(r)

        print(f"\n=== {r['file']}  ({r['rows']} rows) ===")
        print(f"  detected: {r['detected']}")
        if "label" in r:
            L = r["label"]
            if L.get("kind") == "categorical":
                print(f"  labels: {L['counts']}")
                print(f"          majority={L['majority_class_fraction']:.2%} "
                      f"-> always-majority scores {L['always_majority_accuracy']:.2%}")
                if not L.get("USABLE", True):
                    blockers.append(f"{r['file']}: {L['verdict']}")
            else:
                print(f"  labels: continuous, {L['unique']} distinct, "
                      f"quantiles {L['quantiles']}")
                print(f"          (median split would give "
                      f"{L['median_split_balance']:.2%} positive -- diagnostic "
                      "only; pick the cutoff from the assay)")
        S = r.get("sequences", {})
        if "convertible_fraction" in S:
            print(f"  convertible to amino acids: {S['convertible']}"
                  f" ({S['convertible_fraction']:.1%})")
            print(f"  reasons: {S['reasons']}")
            if S["convertible_fraction"] < 0.5:
                blockers.append(
                    f"{r['file']}: only {S['convertible_fraction']:.1%} readable "
                    "by ESM-2 -- needs a chemical encoder instead")
        elif S.get("source"):
            print(f"  sequences: {S['source']}")
        if "duplicates" in r:
            D, Ln = r["duplicates"], r["length"]
            print(f"  duplicates: {D['exact_duplicate_rows']} rows "
                  f"({D['duplicate_fraction']:.1%}); unique {D['unique']}")
            print(f"  length: {Ln['min']}-{Ln['max']} (median {Ln['median']})")
            if D["duplicate_fraction"] > 0.02:
                blockers.append(
                    f"{r['file']}: {D['duplicate_fraction']:.1%} duplicates -- "
                    "split by identity cluster, not randomly")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {a.out}")

    if blockers:
        print("\nBLOCKERS -- do not train until these are resolved:")
        for b in blockers:
            print("  - " + b)
    else:
        print("\nNo blockers found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
