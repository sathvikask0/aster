"""PeptiVerse loader, with the audit's findings enforced in code.

Findings this module acts on (see reports/PEPTIVERSE_AUDIT.md):

  * All 9,668 solubility negatives are also nephrotoxicity negatives, labelled
    0 in both, with zero exceptions -- 52% of the solubility task. Train on
    nephrotoxicity, hold out solubility, and half the test set is answerable by
    recalling peptides already seen. That is fake transfer arriving through the
    data rather than the architecture, so `drop_shared` removes it by default.

  * Only four tasks carry amino-acid sequences. Toxicity, Caco2 and PAMPA are
    SMILES-only and ESM-2 cannot read them.

  * Chance is not the baseline. Composition alone scores 78% on penetrance
    against a 50% majority class. Every claim is a margin over the measured
    shortcut, never over chance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")

# Question and answer wording matter: the text encoder is the thing under test,
# so answers are verbalized per task rather than a bare "yes"/"no". With a
# shared yes/no the answer embedding carries no task signal at all.
TASKS = {
    "hemolysis": dict(
        path="hemolysis/hemo_meta_with_split.csv",
        question="Does this peptide damage red blood cells?",
        answers=("it damages red blood cells", "it leaves red blood cells intact"),
    ),
    "solubility": dict(
        path="solubility/sol_meta_with_split.csv",
        question="Does this peptide dissolve well in water?",
        answers=("it dissolves well in water", "it does not dissolve well in water"),
    ),
    "nephrotoxicity": dict(
        path="nf/nf_meta_with_split.csv",
        question="Is this peptide toxic to the kidneys?",
        answers=("it is toxic to the kidneys", "it is not toxic to the kidneys"),
    ),
    "penetrance": dict(
        path="permeability_penetrance/permeability_meta_with_split.csv",
        question="Can this peptide cross the cell membrane?",
        answers=("it crosses the cell membrane", "it cannot cross the cell membrane"),
    ),
}

# label 1 == the first answer option
POSITIVE_FIRST = True


@dataclass(frozen=True)
class Example:
    sequence: str
    task: str
    question: str
    options: tuple[str, ...]
    label: int
    source_split: str


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    ok = df["sequence"].map(lambda s: len(s) > 1 and set(s) <= AA)
    return df[ok & df["label"].notna()]


def load(root: str | Path, tasks=None, drop_shared=True, max_len=None,
         verbose=True) -> list[Example]:
    root = Path(root).expanduser()
    tasks = list(tasks or TASKS)
    frames = {}

    for t in tasks:
        spec = TASKS[t]
        p = root / spec["path"]
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download with:\n"
                "  hf download ChatterjeeLab/PeptiVerse_data --repo-type dataset "
                f"--include '*.csv' --local-dir {root}")
        frames[t] = _clean(pd.read_csv(p))
        if verbose:
            print(f"  loaded {t}: {len(frames[t])} rows", flush=True)

    # --- enforce the audit finding -------------------------------------
    # Solubility's negative class IS a subset of nephrotoxicity's, all labelled
    # 0 in both. Dropping the shared rows leaves solubility single-class, so the
    # rows cannot be salvaged: the two are not independent tasks. Use one.
    if drop_shared and {"solubility", "nephrotoxicity"} <= set(frames):
        drop = "nephrotoxicity"
        frames.pop(drop); tasks = [t for t in tasks if t != drop]
        if verbose:
            print(f"  dropped task '{drop}': solubility's negatives are a "
                  "subset of it, so keeping both fakes transfer", flush=True)

    out: list[Example] = []
    for t, df in frames.items():
        spec = TASKS[t]
        opts = tuple(spec["answers"])
        for seq, lab, sp in zip(df.sequence, df.label, df.get("split", "train")):
            if max_len and len(seq) > max_len:
                continue
            out.append(Example(seq, t, spec["question"], opts,
                               0 if int(lab) == 1 else 1, str(sp)))

    if verbose:
        print("\n  task composition:")
        for t in tasks:
            e = [x for x in out if x.task == t]
            if not e:
                continue
            c = Counter(x.label for x in e)
            maj = max(c.values()) / len(e)
            print(f"    {t:16s} n={len(e):6d}  majority={maj:.1%}")
    return out


# ------------------------------------------------------------------ shortcuts

def composition_matrix(seqs, with_length=True) -> np.ndarray:
    """Amino-acid frequencies (+ length). The cheap shortcut to beat."""
    order = sorted(AA)
    X = np.zeros((len(seqs), len(order) + (1 if with_length else 0)),
                 dtype=np.float32)
    idx = {a: i for i, a in enumerate(order)}
    for i, s in enumerate(seqs):
        for ch in s:
            X[i, idx[ch]] += 1
        X[i, :len(order)] /= max(len(s), 1)
        if with_length:
            X[i, -1] = len(s) / 200.0
    return X


def composition_ceiling(examples, seed=0) -> dict[str, float]:
    """Per-task accuracy using ONLY amino-acid composition.

    This is the real zero. On penetrance it is 78% against a 50% majority
    class, so a model scoring 80% there has demonstrated almost nothing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    out = {}
    for t in sorted({e.task for e in examples}):
        print(f"    fitting composition baseline for {t} ...", flush=True)
        sub = [e for e in examples if e.task == t]
        y = np.array([e.label for e in sub])
        maj = max(np.mean(y == 0), np.mean(y == 1))
        X = composition_matrix([e.sequence for e in sub])
        try:
            acc = cross_val_score(
                LogisticRegression(max_iter=2000), X, y, cv=4,
                scoring="accuracy").mean()
        except Exception:
            acc = float(maj)
        out[t] = {"majority": float(maj), "composition": float(acc),
                  "ceiling": float(max(maj, acc))}
    return out
