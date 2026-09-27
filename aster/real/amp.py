"""AMP Multi-Task Benchmark Loader with Mechanistic Prompts.

Curates multi-pathogen antimicrobial assays from DRAMP, APD3, CAMP, etc.
Converts target classifications into biophysical/mechanistic language prompts
to test cross-task zero-shot transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "amp"


@dataclass(frozen=True)
class Example:
    sequence: str
    task: str
    question: str
    options: tuple[str, ...]
    label: int
    split: str  # "train", "val", "test"


def build_mechanistic_prompt(pathogen: str, p_type: str, desc: str) -> tuple[str, tuple[str, str]]:
    """Build a biophysical prompt incorporating cell-envelope and membrane mechanics."""
    # Clean description to first 1-2 informative sentences
    clean_desc = desc.strip().replace('"', '')
    first_sentence = clean_desc.split(".")[0].strip()

    if "Gram-negative" in str(p_type):
        q = (
            f"Does this peptide carry cationic amphipathic structure capable of disrupting the "
            f"negatively-charged lipopolysaccharide (LPS) outer membrane and thin peptidoglycan layer of "
            f"Gram-negative {pathogen} ({first_sentence})?"
        )
    elif "Gram-positive" in str(p_type):
        q = (
            f"Does this peptide carry sufficient positive charge and hydrophobicity to traverse the "
            f"thick porous peptidoglycan cell wall and lyse the cytoplasmic membrane of "
            f"Gram-positive {pathogen} ({first_sentence})?"
        )
    elif "Fungus" in str(p_type) or "yeast" in clean_desc.lower() or "mold" in clean_desc.lower():
        q = (
            f"Does this peptide bind and permeabilize the chitin and beta-glucan cell wall or "
            f"ergosterol membrane barrier of fungal pathogen {pathogen} ({first_sentence})?"
        )
    else:
        q = (
            f"Does this peptide disrupt the microbial membrane barrier and inhibit "
            f"growth of pathogenic {pathogen} ({first_sentence})?"
        )

    answers = (
        "it potently disrupts the membrane barrier and inhibits growth",
        "it fails to disrupt the microbial membrane",
    )
    return q, answers


def load_amp_benchmark(
    data_dir: Path | str = DATA_DIR,
    min_samples: int = 300,
    max_len: int = 60,
    min_len: int = 6,
    test_tasks: tuple[str, ...] | None = None,
    seed: int = 42,
) -> tuple[list[Example], dict[str, dict]]:
    """Load and format the multi-task AMP dataset with mechanistic prompts."""
    data_dir = Path(data_dir)
    triples_path = data_dir / "peptide_pathogen_triple.csv"
    desc_path = data_dir / "pathogen_description.csv"

    if not triples_path.exists():
        raise FileNotFoundError(f"Missing {triples_path}. Run `uv run python scripts/fetch_amp_data.py` first.")

    df_triples = pd.read_csv(triples_path)
    df_desc = pd.read_csv(desc_path).set_index("pathogen")

    # Clean sequences
    df_triples["sequence"] = df_triples["sequence"].astype(str).str.strip().str.upper()
    valid_mask = df_triples["sequence"].map(
        lambda s: min_len <= len(s) <= max_len and set(s) <= AA
    )
    df = df_triples[valid_mask].copy()

    # Filter tasks by minimum sample count
    task_counts = df["pathogen"].value_counts()
    selected_tasks = task_counts[task_counts >= min_samples].index.tolist()

    # Default held-out tasks across diverse classes (Gram-negative, Gram-positive, Fungus)
    if test_tasks is None:
        test_tasks = (
            "K.pneumoniae",   # Gram-negative
            "S.epidermidis",  # Gram-positive
            "C.albicans",     # Fungus
            "S.typhimurium",  # Gram-negative
            "B.cereus",       # Gram-positive
        )

    # All unique valid peptides
    all_peptides = list(df["sequence"].unique())
    rng = np.random.default_rng(seed)

    task_metadata = {}
    examples = []

    # Map each peptide to its active pathogens
    active_map = df.groupby("sequence")["pathogen"].apply(set).to_dict()

    for task in selected_tasks:
        if task not in df_desc.index:
            p_type, desc = "Pathogen", f"Target pathogen {task}."
        else:
            p_type = df_desc.loc[task, "types"] if "types" in df_desc.columns else "Pathogen"
            desc = df_desc.loc[task, "description"] if "description" in df_desc.columns else ""

        question, options = build_mechanistic_prompt(task, p_type, desc)
        task_metadata[task] = {
            "type": p_type,
            "question": question,
            "options": options,
            "is_held_out": task in test_tasks,
        }

        # Positive examples for this task
        pos_seqs = sorted(set(df[df["pathogen"] == task]["sequence"]))

        # Balanced negative sampling: peptides active against other pathogens but not this one
        neg_candidates = [seq for seq in all_peptides if task not in active_map.get(seq, set())]
        rng.shuffle(neg_candidates)
        neg_seqs = neg_candidates[:len(pos_seqs)]

        # Determine train/val/test split
        split = "test" if task in test_tasks else "train"

        # Add positives (label 1 = options[0])
        for s in pos_seqs:
            # If training task, carve out 15% validation
            row_split = split
            if split == "train" and rng.random() < 0.15:
                row_split = "val"
            examples.append(Example(sequence=s, task=task, question=question, options=options, label=1, split=row_split))

        # Add negatives (label 0 = options[1])
        for s in neg_seqs:
            row_split = split
            if split == "train" and rng.random() < 0.15:
                row_split = "val"
            examples.append(Example(sequence=s, task=task, question=question, options=options, label=0, split=row_split))

    return examples, task_metadata
