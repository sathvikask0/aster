"""AMP Multi-Task Benchmark Loader with Mechanistic Prompts.

Curates multi-pathogen antimicrobial assays from DRAMP, APD3, CAMP, etc.
Converts target classifications into biophysical/mechanistic language prompts
to test cross-task zero-shot transfer.

Label semantics -- read this before trusting an accuracy from here. The source
tables are **presence-only**: a row says a peptide was reported active against a
pathogen, and there is no row anywhere that reports a measured inactivity. Every
negative in this benchmark is therefore a *presumed* negative, inferred from the
absence of a record, and absence is partly a function of which peptide-pathogen
pairs anyone bothered to assay. Three policies are offered:

  random   (default; same sampling policy as v0.2) any peptide without a
           record for this pathogen may be drawn as a negative, including one
           that was simply never tested against it.
  covered  a peptide may only be drawn as a negative for this pathogen if it has
           records for at least `min_coverage` other pathogens, so it is at
           least a well-studied peptide whose silence here is more informative.
  matched  as `covered`, but each negative is greedily matched to a positive by
           amino-acid composition and length to reduce that shortcut. Measure
           the remaining composition baseline on every constructed split.

  scrambled  a *positive control*, not a biology claim. Each negative is an
           order-shuffled copy of a positive: identical amino-acid composition
           and length, destroyed motif and helicity. Composition and majority
           ceilings are therefore 0.5 by construction, so any lift is
           sequence-order signal. Use it to establish that the encoder path can
           learn at all before reading a presumed-negative number. It does NOT
           test pathogen specificity -- the correct answer does not depend on
           the pathogen, so `entity_only` matching `dual` is expected here and
           is not evidence about the prompts.

None of these policies manufactures a measured negative. `covered` narrows the coverage
confound; it does not remove it. `scrambled` sidesteps it by not using presumed
negatives at all, at the cost of answering a different, easier question.

Labels are option indices: 0 selects the presumed-inactive answer and 1 the
reported-active answer. Version 1 put the answer texts in the opposite order,
so the training loss rewarded the wrong answer text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "amp"
LABEL_SEMANTICS_VERSION = 2


@dataclass(frozen=True)
class Example:
    sequence: str
    task: str
    question: str
    options: tuple[str, ...]
    label: int  # Index into options; 0 = presumed inactive, 1 = reported active.
    split: str  # "train", "val", "test"


def prompt_family(p_type: str, desc: str) -> str:
    """Which mechanism family a target's prompt is drawn from.

    Exposed because the mechanism-swap ablation has to hand a target a prompt
    from a *different* family, and a second copy of this branching would
    silently disagree with the prompts actually built.
    """
    clean_desc = str(desc).strip().replace('"', '').lower()
    if "Gram-negative" in str(p_type):
        return "gram_negative"
    if "Gram-positive" in str(p_type):
        return "gram_positive"
    if "Fungus" in str(p_type) or "yeast" in clean_desc or "mold" in clean_desc:
        return "fungal"
    return "generic"


def build_mechanistic_prompt(pathogen: str, p_type: str, desc: str) -> tuple[str, tuple[str, str]]:
    """Build a biophysical prompt incorporating cell-envelope and membrane mechanics."""
    # Clean description to first 1-2 informative sentences
    clean_desc = desc.strip().replace('"', '')
    first_sentence = clean_desc.split(".")[0].strip()
    family = prompt_family(p_type, desc)

    if family == "gram_negative":
        q = (
            f"Does this peptide carry cationic amphipathic structure capable of disrupting the "
            f"negatively-charged lipopolysaccharide (LPS) outer membrane and thin peptidoglycan layer of "
            f"Gram-negative {pathogen} ({first_sentence})?"
        )
    elif family == "gram_positive":
        q = (
            f"Does this peptide carry sufficient positive charge and hydrophobicity to traverse the "
            f"thick porous peptidoglycan cell wall and lyse the cytoplasmic membrane of "
            f"Gram-positive {pathogen} ({first_sentence})?"
        )
    elif family == "fungal":
        q = (
            f"Does this peptide bind and permeabilize the chitin and beta-glucan cell wall or "
            f"ergosterol membrane barrier of fungal pathogen {pathogen} ({first_sentence})?"
        )
    else:
        q = (
            f"Does this peptide disrupt the microbial membrane barrier and inhibit "
            f"growth of pathogenic {pathogen} ({first_sentence})?"
        )

    # Cross-entropy consumes option indices directly. Keep the answer order
    # aligned with the activity labels, including the reported positive prior.
    answers = (
        "it fails to disrupt the microbial membrane",
        "it potently disrupts the membrane barrier and inhibits growth",
    )
    return q, answers


def _feature_matrix(seqs: list[str]) -> np.ndarray:
    """20-dim amino-acid frequency plus a scaled length, for matching."""
    aa = sorted(AA)
    idx = {a: i for i, a in enumerate(aa)}
    mat = np.zeros((len(seqs), 21), dtype=np.float64)
    for i, s in enumerate(seqs):
        for a in s:
            j = idx.get(a)
            if j is not None:
                mat[i, j] += 1.0
        mat[i, :20] /= max(1, len(s))
        mat[i, 20] = len(s) / 60.0
    return mat


def scrambled_negatives(
    pos_seqs: list[str],
    rng,
    active_map: dict[str, set] | None = None,
    max_tries: int = 20,
) -> list[str]:
    """Order-shuffled copies of the positives: identical composition, broken motifs.

    A shuffle that reproduces the original string, collides with another decoy,
    or lands on any real peptide in the library is rejected and retried, so a
    decoy is never silently a true positive. A positive whose shuffles all fail
    (homopolymers, very short sequences) contributes no negative, exactly as an
    unmatched positive does under `matched`; `balance_tasks` then trims the
    surplus positive.
    """
    known = set(active_map or ())
    used: set[str] = set()
    chosen: list[str] = []
    for seq in pos_seqs:
        chars = list(seq)
        for _ in range(max_tries):
            rng.shuffle(chars)
            cand = "".join(chars)
            if cand != seq and cand not in known and cand not in used:
                used.add(cand)
                chosen.append(cand)
                break
    return chosen


def match_negatives(
    pos_seqs: list[str],
    candidates: list[str],
    caliper: float = 0.35,
) -> list[str]:
    """Greedy 1:1 nearest-neighbour matching of negatives to positives.

    Each positive claims its closest unused candidate in composition+length
    space. A positive whose nearest candidate is further than `caliper` adds no
    negative. The loader can separately downsample positives with balance_tasks;
    it does not retain pair identities or guarantee identical class composition.
    """
    if not pos_seqs or not candidates:
        return []
    P = _feature_matrix(pos_seqs)
    C = _feature_matrix(candidates)
    # Squared euclidean distances, positives x candidates.
    d2 = (P**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2.0 * P @ C.T
    np.maximum(d2, 0.0, out=d2)
    taken = np.zeros(len(candidates), dtype=bool)
    chosen: list[str] = []
    # Match the hardest positives first: those with the fewest close candidates.
    order = np.argsort(np.partition(d2, min(4, d2.shape[1] - 1), axis=1)[:, :5].sum(1))[::-1]
    for i in order:
        row = np.where(taken, np.inf, d2[i])
        j = int(np.argmin(row))
        if not np.isfinite(row[j]) or np.sqrt(row[j]) > caliper:
            continue
        taken[j] = True
        chosen.append(candidates[j])
    return chosen


def load_amp_benchmark(
    data_dir: Path | str = DATA_DIR,
    min_samples: int = 300,
    max_len: int = 60,
    min_len: int = 6,
    test_tasks: tuple[str, ...] | None = None,
    seed: int = 42,
    negative_policy: str = "random",
    min_coverage: int = 3,
    match_caliper: float = 0.35,
    balance_tasks: bool = False,
) -> tuple[list[Example], dict[str, dict]]:
    """Load and format the multi-task AMP dataset with mechanistic prompts.

    `balance_tasks` downsamples a task's positives to the number of negatives
    actually obtained. It is off by default because switching it on changes
    every published number. Leaving it off is not harmless: a high-prevalence
    pathogen exhausts its negative pool (E. coli has 11,203 positives and only
    2,811 peptides with no E. coli record), so its rows arrive 80-93% positive
    and the label prior a model reads off the training tasks does not hold on
    the balanced held-out ones. See `label_prior` in the returned metadata.
    """
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

    if negative_policy not in ("random", "covered", "matched", "scrambled"):
        raise ValueError(f"Unknown negative_policy: {negative_policy!r}")

    # Map each peptide to its active pathogens
    active_map = df.groupby("sequence")["pathogen"].apply(set).to_dict()

    # Under "covered", only well-assayed peptides are eligible as presumed
    # negatives, so that a missing record is less likely to mean "never tested".
    negative_pool = all_peptides
    if negative_policy in ("covered", "matched"):  # scrambled builds its own
        negative_pool = [
            seq for seq in all_peptides if len(active_map.get(seq, ())) >= min_coverage
        ]
        if not negative_pool:
            raise ValueError(
                f"No peptide has records for >= {min_coverage} pathogens; "
                "lower min_coverage or use negative_policy='random'."
            )

    for task in selected_tasks:
        if task not in df_desc.index:
            p_type, desc = "Pathogen", f"Target pathogen {task}."
        else:
            p_type = df_desc.loc[task, "types"] if "types" in df_desc.columns else "Pathogen"
            desc = df_desc.loc[task, "description"] if "description" in df_desc.columns else ""

        question, options = build_mechanistic_prompt(task, p_type, desc)
        task_metadata[task] = {
            "type": p_type,
            "prompt_family": prompt_family(p_type, desc),
            "question": question,
            "options": options,
            "is_held_out": task in test_tasks,
            "negative_policy": negative_policy,
            "negatives_are_presumed": negative_policy != "scrambled",
            "negatives_are_synthetic_decoys": negative_policy == "scrambled",
        }

        # Positive examples for this task
        pos_seqs = sorted(set(df[df["pathogen"] == task]["sequence"]))

        # Balanced presumed-negative sampling: peptides with no record for this
        # pathogen. These are unlabelled, not measured-inactive (see module docstring).
        # `scrambled` is the exception: it synthesises composition-identical decoys
        # instead, so no decoy can also be a positive for another task.
        if negative_policy == "scrambled":
            neg_seqs = scrambled_negatives(pos_seqs, rng, active_map)
        else:
            neg_candidates = [seq for seq in negative_pool
                              if task not in active_map.get(seq, set())]
            if negative_policy == "matched":
                neg_seqs = match_negatives(pos_seqs, neg_candidates,
                                           caliper=match_caliper)
            else:
                rng.shuffle(neg_candidates)
                neg_seqs = neg_candidates[:len(pos_seqs)]

        # The pairing above is 1:1 only while candidates last. A pathogen assayed
        # against most of the library runs out, and under "matched" a positive with
        # no candidate inside the caliper goes unpaired, so the classes arrive
        # skewed unless the surplus positives are dropped.
        if balance_tasks and len(neg_seqs) < len(pos_seqs):
            keep = rng.choice(len(pos_seqs), size=len(neg_seqs), replace=False)
            pos_seqs = [pos_seqs[i] for i in sorted(keep)]

        n_pos, n_neg = len(pos_seqs), len(neg_seqs)
        task_metadata[task]["n_positive"] = n_pos
        task_metadata[task]["n_negative"] = n_neg
        task_metadata[task]["label_prior"] = n_pos / max(n_pos + n_neg, 1)
        task_metadata[task]["negatives_exhausted"] = n_neg < n_pos

        # Determine train/val/test split
        split = "test" if task in test_tasks else "train"

        # Add positives (label 1 selects the active answer at options[1]).
        for s in pos_seqs:
            # If training task, carve out 15% validation
            row_split = split
            if split == "train" and rng.random() < 0.15:
                row_split = "val"
            examples.append(Example(sequence=s, task=task, question=question, options=options, label=1, split=row_split))

        # Add presumed negatives (label 0 selects options[0]).
        for s in neg_seqs:
            row_split = split
            if split == "train" and rng.random() < 0.15:
                row_split = "val"
            examples.append(Example(sequence=s, task=task, question=question, options=options, label=0, split=row_split))

    return examples, task_metadata
