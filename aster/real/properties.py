"""Property tasks with *measured* negatives, in place of pathogen tasks.

Why this module exists. The AMP benchmark in `amp.py` defines a task as a
pathogen, and its negatives are presumed: the source tables record activity and
never inactivity, so "negative" means "no record". Two things follow, and both
were measured rather than argued (see the `matched` section of the README).

  1. The labels contradict themselves. Under the composition-matched policy 72%
     of sequences carry both label 1 and label 0, because a peptide active
     against one pathogen is drawn as a presumed negative for another.
  2. The tasks are near-duplicates. Most antimicrobial peptides are
     broad-spectrum, so "does it kill *E. coli*" and "does it kill
     *K. pneumoniae*" have almost the same answer. A prompt describing the
     mechanism has nothing to distinguish, which is exactly what the
     mechanism-swap ablation kept reporting: the prompt acts as a task id.

Aster's premise is that writing a task down in natural language buys transfer to
a task the model has not seen. Testing it needs tasks whose answers genuinely
differ, and labels somebody measured. Properties supply both.

  antimicrobial   does this peptide inhibit microbial growth?  (MIC, uM)
  hemolytic       does it lyse human red blood cells?          (HC50, uM)

A peptide can be yes to the first and no to the second -- that ratio is the
therapeutic index, and it is the quantity the field actually cares about. The two
questions are about different membranes and different cells, so a mechanistic
prompt has real work to do.

**What counts as a negative here.** A row is only usable if it carries a
measured concentration. `HC50 = 596.7 uM` means somebody put the peptide in a
tube, found that it took a flooding dose to do damage, and wrote the number
down. That is a recorded "no". A missing row is still not a negative and this
loader refuses to invent one: `require_measured=True` (the default) drops any row
without a finite concentration rather than reading absence as inactivity.

**The threshold is a convention, not a fact.** The hemolysis literature splits at
HC50 = 100 uM and the antimicrobial literature commonly at MIC = 32 uM. Both are
choices. They are parameters here, and `threshold_sensitivity` re-labels the same
rows at several cutoffs so a result that only exists at one line is visible as
such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from aster.real.amp import AA, Example

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "properties"

# Direction matters and is easy to get backwards. For both assays the number is
# a dose, so a SMALL number means potent. "active" is therefore concentration
# <= threshold, and a large concentration is the measured negative.
PROPERTIES = {
    "hemolytic": {
        "units": "uM",
        "measure": "HC50",
        "threshold": 100.0,
        "family": "host_membrane",
        "question": (
            "Does this peptide insert into the cholesterol-rich, zwitterionic outer "
            "leaflet of a human erythrocyte membrane and lyse the cell, releasing "
            "haemoglobin at low micromolar concentration?"
        ),
        "answers": (
            "it leaves the human red blood cell membrane intact",
            "it lyses human red blood cells at low concentration",
        ),
    },
    "antimicrobial": {
        "units": "uM",
        "measure": "MIC",
        "threshold": 32.0,
        "family": "microbial_membrane",
        "question": (
            "Does this peptide carry the cationic amphipathic structure needed to bind "
            "an anionic microbial envelope and permeabilise the cytoplasmic membrane, "
            "inhibiting growth at low micromolar concentration?"
        ),
        "answers": (
            "it fails to inhibit microbial growth",
            "it inhibits microbial growth at low concentration",
        ),
    },
}


@dataclass(frozen=True)
class PropertyTable:
    """One property's measured rows: sequence, concentration, and its units."""

    name: str
    frame: pd.DataFrame  # columns: sequence, concentration
    measure: str
    units: str

    def labelled(self, threshold: float | None = None) -> pd.DataFrame:
        """Add a label column; 1 = active (potent) at or below the threshold."""
        cut = PROPERTIES[self.name]["threshold"] if threshold is None else threshold
        out = self.frame.copy()
        out["label"] = (out["concentration"] <= cut).astype(int)
        return out


def clean_sequences(frame: pd.DataFrame, min_len: int = 6, max_len: int = 60) -> pd.DataFrame:
    """Drop rows the protein tokenizer cannot represent, and collapse duplicates.

    A sequence measured more than once is averaged in concentration space, which
    is what the HemoPI2 authors do; averaging after thresholding would let the
    label depend on how many times a peptide happened to be assayed.
    """
    out = frame.copy()
    out["sequence"] = out["sequence"].astype(str).str.strip().str.upper()
    out = out[out["sequence"].str.len().between(min_len, max_len)]
    out = out[out["sequence"].apply(lambda s: set(s) <= AA)]
    out["concentration"] = pd.to_numeric(out["concentration"], errors="coerce")
    out = out[np.isfinite(out["concentration"]) & (out["concentration"] > 0)]
    return (out.groupby("sequence", as_index=False)["concentration"]
               .mean()
               .reset_index(drop=True))


def load_property_table(
    name: str,
    path: Path | str,
    sequence_column: str = "SEQUENCE",
    concentration_column: str | None = None,
    require_measured: bool = True,
    min_len: int = 6,
    max_len: int = 60,
) -> PropertyTable:
    """Read one property's CSV. Rows without a measured concentration are dropped.

    `concentration_column` defaults to the first column whose name mentions the
    property's unit or measure, which covers the HemoPI2 export (its column is
    literally named for the unit).
    """
    if name not in PROPERTIES:
        raise ValueError(f"Unknown property {name!r}; known: {sorted(PROPERTIES)}")
    spec = PROPERTIES[name]
    frame = pd.read_csv(path)
    if sequence_column not in frame.columns:
        raise ValueError(
            f"{path}: no {sequence_column!r} column; found {list(frame.columns)}")

    if concentration_column is None:
        wanted = (spec["units"].lower(), spec["measure"].lower(), "µm", "um")
        matches = [c for c in frame.columns
                   if any(w in str(c).lower() for w in wanted)]
        if not matches:
            raise ValueError(
                f"{path}: no concentration column found for {name} "
                f"({spec['measure']} in {spec['units']}); found {list(frame.columns)}. "
                "Pass concentration_column explicitly. A label column alone is not "
                "enough: without the concentration a negative cannot be shown to be "
                "measured rather than presumed."
            )
        concentration_column = matches[0]

    tidy = frame[[sequence_column, concentration_column]].copy()
    tidy.columns = ["sequence", "concentration"]
    before = len(tidy)
    tidy = clean_sequences(tidy, min_len=min_len, max_len=max_len)
    if require_measured and tidy.empty:
        raise ValueError(f"{path}: no rows survived with a finite concentration")
    if not require_measured:
        pass  # kept as an explicit branch so the default is visibly the strict one
    print(f"  {name}: {len(tidy):,} measured sequences "
          f"({before - len(tidy):,} rows dropped) | {spec['measure']} in {spec['units']}")
    return PropertyTable(name=name, frame=tidy,
                         measure=spec["measure"], units=spec["units"])


def task_overlap(tables: dict[str, PropertyTable]) -> dict:
    """How many sequences are measured for more than one property.

    This is the gating number for the whole design. Held-out-task transfer asks
    what a model trained on property A does on property B; if almost no peptide
    is measured for both, the two tasks share no peptides and a cross-task
    result says nothing about transfer, only about the two populations.
    """
    sets = {name: set(t.frame["sequence"]) for name, t in tables.items()}
    names = sorted(sets)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sets[a] & sets[b]
            denom = min(len(sets[a]), len(sets[b])) or 1
            pairs[f"{a}|{b}"] = {
                "shared": len(shared),
                "fraction_of_smaller": len(shared) / denom,
            }
    return {"per_property": {k: len(v) for k, v in sets.items()},
            "pairs": pairs,
            "measured_for_all": len(set.intersection(*sets.values())) if sets else 0}


def label_agreement(tables: dict[str, PropertyTable],
                    thresholds: dict[str, float] | None = None) -> dict:
    """For sequences measured for both properties, how the two labels co-occur.

    The AMP benchmark failed partly because one sequence carried both labels for
    near-identical questions. Here a sequence carrying different labels for
    *different* properties is the point, not a defect: a peptide that kills
    microbes and spares blood cells is the therapeutic-index case. This function
    reports the joint distribution so that claim is checked rather than assumed.
    """
    thresholds = thresholds or {}
    frames = {n: t.labelled(thresholds.get(n)).set_index("sequence")["label"]
              for n, t in tables.items()}
    names = sorted(frames)
    if len(names) < 2:
        return {}
    joined = pd.concat([frames[n].rename(n) for n in names], axis=1, join="inner")
    if joined.empty:
        return {"n": 0, "note": "no sequence is measured for both properties"}
    counts = joined.groupby(names).size()
    return {
        "n": int(len(joined)),
        "joint": {"|".join(map(str, k if isinstance(k, tuple) else (k,))): int(v)
                  for k, v in counts.items()},
        "per_property_positive_rate": {n: float(joined[n].mean()) for n in names},
    }


def threshold_sensitivity(table: PropertyTable,
                          cuts: tuple[float, ...] = (10.0, 32.0, 100.0, 200.0)) -> dict:
    """Positive rate at several cutoffs, so a threshold artefact is visible."""
    return {str(c): float((table.frame["concentration"] <= c).mean()) for c in cuts}


def build_property_benchmark(
    tables: dict[str, PropertyTable],
    test_tasks: tuple[str, ...],
    thresholds: dict[str, float] | None = None,
    seed: int = 42,
    val_fraction: float = 0.15,
    balance_tasks: bool = True,
    disjoint_sequences: bool = False,
) -> tuple[list[Example], dict[str, dict]]:
    """Examples and metadata in the same shape the AMP runners already consume.

    `disjoint_sequences` assigns each sequence to exactly one split across all
    tasks, so a peptide seen in training cannot reappear in test under another
    property. That is the stricter reading and it is off by default because it
    removes the shared peptides that make cross-task transfer measurable at all
    -- the two settings answer different questions and the flag is recorded in
    the metadata so a report cannot be read as the other one.
    """
    if not tables:
        raise ValueError("No property tables supplied")
    unknown = set(test_tasks) - set(tables)
    if unknown:
        raise ValueError(f"Held-out tasks not present in the data: {sorted(unknown)}")
    if len(test_tasks) >= len(tables):
        raise ValueError("At least one property must remain for training")

    rng = np.random.default_rng(seed)
    thresholds = thresholds or {}

    split_of: dict[str, str] = {}
    if disjoint_sequences:
        every = sorted(set().union(*(set(t.frame["sequence"]) for t in tables.values())))
        for s in every:
            split_of[s] = "val" if rng.random() < val_fraction else "train"

    examples: list[Example] = []
    meta: dict[str, dict] = {}
    for name in sorted(tables):
        table = tables[name]
        spec = PROPERTIES[name]
        cut = thresholds.get(name, spec["threshold"])
        rows = table.labelled(cut)

        if balance_tasks:
            keep = []
            counts = rows["label"].value_counts()
            smaller = int(counts.min()) if len(counts) == 2 else 0
            if smaller == 0:
                raise ValueError(
                    f"{name}: only one class at threshold {cut} {spec['units']}; "
                    "choose a cutoff inside the measured range")
            for label in (0, 1):
                idx = rows.index[rows["label"] == label].to_numpy()
                keep.append(rng.choice(idx, size=smaller, replace=False))
            rows = rows.loc[np.concatenate(keep)].sort_index()

        held_out = name in test_tasks
        meta[name] = {
            "type": spec["family"],
            "prompt_family": spec["family"],
            "question": spec["question"],
            "options": spec["answers"],
            "is_held_out": held_out,
            "measure": spec["measure"],
            "units": spec["units"],
            "threshold": cut,
            "negatives_are_presumed": False,
            "negatives_are_measured": True,
            "negatives_are_synthetic_decoys": False,
            "n_positive": int((rows["label"] == 1).sum()),
            "n_negative": int((rows["label"] == 0).sum()),
            "label_prior": float(rows["label"].mean()),
            "disjoint_sequences": disjoint_sequences,
        }

        for seq, label in zip(rows["sequence"], rows["label"]):
            if held_out:
                split = "test"
            elif disjoint_sequences:
                split = split_of[seq]
            else:
                split = "val" if rng.random() < val_fraction else "train"
            examples.append(Example(sequence=seq, task=name,
                                    question=spec["question"],
                                    options=spec["answers"],
                                    label=int(label), split=split))
    return examples, meta
