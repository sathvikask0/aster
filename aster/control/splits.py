"""Split strategies.

Aster's central claim is about generalization, so the split is not a detail --
it *is* the experiment. A random split answers a question nobody asked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aster.control.synthetic import Example


@dataclass
class Split:
    name: str
    train: list[Example]
    val: list[Example]
    test: list[Example]
    note: str = ""

    def summary(self) -> str:
        return (f"{self.name}: train={len(self.train)} val={len(self.val)} "
                f"test={len(self.test)}")


def _val_carve(train: list[Example], frac: float, rng) -> tuple[list, list]:
    """Carve a calibration/early-stopping set out of train.

    Drawn from TRAIN, never from test -- calibrating temperature on the test
    distribution would launder the shift we are trying to measure.
    """
    idx = rng.permutation(len(train))
    k = int(len(train) * frac)
    val = [train[i] for i in idx[:k]]
    tr = [train[i] for i in idx[k:]]
    return tr, val


def random_split(examples, seed=0, test_frac=0.2, val_frac=0.1) -> Split:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))
    k = int(len(examples) * test_frac)
    test = [examples[i] for i in idx[:k]]
    train = [examples[i] for i in idx[k:]]
    train, val = _val_carve(train, val_frac, rng)
    return Split("random", train, val, test,
                 "iid. Flatters everything; diagnostic of nothing.")


def held_out_question(examples, seed=0, frac=0.25, val_frac=0.1) -> Split:
    """Hold out whole questions. Their component parts still appear in train.

    Under SEMANTIC this is compositional generalization and should be solvable.
    Under ARBITRARY it is unsolvable by construction.
    """
    rng = np.random.default_rng(seed)
    # Prefer holding out multi-part questions, whose parts are seen elsewhere.
    qids = sorted({e.question_id for e in examples})
    paired = [q for q in qids if q.startswith(("p:", "a:"))]
    pool = paired if len(paired) >= 4 else qids
    pool = list(rng.permutation(pool))
    n = max(1, int(len(pool) * frac))
    heldout = set(pool[:n])

    train = [e for e in examples if e.question_id not in heldout]
    test = [e for e in examples if e.question_id in heldout]
    train, val = _val_carve(train, val_frac, rng)
    return Split("held_out_question", train, val, test,
                 f"{len(heldout)} unseen questions; parts seen in training.")


def held_out_entity_family(examples, seed=0, n_families=2, val_frac=0.1) -> Split:
    """Hold out entity families -- covariate shift on the entity side."""
    rng = np.random.default_rng(seed)
    fams = sorted({e.entity_family for e in examples})
    heldout = set(rng.permutation(fams)[:n_families].tolist())
    train = [e for e in examples if e.entity_family not in heldout]
    test = [e for e in examples if e.entity_family in heldout]
    train, val = _val_carve(train, val_frac, rng)
    return Split("held_out_entity_family", train, val, test,
                 f"families {sorted(heldout)} unseen.")


def held_out_both(examples, seed=0, val_frac=0.1) -> Split:
    """The honest one: novel question applied to a novel entity family."""
    rng = np.random.default_rng(seed)
    qids = sorted({e.question_id for e in examples})
    paired = [q for q in qids if q.startswith(("p:", "a:"))] or qids
    hq = set(list(rng.permutation(paired))[: max(1, len(paired) // 4)])
    fams = sorted({e.entity_family for e in examples})
    hf = set(rng.permutation(fams)[:2].tolist())

    train = [e for e in examples
             if e.question_id not in hq and e.entity_family not in hf]
    test = [e for e in examples
            if e.question_id in hq and e.entity_family in hf]
    train, val = _val_carve(train, val_frac, rng)
    return Split("held_out_both", train, val, test,
                 "unseen question AND unseen entity family.")


ALL_SPLITS = {
    "random": random_split,
    "held_out_question": held_out_question,
    "held_out_entity_family": held_out_entity_family,
    "held_out_both": held_out_both,
}
