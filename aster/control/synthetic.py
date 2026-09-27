"""Synthetic control tasks with known-ground-truth transfer structure.

The point of this module is NOT to model biology. It is to manufacture two
regimes where we already know the right answer, so that the evaluation harness
can be checked before it is ever pointed at real data:

  SEMANTIC  -- the question text compositionally determines the decision rule.
               A model that genuinely reads the question SHOULD generalize to
               held-out questions built from familiar parts.

  ARBITRARY -- the question text is an opaque identifier, and the decision rule
               is drawn independently of it. NOTHING can generalize to a
               held-out question. Chance is the ceiling.

If the harness reports transfer under ARBITRARY, the harness is wrong.
That is the whole reason this file exists.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  # 20 symbols, amino-acid shaped
N_PROPERTIES = 8
DIRECTIONS = ("high", "low")


@dataclass(frozen=True)
class Example:
    entity: str              # the sequence
    entity_family: int       # which generative family it came from
    question_id: str         # stable identity of the decision rule
    question_tokens: tuple[str, ...]
    options: tuple[str, ...]
    label: int               # index into options
    margin: float            # |w . z|, how decisive the ground truth is


@dataclass
class Question:
    qid: str
    tokens: tuple[str, ...]
    probe: np.ndarray        # w, the direction in latent property space
    n_parts: int


@dataclass
class SyntheticConfig:
    regime: str = "semantic"          # "semantic" | "arbitrary"
    n_entities: int = 4000
    n_families: int = 6
    seq_len_range: tuple[int, int] = (24, 48)
    n_single_questions: int = 16      # one (direction, property) part
    n_paired_questions: int = 40      # two parts -- the compositional set
    examples_per_question: int = 260
    min_margin: float = 0.15          # drop near-boundary cases as label noise
    seed: int = 0
    options: tuple[str, ...] = ("yes", "no")
    # Per-question label skew. 0.5 == balanced. Raising this lets us verify the
    # harness notices a question-only model exploiting label priors.
    label_prior_skew: float = 0.5
    _rng: np.random.Generator = field(default=None, repr=False)


class SyntheticWorld:
    """Generates entities, decision rules, and labelled examples."""

    def __init__(self, cfg: SyntheticConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Each property is a fixed linear readout over residue composition.
        # This is deterministic and learnable from the sequence alone.
        self.property_weights = self.rng.normal(
            size=(N_PROPERTIES, len(ALPHABET))
        )

        # Families differ in composition prior -> different regions of z-space.
        # Held-out families therefore constitute genuine covariate shift.
        self.family_priors = self.rng.dirichlet(
            np.ones(len(ALPHABET)) * 0.7, size=cfg.n_families
        )

        self.entities = self._make_entities()
        self.z = self._latent(self.entities)
        # Standardize so that probes are comparable across properties.
        self.z = (self.z - self.z.mean(0)) / (self.z.std(0) + 1e-8)

        self.questions = self._make_questions()

    # ---------------------------------------------------------------- entities

    def _make_entities(self) -> list[tuple[str, int]]:
        cfg = self.cfg
        out = []
        lo, hi = cfg.seq_len_range
        for i in range(cfg.n_entities):
            fam = i % cfg.n_families
            n = int(self.rng.integers(lo, hi + 1))
            idx = self.rng.choice(
                len(ALPHABET), size=n, p=self.family_priors[fam]
            )
            out.append(("".join(ALPHABET[j] for j in idx), fam))
        return out

    def _latent(self, entities: list[tuple[str, int]]) -> np.ndarray:
        counts = np.zeros((len(entities), len(ALPHABET)))
        index = {c: i for i, c in enumerate(ALPHABET)}
        for r, (seq, _) in enumerate(entities):
            for ch in seq:
                counts[r, index[ch]] += 1
            counts[r] /= max(len(seq), 1)
        return counts @ self.property_weights.T

    # --------------------------------------------------------------- questions

    def _make_questions(self) -> list[Question]:
        cfg = self.cfg
        rng = self.rng

        parts = [
            (d, p) for d in DIRECTIONS for p in range(N_PROPERTIES)
        ]
        rng.shuffle(parts)
        singles = parts[: cfg.n_single_questions]

        # Paired questions are combinations of parts that appear individually in
        # the single set. Held-out pairs are therefore novel COMBINATIONS of
        # familiar components -- the thing compositional transfer should buy.
        combos = [c for c in itertools.combinations(singles, 2)
                  if c[0][1] != c[1][1]]
        rng.shuffle(combos)
        pairs = combos[: cfg.n_paired_questions]

        questions: list[Question] = []

        def part_token(d: str, p: int) -> str:
            # Pre-bound token: binding a direction to a property is not the
            # thing under test, composition of bound parts is.
            return f"{d}_prop{p}"

        def probe_of(parts_: tuple) -> np.ndarray:
            w = np.zeros(N_PROPERTIES)
            for d, p in parts_:
                w[p] += 1.0 if d == "high" else -1.0
            return w

        for d, p in singles:
            qid = f"s:{d}:{p}"
            questions.append(Question(qid, ("q", part_token(d, p)), probe_of(((d, p),)), 1))

        for c in pairs:
            qid = "p:" + "+".join(f"{d}:{p}" for d, p in c)
            toks = ("q",) + tuple(part_token(d, p) for d, p in c)
            questions.append(Question(qid, toks, probe_of(c), 2))

        if cfg.regime == "arbitrary":
            # Same decision rules in spirit, but the text no longer describes
            # them: each question gets an opaque id and an INDEPENDENT probe.
            opaque = []
            for i, q in enumerate(questions):
                w = rng.normal(size=N_PROPERTIES)
                w /= np.linalg.norm(w) + 1e-8
                w *= np.linalg.norm(q.probe)  # match difficulty
                opaque.append(
                    Question(f"a:{i}", ("q", f"task{i}"), w, q.n_parts)
                )
            questions = opaque
        elif cfg.regime != "semantic":
            raise ValueError(f"unknown regime {cfg.regime!r}")

        return questions

    # ---------------------------------------------------------------- examples

    def examples(self) -> list[Example]:
        cfg = self.cfg
        rng = self.rng
        out: list[Example] = []

        for q in self.questions:
            scores = self.z @ q.probe
            # Optional per-question label skew, to exercise prior-exploitation
            # detection. Positive shift makes "yes" more common.
            if cfg.label_prior_skew != 0.5:
                shift = np.quantile(scores, 1.0 - cfg.label_prior_skew)
                scores = scores - shift

            order = rng.permutation(len(self.entities))
            taken = 0
            for i in order:
                if taken >= cfg.examples_per_question:
                    break
                margin = abs(float(scores[i]))
                if margin < cfg.min_margin:
                    continue  # ambiguous; excluding keeps labels clean
                seq, fam = self.entities[i]
                label = 0 if scores[i] > 0 else 1  # options = ("yes", "no")
                out.append(
                    Example(
                        entity=seq,
                        entity_family=fam,
                        question_id=q.qid,
                        question_tokens=q.tokens,
                        options=cfg.options,
                        label=label,
                        margin=margin,
                    )
                )
                taken += 1

        rng.shuffle(out)
        return out


def build(regime: str, **kw) -> tuple[list[Example], SyntheticWorld]:
    cfg = SyntheticConfig(regime=regime, **kw)
    world = SyntheticWorld(cfg)
    return world.examples(), world
