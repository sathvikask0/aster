# State of Aster

**Last updated 27 September 2026.** Read this before designing the next
experiment. It is the standing summary: what the premise is, what the code
actually does, what has been established, what failed and why, and which traps
have already bitten once.

---

## 0. In plain language

**What you're testing.** One model that answers biology questions asked in
English, so a new question is a new sentence instead of a new training run.
Everything here is trying to find out whether the English does any work, or
whether it is decoration.

**The first way you get fooled.** The model can ignore the words and memorize
"question #3 → this pattern of answers." That looks exactly like understanding
until you test it on a question it has never seen. So there is a deliberately
stupid baseline, `task_id`, that replaces the text encoder with a lookup table
numbered 1, 2, 3. If the real model does not beat the lookup table, the words
are doing nothing.

**The second way you get fooled, and the one biting now.** The peptides labelled
*active* are real antimicrobials; the ones labelled *inactive* are peptides
nobody wrote down. Real antimicrobials tend to be positively charged and greasy,
so a dumb rule — "count the amino acids; is it charged and greasy?" — answers
most of the benchmark without reading the question. That dumb rule is the
**ceiling**. Beating the lookup table says the architecture is sound. Beating
the ceiling says something was learned about biology. Only the second one is a
scientific claim.

**Where things stand.** v0.1 failed cleanly: with 3 questions the text encoder
was mathematically indistinguishable from a lookup table, which is a counting
problem, not a tuning problem. v0.2 scaled to 21 pathogens with questions that
share vocabulary, and the model finally beat the lookup table — so the words now
matter. But against the dumb amino-acid rule it loses on 4 of 5 pathogens, and a
stripped-down model that never sees the question at all scores 0.533 where the
full model scores 0.555. Almost everything it knows, it knows without reading
the question.

**So: the instruments work, the biology is not there yet.** That is a real
position to be in, and the instruments are the harder half. The synthetic rig
correctly reports "no transfer" when transfer is impossible, which is the check
almost nobody runs.

**Why the next experiment exists.** You cannot tell whether the model learned
biology while the dumb rule is still available to it. So the dumb rule gets
removed: each *inactive* peptide is chosen to have nearly the same amino-acid
makeup as an *active* one. Counting amino acids then buys nothing — it falls
from 0.721 to 0.522, near coin-flipping. Whatever score survives has to come
from somewhere real. It may all collapse to chance, and that would say this
dataset was never able to answer the question.

**The thing to keep in mind.** Every "inactive" label in this project means
"nobody recorded it", not "somebody tested it and it did nothing". No amount of
modelling fixes that; only different data does.

---

## 1. The premise under test

> Does encoding a biological question **as language** buy generalization to
> questions the model was never trained on?

If yes, one model answers a family of typed biological questions and a new
question is a new sentence rather than a new training run. If no, Aster is a
multi-task classifier with an expensive tokenizer attached.

Everything below is in service of answering that one question honestly. The
project has not answered it yet. It has, so far, mostly built the instruments
that would let it be answered and ruled out ways of getting a fake yes.

---

## 2. Architecture

Three separate stacks live in this repo. They are easy to confuse.

### 2.1 `aster/model.py` — the v0.1 joint-encoder model

The real architecture, trained end to end. Not used by the v0.2 benchmark.

```
intervention protein → ESM-2 ─────────────┐
readout protein      → shared ESM-2 ──────┼→ concat → state MLP → q  (L2-normalized)
control-cell profile → learned cell MLP ──┘

question + candidate answer_i → text encoder → option MLP → v_i  (L2-normalized)

logit_i = ⟨q, v_i⟩ / temperature        choices = softmax(logits)
```

- **Trainable:** last *n* blocks of each encoder (default 2 protein / 1 text),
  the final encoder norm, the cell MLP, both projections, and `log_temperature`
  (clamped to [0.02, 2.0]). Everything else frozen; `train()` keeps both
  encoders in `eval()` mode so the frozen prefix stays deterministic.
- **Defaults:** ESM-2 8M + BERT-tiny, latent 128. `configs/research.json`
  specifies ESM-2 650M + ModernBERT-large, latent 256 — **never run**.
- **Choice-scoring by construction:** the number of options is free, options
  carry their own text, and `option_mask` supports per-row variable choices.
  This is the shape the "typed decision" premise needs.
- `decide()` returns `calibrated: False, status: experimental`. Honest, and
  still true.
- **No trained biological weights exist.** Only a 3-step MPS gradient smoke
  test on synthetic inputs.

### 2.2 `aster/real/` — the v0.2 embedding-space models

Encoders are run **once**, offline, and cached (`embed_sequences`,
`embed_texts`, `~/.cache/aster`). Models train on frozen vectors. This is what
every AMP and PeptiVerse number comes from.

- `RealAster` — `entity → u`, `(question, answer) → v`, score `⟨u,v⟩/√h`.
  Same shape as `aster/model.py`, so it is the faithful cheap proxy.
- `CrossAttentionAster` — question projects to a single query; the entity
  projects to `n_slots=8` slots serving as K and V; multi-head cross-attention
  (4 heads), residual + LayerNorm, then a 2-way classification head.

**Both carry the same five modes**, and the modes are the experiment:

| mode | what it is | reads entity | reads question |
|---|---|:---:|:---:|
| `dual` | the hypothesis | ✓ | ✓ |
| `task_id` | **lookup floor** — question as an embedding row; row 0 is zeroed and never trained, so an unseen task is structurally chance | ✓ | id only |
| `composition` | 20-dim AA frequencies instead of ESM-2 | ✓ | ✓ |
| `question_only` | label prior | ✗ | ✓ |
| `entity_only` | entity prior | ✓ | ✗ |

> **Fixed 27 September 2026 (architecture v2).** Version 1 of
> `CrossAttentionAster` never touched `a_emb`: it built one query from the
> question and ended in a fixed 2-way head, so the answer text was unused and
> the option count was frozen at two -- not a typed-decision model at all, and
> not something `aster/model.py` could grow into. It now builds one query per
> candidate answer and emits one score per option, so k options in gives k
> logits out. **Every published v0.2 number came from version 1 and is not
> comparable to anything this class produces now.** The re-run happens with
> v0.3; `config.arch_versions` in the results JSON records which is which.

### 2.3 `aster/control/` — the synthetic control rig

Tests the **evaluation**, not the biology. Two regimes with known ground truth
(`semantic`: question text compositionally determines the rule, transfer
possible; `arbitrary`: opaque ids, chance is the ceiling by construction), four
splits (`random`, `held_out_question`, `held_out_entity_family`,
`held_out_both`), the same five models, 16 pre-registered checks. Imports
nothing from `aster/model.py`.

---

## 3. Reading rules (these are load-bearing)

1. **The shortcut ceiling is the zero.** `max(majority, out-of-fold composition
   probe)`, computed in one place, `aster/real/ceilings.py`. Beating `task_id`
   is a statement about architecture; beating the ceiling is a statement about
   biology.
2. **A lift inside the combined 95% intervals is not a result.** Tracked as
   `lift_is_significant` in the results JSON.
3. **A number from a random split supports no generalization claim.** In the
   rig, `dual` (+0.99) and `task_id` (+0.98) are indistinguishable on `random`.
4. **Vocabularies and temperature come from train only.** Letting test tokens
   in hands unseen questions a trained embedding and manufactures transfer.
5. **Calibration next to accuracy, never after it.** A model that is right 80%
   of the time and confident 99% of the time is useless for choosing the next
   experiment.

---

## 4. What works

- **The evaluation instrument.** 16/16 pre-registered checks. `semantic`
  held-out-question: `dual` **+0.96**, `task_id` −0.01. `arbitrary`
  held-out-question: `dual` **−0.09**. It reports transfer when transfer is
  possible and none when it is impossible.
- **Leakage and split discipline.** The trainer rejects train/val/test overlap
  in intervention groups; splits hold out whole tasks and whole entity
  families; temperature is fit on a validation set carved from train.
- **Joint fine-tuning mechanics.** Both encoder suffixes receive finite nonzero
  gradients on MPS (34 ESM tensors, 16 text tensors updated); every frozen
  parameter verified unchanged; checkpoint replay, option permutation and
  option masking all covered by tests.
- **Shortcut measurement.** Ceiling is measured per split and reported on every
  run, with a WARN when >15% of a split is answerable without both inputs. That
  WARN now fires on two `arbitrary` splits where the broken control kept it
  silent.
- **Data audit hygiene.** Arc sample pinned by SHA-256 and upstream commit;
  descriptive baselines kept in the JSON and explicitly not promoted.
- **Task scaling breaks the lookup equivalence** (v0.2): `cross_attention`
  0.555 vs `task_id` 0.501. Under v0.1 that gap was zero. Real, and the only
  positive architectural finding so far.

---

## 5. What failed, and the mechanism in each case

| # | What | Result | Why |
|---|---|---|---|
| 1 | **Arc virtual cell sample** | Unusable | 99.94% of labels are "no change" (1 / 4,997 / 2). Five target genes = five questions, so nothing can be held out. Plumbing only. |
| 2 | **PeptiVerse in-distribution (v0.1)** | Decisive negative | `dual` 0.715 vs `task_id` **0.718**. The text tower is an expensive one-hot. 3 questions in a 384-d space is 2 degrees of freedom, and atomic questions ("Is it soluble?") share no tokens, so there is nothing compositional to learn. |
| 3 | **PeptiVerse leave-one-task-out** | Total failure | Accuracy 0.216–0.489, i.e. **below chance**, with ECE 0.22–0.32 and overconfidence positive everywhere. Below chance is the informative part: the model transfers the *other* tasks' label prior, which is inverted for the held-out task (hemolysis majority 0.784 → accuracy 0.216 ≈ 1 − 0.784). Confidently wrong is the opposite of the property that would make this useful. |
| 4 | **Conjunctive questions on PeptiVerse** | Not formable | The fix that made the synthetic regime work needs multi-labelled peptides. Overlap is 186 peptides (hemolysis ∧ penetrance) and **2** (hemolysis ∧ solubility). |
| 5 | **AMP multi-task v0.2 against the real zero** | Negative | Mean lift vs ceiling **−0.058**; 4 of 5 held-out targets below a 20-dim bag of amino acids; the one positive (+0.022, *K. pneumoniae*) inside the intervals. `entity_only` 0.533 vs `cross_attention` 0.555 — most of the gain never needed the question. |
| 6 | **The original v0.2 write-up** | Reporting error, self-inflicted | Headlined lift over `task_id` rather than over the ceiling, and the ceiling was fitted in sample (inflated 0.012–0.039). The rig was built to catch exactly this and the report bypassed it. Fixed 27 Sep. |
| 7 | **Pre-registered "entity_only sits at chance"** | Expectation was wrong | Under `arbitrary`/held-out-family it scored **+0.204**. Families far from the origin (‖mean z‖ ≈ 2.6) make `sign(w·z)` dominated by the family offset, so per-entity label agreement hits 0.63 and "predict this entity's majority" legitimately scores ~0.60. Fix was to measure the shortcut per split, not to loosen the threshold. |
| 8 | **The `entity_only` control itself** | Broken, silently | It set `v` to a constant, so every option scored identically and the model emitted one answer for every row. It could not use the entity at all: the 0.602 above is exactly what "always answer option 0" scores on that split. A control that cannot express its own shortcut measures nothing *while producing a plausible number*. Fixed 27 Sep in both the rig and `aster/real/models.py`; all 16 rig checks still pass, but measured ceilings moved, most sharply `arbitrary`/held_out_question, understated by 0.13. |

---

## 6. Traps that have already bitten (do not re-enter)

1. **Comparing to the floor instead of the ceiling.** #6 above.
2. **In-sample baselines.** A probe fitted and scored on the same rows reports
   its own capacity. Always out of fold.
3. **Random splits.** Flatter everything, diagnose nothing.
4. **Too few questions.** Below ~10 distinct tasks with shared vocabulary, the
   text tower *cannot* be distinguished from a lookup table, whatever it scores.
   This is structural, not a tuning problem.
5. **Family-shift entity shortcuts.** A compositionally extreme family answers
   most questions the same way; `entity_only` then looks smart on a held-out
   family split.
6. **Presumed negatives.** Every AMP negative means "no record", not "measured
   inactive". Absence tracks who assayed what. `negative_policy="covered"`
   narrows this; it does not fix it.
7. **Composition confounds.** Randomly drawn negatives differ in bulk
   composition from real AMPs, so a 20-dim vector answers the benchmark.
   `negative_policy="matched"` removes it by construction (C. albicans ceiling
   0.721 → 0.522).
8. **Below-chance accuracy is a prior-inversion signature**, not a bug — check
   the label prior before debugging the model.
9. **A control is code, and can be silently wrong.** It fails by producing a
   believable number, not by crashing. Before trusting a control, check what it
   is *structurally able to see* — and if its accuracy matches "always answer
   option 0", that is what it is doing. Every mode is now pinned by a test that
   asserts what it may and may not read.

---

## 7. Current gaps in the harness

- **The stored v0.2 model accuracies predate three fixes.** Out-of-fold ceilings
  were applied retroactively, but architecture v2, the repaired `entity_only`
  and calibration all require a re-run. Treat them as version-1 numbers.
- ~~The AMP benchmark reports no calibration~~ — fixed 27 Sep. It fits
  temperature on validation, reports ECE, Brier, NLL and overconfidence, and
  warns when a model is confident and wrong.
- **One seed, one split, fixed five held-out targets.** No seed variance, and
  binomial intervals ignore peptide-level correlation, so they are optimistic.
- **ESM-2 8M everywhere.** The 650M/ModernBERT research config has never run.
- **No pretrained cell encoder.** The cell tower is a learned MLP baseline; Arc
  SE and scGPT remain candidates pending a license and data-overlap audit.
- **`aster/model.py` has never been trained on real biological data.** The v0.2
  results say nothing directly about it, because they were produced by a
  different (and, per §2.2, non-equivalent) model over frozen embeddings.

---

## 8. Where the next experiment starts

[IN_PROGRESS_01.md](IN_PROGRESS_01.md) — composition-matched negatives so the
shortcut cannot exist, plus a mechanism-swap ablation (wrong mechanism prompt
for the target; if accuracy does not move, the prompts are task ids spelled in
English). Design frozen, pre-check run, pre-registered outcomes written,
model comparison pending ESM access.

Queued after that: data with measured inactives; seed and split variance;
ESM-2 650M; calibration and abstention under shift; a real perturbation cohort
for the original cell-state question; conjunctive questions across pathogen
targets, which the multi-pathogen data can now supply and PeptiVerse could not.

**The single highest-value change is not a model change.** It is a dataset with
recorded inactives. Until then, every accuracy here is measured against labels
that encode what people chose to publish.
