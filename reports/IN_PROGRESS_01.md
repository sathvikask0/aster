# In progress: make the shortcut impossible, then see what's left

**Aster v0.3 · design frozen 27 September 2026 · run pending**

Status: the data construction and the pre-check below have run. The model
comparison has not — it needs ESM-2 embeddings, and the machine this was
prepared on has no access to the weights. Nothing here is a result yet.

---

## The question this experiment exists to answer

v0.2 established two things that point in opposite directions. The text encoder
stopped being a lookup table (+0.054 over `task_id`). And against the shortcut
ceiling, the honest zero, the mean lift was **−0.058**, with 4 of 5 held-out
targets below a 20-dimensional bag of amino acids.

Both can be true because the benchmark lets a model score well for a reason
that has nothing to do with the question. Positives for a target are real
antimicrobial peptides; negatives are peptides with no record for that target,
drawn at random. Those two pools differ in bulk composition — cationic, more
hydrophobic — so "is this peptide cationic and amphipathic?" answers most of the
benchmark without reading the prompt. On *C. albicans* a composition probe
reaches **0.721** that way.

So the number that decides whether Aster's premise is real — does encoding the
question as language buy anything? — is measured against a ceiling high enough
to swallow the entire effect. **The fix is not a better model. It is a benchmark
where the shortcut does not exist.**

---

## Design

**1. Composition-matched negatives.** Each positive claims its nearest unused
negative candidate in 21-dimensional space (20 amino-acid frequencies plus
scaled length), greedily, hardest positives first, with a caliper of 0.35 —
a positive with no candidate inside the caliper is dropped rather than paired
with a distant one. Negatives are additionally drawn only from peptides with
records against ≥3 other pathogens (`negative_policy="matched"`), so absence of
a record is less likely to mean "never assayed".

The two classes then have near-identical composition, and a composition probe is
at chance **by construction** — not as an empirical hope.

**2. Mechanism-swap ablation.** At test time, give each held-out target the
*wrong* mechanism prompt: the fungal chitin/ergosterol prompt for a
Gram-negative target, the LPS prompt for the fungal one. If accuracy is
unchanged, the prompt is functioning as a task identifier that happens to be
spelled in English, and the mechanism language is decoration.

**3. Same five models, same held-out targets, same reading rule.**
`cross_attention`, `dual_dot`, `task_id`, `question_only`, `entity_only`; lift
reported against the ceiling with binomial intervals; `task_id` shown only as
the lookup floor, never as the zero.

---

## Pre-check: the matching does what it claims

Held-out composition ceilings, out of fold, under each negative policy:

| Held-out target | `random` (v0.2) | `covered` | `matched` |
|---|---:|---:|---:|
| *C. albicans* | 0.721 | 0.750 | **0.522** |
| *S. epidermidis* | 0.602 | 0.605 | **0.513** |
| *B. cereus* | 0.575 | 0.588 | **0.540** |
| *K. pneumoniae* | 0.554 | 0.557 | **0.531** |
| *S. typhimurium* | 0.549 | 0.553 | **0.517** |

The shortcut is gone. Note that `covered` alone *raises* the ceiling slightly —
restricting negatives to well-assayed peptides does not make the classes
compositionally similar, which is worth knowing before anyone reaches for it as
a fix on its own.

```
python scripts/recompute_ceilings.py reports/amp_multitask_results.json \
    --negative-policy matched
```

---

## Pre-registered outcomes

Written before the run, so the result cannot be re-read into a success.

| Outcome | Reading |
|---|---|
| `cross_attention` clears the matched ceiling by more than the combined intervals, on ≥3 of 5 targets | The premise survives its first honest test. The text tower carries transferable biology. |
| Lift is positive but inside the intervals | Undecided. Needs seed variance and more held-out targets, not a louder claim. |
| `cross_attention` ≈ `entity_only` on matched data | Whatever the model knows, it is not coming from the question. The v0.2 gain was composition. |
| Everything collapses to chance | The benchmark's signal *was* the shortcut. The dataset cannot test the premise and a different data source is required. |
| Mechanism-swap leaves accuracy unchanged | The prompts are task ids in English. Mechanistic wording bought nothing, whatever the accuracy says. |

The fourth row is a live possibility and is the reason this is worth running:
after matching, the remaining signal must come from residue *order*, which is
exactly what ESM-2 is supposed to provide and a composition vector cannot.

## What would still be wrong afterwards

Matching removes a confound; it does not create measured negatives. Every label
in this benchmark is still presence-or-absence in a literature table. A clean
answer needs assay data with recorded inactives — an MIC table with upper
bounds, not a list of hits.
