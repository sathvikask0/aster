# Breaking the Rank-3 Trap: what v0.2 fixed, and what it didn't

**September 2026 · Aster v0.2**

```
python scripts/run_amp_multitask.py --esm 8M --max-peptides 4000
python scripts/recompute_ceilings.py reports/amp_multitask_results.json
```

> **Correction (27 September 2026).** The first version of this report headlined
> "+0.054 over the lookup floor" and called it positive transfer. That is the
> wrong zero. The control rig's own reading rule says the shortcut ceiling is
> the real zero, and the ceiling here was additionally computed *in sample* —
> fitted on the same rows it scored. Both are fixed below: the ceiling is now an
> out-of-fold probe, and every claim is a margin over it. The verdict changes
> from "resolution" to "half a resolution."

---

## 1. Context: the flaw v0.2 set out to fix

The September negative result showed that on PeptiVerse's 3 usable tasks,

$$\text{dual (text encoder)} \approx \text{task\_id (3-row lookup table)}$$

Three questions in a 384-dimensional space are 2 degrees of freedom. With no
shared tokens across atomic questions like *"Is it soluble?"*, the text tower
had nothing compositional to learn, so it collapsed into an expensive one-hot
and transfer to an unseen question was structurally impossible.

---

## 2. What changed

1. **Task scaling (21 pathogen assays).** 64,602 peptide-pathogen associations
   across 56 microbial targets from 7 public databases (DRAMP, APD3, CAMP, …).
   16 tasks train; 5 are held out across three envelope classes
   (*K. pneumoniae*, *S. typhimurium* — Gram-negative; *S. epidermidis*,
   *B. cereus* — Gram-positive; *C. albicans* — fungal).

2. **Mechanistic biophysical prompts.** Atomic names replaced by mechanism
   descriptions that share constituent concepts across tasks, so an unseen
   target is a recombination of seen mechanism terms rather than a new symbol:
   LPS outer membrane / thin peptidoglycan (Gram-negative), thick porous
   peptidoglycan / cytoplasmic lysis (Gram-positive), chitin, beta-glucan,
   ergosterol (fungal).

3. **Multi-head cross-attention (`CrossAttentionAster`).** The pooled dot
   product $\mathbf{u}^\top\mathbf{v}$ is replaced by question-conditioned
   attention, $\text{Attn}(Q=\mathbf{q}_\text{text}, K=V=\mathbf{x}_\text{slots})$,
   so prompt terms can weight residue feature subspaces.

---

## 3. Results on 5 unseen pathogens (5,144 held-out pairs)

Two zeros matter, and they disagree.

**`task_id` — the lookup floor.** Structurally chance on an unseen task.
Beating it says the text tower is no longer a lookup table.

**`ceiling` — the shortcut ceiling.** `max(majority, composition probe)`, where
the probe is a logistic regression on 20-dim amino-acid frequencies scored
**out of fold**. Beating it says the model knows something a bag of amino acids
does not. This is the real zero.

| Model | Overall | vs `task_id` (floor) | Mean lift vs **ceiling** | Tasks above ceiling |
|---|---:|---:|---:|---:|
| `cross_attention` | **0.555** | **+0.054** | **−0.058** | 1 / 5 |
| `dual_dot` | 0.537 | +0.036 | −0.075 | 1 / 5 |
| `entity_only` | 0.533 | +0.032 | −0.072 | 0 / 5 |
| `task_id` | 0.501 | 0.000 | −0.098 | 0 / 5 |
| `question_only` | 0.499 | −0.002 | −0.103 | 0 / 5 |

Per held-out task, `cross_attention` against the ceiling:

| Held-out target | Class | n | Ceiling | `cross_attention` | Lift | Beyond noise? |
|---|---|---:|---:|---:|---:|:---:|
| *K. pneumoniae* | Gram-negative | 1,524 | 0.554 | 0.576 ± 0.025 | **+0.022** | no |
| *S. epidermidis* | Gram-positive | 1,464 | 0.602 | 0.590 ± 0.025 | −0.012 | no |
| *S. typhimurium* | Gram-negative | 985 | 0.549 | 0.528 ± 0.031 | −0.021 | no |
| *B. cereus* | Gram-positive | 586 | 0.575 | 0.531 ± 0.040 | −0.044 | no |
| *C. albicans* | Fungal | 585 | 0.721 | 0.485 ± 0.041 | **−0.236** | yes |

The in-sample ceilings this report previously used were inflated by 0.012 to
0.039 (e.g. *S. typhimurium* 0.578 → 0.549, *B. cereus* 0.614 → 0.575). Fixing
them moves individual lifts but not the verdict.

---

## 4. Findings

1. **The lookup equivalence is broken — that part holds.** `cross_attention`
   beats `task_id` by +0.054 overall and `dual_dot` by +0.018. Under v0.1 those
   gaps were zero. Task scaling plus shared mechanism vocabulary gave the text
   tower something to learn that a lookup row cannot represent.

2. **The composition shortcut is not broken.** Against the real zero, the mean
   lift is **−0.058**, and 4 of 5 held-out targets sit below a 20-dimensional
   bag of amino acids. The single positive lift (+0.022 on *K. pneumoniae*) is
   smaller than the combined 95% intervals — it is not a result.

3. **The fungal target is a blowout, and it is diagnostic.** On *C. albicans*
   the composition probe reaches 0.721 while every model sits at chance. The
   presumed-negatives drawn for that task differ from its positives so sharply
   in composition that the split is nearly solved by amino-acid frequency alone
   — which is a statement about how the negatives were constructed, not about
   antifungal biology (§5).

4. **`entity_only` explains most of the gain.** It reaches 0.533 without ever
   seeing the question, against `cross_attention`'s 0.555. The margin genuinely
   attributable to the prompt is ~0.022 overall, in the same range as the noise
   on any single task.

**Verdict: the architecture question is answered, the biology question is not.**
The text encoder now earns parameters relative to a lookup table, and against
the only baseline that matters this benchmark remains a negative result.

---

## 5. Known limitations of this benchmark

- **Every negative is presumed, not measured.** The source tables are
  presence-only: they record that a peptide was reported active against a
  pathogen, never that one was tested and found inactive. A negative here means
  "no record", which is partly a statement about what anyone assayed.
  `load_amp_benchmark(negative_policy="covered")` now restricts negatives to
  peptides with records against ≥3 other pathogens, which narrows the coverage
  confound without removing it. The results above predate that option and use
  `negative_policy="random"`.
- **Negatives are not composition-matched**, which is why the ceiling is high in
  the first place. Addressed directly by the experiment in
  [IN_PROGRESS_01.md](IN_PROGRESS_01.md).
- **One seed, one split.** No seed variance is reported. Per-task intervals are
  normal-approximation binomial and ignore peptide-level correlation, so they
  are, if anything, optimistic.
- **ESM-2 8M only.** Whether a larger protein backbone clears the ceiling is
  untested.
