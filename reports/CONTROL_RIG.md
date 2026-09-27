# Control rig: testing the evaluation before trusting it

`python scripts/run_control.py` · `pytest tests/test_control.py` · ~15s CPU

## Why this exists

Aster's claim is that encoding the question with a text encoder buys
generalization to questions the model never trained on. The failure mode is
that the text encoder quietly degenerates into task identity — an expensive
one-hot — which produces transfer-shaped numbers with no transfer behind them.

On a random split that failure is invisible. Accuracy looks great either way.

So before pointing the harness at real data, it is pointed at synthetic data
whose ground truth we control, in two regimes:

| regime | question text | can anything generalize? |
|---|---|---|
| `semantic` | compositionally determines the decision rule | **yes** — unseen questions are new combinations of seen parts |
| `arbitrary` | opaque id; rule drawn independently | **no** — chance is the ceiling, by construction |

A harness worth trusting reports transfer in the first and none in the second.
**If it reports transfer under `arbitrary`, the harness is broken** and nothing
it later says about peptides means anything.

## What runs

Five models on four splits in both regimes.

**Models** — one hypothesis, four ways of explaining it away:

- `dual` — the hypothesis. entity → u, (question, answer) → v, score = ⟨u,v⟩.
- `task_id` — **the control.** Text encoder replaced by a lookup table. Unseen
  question → untrained `<unk>` row → chance. This is the transfer floor. If it
  ties `dual` on seen questions, the text encoder is not earning its parameters.
- `frozen_entity` — entity encoder not fine-tuned. How much comes from the
  entity side.
- `question_only` — never sees the entity. Whatever it scores is label prior.
- `entity_only` — never sees the question. The mirror shortcut.

**Splits** — the split *is* the experiment:

- `random` — iid. Flatters everything, diagnoses nothing. Included to show that.
- `held_out_entity_family` — covariate shift on the entity side.
- `held_out_question` — unseen questions, parts seen in training.
- `held_out_both` — the honest one.

Vocabularies are built from **train only** — letting test tokens in would hand
unseen questions a trained embedding and manufacture transfer. Temperature is
fit on a val set carved from **train**, never on test.

## Result

16/16 pre-registered checks pass. The headline cells, as lift over chance
(0 = guessing, 1 = perfect), calibrated:

| | `semantic` held_out_question | `arbitrary` held_out_question |
|---|---|---|
| `dual` | **+0.96** | **−0.09** |
| `task_id` | −0.01 | −0.01 |
| `question_only` | +0.07 | −0.01 |
| `entity_only` | −0.01 | −0.01 |

The rig transfers when transfer is possible and does not when it isn't. That is
the whole point: the harness is not fooling itself.

Note the `random` row, where `dual` (+0.99) and `task_id` (+0.98) are
indistinguishable. Any future result reported on a random split is consistent
with the text encoder doing nothing at all.

## What it caught

The rig's first run **failed**, and the failure was the useful part.

I had pre-registered "`entity_only` sits at chance everywhere." Under
`arbitrary` / `held_out_entity_family` it scored **+0.204**.

Not a bug. Families 4 and 5 sit far from the origin in latent space
(‖mean z‖ ≈ 2.6). For an entity far from the origin, `sign(w·z)` is dominated
by the family offset rather than by the probe `w`, so most questions share an
answer for those entities — per-entity label agreement across questions reaches
**0.63** for family 5. "Ignore the question, predict this entity's majority
label" therefore scores ~0.60 legitimately, matching the observed 0.602.

**This will happen with real peptides.** A family with extreme composition will
get the same answer to most property questions, and an entity-only baseline
will look smart on a held-out-family split.

The fix was not to loosen the threshold until it went green. The expectation was
wrong, so:

- the shortcut level is now **measured per split** (`shortcut_ceiling`) and
  reported on every run as an `INFO` line;
- every claim is checked as a **margin over that floor**, not as an absolute;
- a `WARN` fires when >15% of a split is answerable without using both inputs;
- `test_family_shift_creates_a_real_entity_only_shortcut` pins the finding so it
  cannot regress into a silent assumption.

## Reading a future result

1. `INFO shortcut ceiling` first. It is the real zero.
2. `task_id` on an unseen-question split. That is the transfer floor. `dual`
   beating chance is uninteresting; `dual` beating **`task_id`** is the claim.
3. Calibration next to accuracy, never after it. A model that is right 80% of
   the time and confident 99% of the time spends bench time on its own
   overconfidence. `overconfidence` (mean confidence − accuracy) is the column
   that matters for "should I run this experiment."
4. A number from `random` supports no generalization claim.

## Limits

Synthetic by design. The entity encoder is a small mean-pooled MLP, not ESM-2 —
this rig tests the evaluation contract, not representation quality. Passing here
is necessary, nowhere near sufficient. It says the *instrument* works; it says
nothing yet about biology.

## Files

```
aster/control/synthetic.py   two regimes with known ground truth
aster/control/splits.py      random / question / family / both
aster/control/models.py      dual + the four controls
aster/control/metrics.py     accuracy, NLL, Brier, ECE/MCE, temperature
aster/control/harness.py     training, evaluation, and verdict()
aster/control/tensors.py     whole-dataset tensorization
scripts/run_control.py       the experiment
tests/test_control.py        the same, as CI
```

Nothing here imports or modifies `aster/model.py`.
