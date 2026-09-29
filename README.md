# Aster

**An open research project for typed biological decisions using protein, cell, and language representations.**

Give Aster a biological context and candidate answers; its proposed role is to return structured predictions that researchers can evaluate against experiments.

> 📄 **Project site: [sathvikask0.github.io/aster](https://sathvikask0.github.io/aster/)** — every experiment, indexed. Start with [the plain-language summary](https://sathvikask0.github.io/aster/state.html).

## PeptiVerse: hold out entire questions

The eight-assay benchmark covers solubility, penetrance, PAMPA, Caco-2,
half-life, hemolysis, toxicity, and peptide–target binding affinity. The transfer
protocol holds out **questions and peptide identities simultaneously**:

| Split | Questions | Examples after overlap removal |
| --- | --- | ---: |
| Train | Solubility, penetrance, PAMPA, binding affinity | 28,761 |
| Validation | Hemolysis, Caco-2 | 5,492 |
| Test | Toxicity, half-life | 11,198 |

These counts describe the default pinned dataset. All rows for a question belong
to one split. If a peptide identity occurs in several questions, test takes
priority over validation, and validation over training; lower-priority labels
are excluded. The default split excludes 1,752 labels, without consulting their
values. No missing labels are inferred. Exact structures and source sequence
aliases are grouped globally; this is **not a sequence-family holdout**.

```bash
uv sync --python 3.11 --extra test --extra benchmark --locked
# Only needed if the original eight-assay dataset has not been built:
uv run --extra benchmark python scripts/build_peptiverse_benchmark.py

uv run --extra benchmark python scripts/run_peptiverse_transfer.py prepare
uv run --extra benchmark python scripts/run_peptiverse_transfer.py train
# Run once, after completing all model selection:
uv run --extra benchmark python scripts/run_peptiverse_transfer.py evaluate
```

`prepare` writes `data/peptiverse_transfer/{train,validation,test}.jsonl` plus a
checksum-verified manifest. Change the question assignment using `--train-tasks`,
`--validation-tasks`, and `--test-tasks`; every assay must appear exactly once.
Choose fresh `--data` and `--out` directories for another experiment. The
original supervised benchmark and its existing models are preserved.

The pilot trains shared classification and regression heads from scratch on
fixed chiral Morgan fingerprints, molecular descriptors, and target protein
composition for affinity. Frozen MiniLM encodes question text and answer
semantics. **This pilot does not fine-tune ESM or PeptideCLM.** A separate
per-property head cannot answer a held-out question, so the ordinary
`run_peptiverse_benchmark.py train` command rejects this transfer dataset.

Four predeclared controls run on seeds 42, 43, and 44: `question` reads peptide
and question, `entity_only` ignores question text, `task_id` uses a zero vector
for unseen task IDs, and `question_only` ignores the peptide. Tasks receive
equal weight in gradient steps. Checkpoint selection uses the equal-weight
mean of validation classification log loss and regression MAE (log10 hours for
half-life); that selection objective is a fixed convention, not an overall
biological performance score. Regression normalization uses only training
labels. Test task text is first encoded at evaluation, and test labels never
select a checkpoint. The explicit evaluator checks checkpoint hashes and
compares the question models with deliberately wrong, same-output-type prompts.

Results and model provenance are saved under `checkpoints/peptiverse_transfer/`.
The protocol is saved before optimization; every seed and control is reported.
Holding out half-life also requires extrapolating to a new measurement scale,
so weak transfer can reflect inadequate scale learning as well as biology.
Validation questions influence model selection; they are not final unseen tests.
Earlier supervised runs used these source tasks, so this is an exploratory
transfer experiment with fresh models, not a previously undisclosed dataset.

## Current status

**v0.1 is a working engineering prototype, not a biologically validated model.**

Implemented and tested:

- Joint ESM-2 and text-encoder fine-tuning, with configurable trainable suffix layers.
- A small learned cell-state encoder as an explicit baseline (not pretrained Arc SE/scGPT).
- Shared ESM weights for the intervention protein and readout protein.
- Dot-product choice scores and typed outputs, explicitly marked **uncalibrated**.
- A trainer that rejects train/validation/test overlap in intervention groups.
- An audit and descriptive baselines on an official Arc training-data sample.
- Real pretrained-encoder gradient checks on Apple MPS, using synthetic inputs/labels.

No trained biological Aster weights are released. The larger research configuration, pretrained cell backbone, calibration, and full-cohort benchmark remain unfinished.

## First capability

> Given a cell's starting state, will reducing one gene's activity increase or decrease another gene's RNA level?

A gene contains instructions that a cell can use to make RNA and often protein. Arc's perturbation experiments measure how RNA levels change after a gene is turned down. These measurements can supervise bounded questions about cellular responses; they do not establish effects on lifespan or clinical outcomes.

## Architecture

```text
Intervention protein → ESM-2 ──────────────┐
Readout protein      → shared ESM-2 ───────┼→ state projection → q
Control-cell profile → learned cell MLP ──┘

Question + each candidate answer → text encoder → projection → v_i

logit_i = dot(normalize(q), normalize(v_i)) / temperature
choice probabilities = softmax(logits)
```

The default compact configuration uses ESM-2 8M and BERT-tiny for integration tests. The last two ESM blocks and last text block are trainable. Larger encoders (ESM-2 650M, ModernBERT-large) have not been run here, so no configuration for them is checked in. The cell MLP, projections and temperature are trainable. Encoder outputs are recomputed during training; only tokenization is cached.

Independent branches can be batched or run concurrently at inference. No latency claim has been measured. Softmax is a distribution over supplied choices, not automatically a calibrated biological probability. A future abstention rule must be validated under distribution shift; entropy alone is not evidence that a model knows when it is wrong.

## Run the checks

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11:

```bash
uv sync --python 3.11 --extra test --locked
uv run pytest -q
uv run python scripts/audit_arc.py
uv run python scripts/smoke_pretrained.py
```

The audit downloads a roughly 5 MB checksum-verified sample directly from a pinned Arc repository commit. The pretrained smoke test downloads the compact encoder weights, executes three optimization steps on MPS when available, and verifies updates in both encoders and unchanged frozen weights. Its inputs and labels are artificial: loss values are not biological performance.

Every report lives on the project site — there are no markdown reports to keep in sync:

| Page | What it says |
|---|---|
| [Overview](https://sathvikask0.github.io/aster/) | The index: what the premise is, where it stands, and links to everything below. |
| [The negative result](https://sathvikask0.github.io/aster/negative-result.html) | v0.1: the text encoder scored 0.715 against a three-row lookup table's 0.718. |
| [**State of the project**](https://sathvikask0.github.io/aster/state.html) | **Start here.** The architecture, what works, what failed and why, and the traps that have already bitten once. Opens with a plain-language summary. |
| [Control rig](https://sathvikask0.github.io/aster/rig.html) | The evaluation, tested on synthetic data whose ground truth we control, before it was pointed at biology. Includes the correction to the `entity_only` control. |
| [v0.2 benchmark](https://sathvikask0.github.io/aster/resolution.html) | Task scaling breaks the lookup equivalence; the composition shortcut survives. |
| [In progress](https://sathvikask0.github.io/aster/in-progress.html) | Composition-matched negatives, so the shortcut cannot exist. Pre-registered, not yet run. |
| [Milestone 01](https://sathvikask0.github.io/aster/milestone.html) | First executable prototype and the Arc data audit. |

Machine-readable results stay in [`reports/`](reports/) as JSON. The site is built
from `docs/`; edit the pages there.

### Benchmark reporting rule

Every claim in these reports is a margin over the **shortcut ceiling** —
`max(majority, out-of-fold composition probe)` — and never over `task_id`.
`task_id` is the lookup floor: beating it shows the text tower is not a lookup
table, which is a statement about the architecture, not about biology. The
ceiling is computed in one place, [`aster/real/ceilings.py`](aster/real/ceilings.py),
scored out of fold so it cannot be inflated by fitting and scoring on the same
rows. A lift smaller than the combined 95% intervals is not a result.

```bash
uv run python scripts/run_amp_multitask.py --esm 8M --max-peptides 4000
```

Negatives in the AMP benchmark are **presumed, not measured**: the source tables
record reported activity and never recorded inactivity, so a negative means "no
record". `--negative-policy` selects how that is handled — `random` (reproduces
the published v0.2 run), `covered` (only well-assayed peptides), `matched`
(composition- and length-matched to the positives, which drives the shortcut
ceiling to chance), or `scrambled` (a positive control; see below).

### `matched` is unlearnable, and the reason is measurable

Running the `matched` benchmark produced a flat null at every model size. The
cause is in the construction, not in the model. Measured on the shipped data
(`negative_policy="matched"`, `balance_tasks=True`, seed 42):

| property | value |
| --- | --- |
| sequences carrying **both** label 1 and label 0 | 8,338 of 11,545 (**72%**) |
| negatives with confirmed activity against ≥3 other pathogens | **100%** |
| mean confirmed pathogen activities, negatives | 5.3 |
| mean confirmed pathogen activities, positives | 7.3 |

`covered` and `matched` require `min_coverage` records, so every presumed
negative is a *confirmed broad-spectrum AMP*; composition matching then removes
the length and residue-frequency signal. The encoder is handed the same sequence
with opposite labels and the only differing input is the prompt — which the
mechanism swap shows is a task identifier. The ~0.51 ceiling is therefore the
true ceiling, and no amount of capacity moves it. ESM-2 650M with the last two
blocks unfrozen reaches 0.514 against a 0.515 ceiling, with validation loss
pinned at ln 2 and `encoder_drift` 7.8e-05.

### `scrambled`: a positive control with a provable 0.5 ceiling

`--negative-policy scrambled` replaces presumed negatives with order-shuffled
copies of the positives: identical composition, identical length, destroyed motif
and helicity. A shuffle that reproduces its source, collides with another decoy,
or matches any real peptide in the library is rejected, so a decoy is never
silently a true positive. Both shortcut ceilings are then 0.5 by construction and
any lift is residue-*order* signal — exactly what ESM-2 provides and a
composition vector cannot.

It exists to separate "the code cannot learn" from "the task cannot be learned",
which the `matched` null alone cannot distinguish. ESM-2 8M, 2,000 peptides,
2 trainable blocks:

| model | accuracy |
| --- | --- |
| `entity_only_live` | 0.860 |
| `dual_live` | 0.858 |
| `cross_attention_live` | 0.854 |
| `entity_only_frozen` | 0.814 |
| `dual_frozen` | 0.811 |
| `cross_attention_frozen` | 0.684 |
| `task_id_frozen` | 0.487 |
| `question_only_frozen` | 0.513 |

Ceilings 0.526–0.545, with `composition_oof` at 0.43–0.52 — at or below chance,
as the design requires. Unfreezing reproduces across seeds at 35M:

| live − frozen | s42 | s43 | s44 | mean ± sd |
| --- | --- | --- | --- | --- |
| `dual` | +0.055 | +0.053 | +0.050 | **+0.052 ± 0.003** |
| `entity_only` | +0.023 | +0.059 | +0.052 | +0.044 ± 0.019 |
| `cross_attention` | +0.074 | +0.036 | +0.037 | +0.049 ± 0.022 |

`task_id_frozen` 0.486 ± 0.007 and `question_only_frozen` 0.514 ± 0.007 stay at
chance in every seed; `encoder_drift` is stable at 4.2–4.4e-04. The fine-tuning
path works and unfreezing two blocks is worth about five points.

**What this control does not show.** The correct answer does not depend on the
pathogen, so `entity_only` ≈ `dual` here is the expected outcome and says nothing
about the mechanistic-prompt hypothesis. Quote the within-seed live−frozen gap
rather than raw accuracies: seed 44 draws a different held-out mix and its
ceiling rises to 0.621. Testing the prompt hypothesis still needs *measured*
negatives — an MIC table with recorded inactives — which presence-only sources
cannot provide.

```bash
uv run python scripts/run_amp_finetune.py --esm 35M --trainable-blocks 2 \
    --negative-policy scrambled --max-peptides 2000 --epochs 6 --seed 42
```

## Fine-tuning the protein encoder

Every AMP number so far was produced over cached ESM-2 vectors, so the protein
encoder never learned anything about peptides. `run_amp_finetune.py` puts it back
in the graph with its last blocks trainable:

```bash
uv run python scripts/run_amp_finetune.py --esm 8M --trainable-blocks 2 \
    --negative-policy matched --epochs 6
```

Three things make the output readable as evidence rather than as a number.

`entity_only` gets a live encoder too. It never sees the question, and it is the
control that detects "answers without reading the question". Unfreezing only the
hypothesis model would hand it capacity the control was denied and call the
difference transfer. If `entity_only_live` moves as much, the fine-tune bought a
better peptide classifier and said nothing about typed decisions.

`*_live` is reported next to `*_frozen` — same head, same data, same reading
rule, one factor different. That pair is the only thing in the report that
isolates unfreezing. `task_id` and `question_only` stay frozen and are reference
rows, not matched controls. Pass `--skip-frozen-reference` to drop the pair, at
the cost of an unattributable number.

`encoder_drift` is printed beside each accuracy: the mean absolute change in the
trainable encoder weights. A gain with drift near zero is a head effect wearing a
fine-tuning label. A large drift usually means the encoder was destroyed and the
head compensated.

### Local ESM-2 650M pilot

```bash
bash scripts/run_650m_pilot.sh
```

This runs a 2,000-peptide pilot on Apple MPS with the last two protein blocks
trainable, MiniLM frozen, six maximum epochs, and batch size 16. It includes
the live peptide-only control and all frozen references. Reports and epoch logs
are saved under `reports/` and `logs/`; best trainable weights and the base model
revision are saved under `checkpoints/`. These small checkpoints require the
original pretrained base weights when restoring a model.

The optional `--cache-frozen-prefix` stores full-precision hidden states from the
frozen first 31 blocks on CPU, once per unique peptide. The final two blocks and
final norm are recomputed with gradients on every step. Tests compare cached and
uncached outputs and suffix gradients. This cache grows with sequence count and
length, so it is intended for the bounded local pilot first.

Training protocol version 2 gives live and frozen heads the same maximum epochs,
batch size and head learning-rate schedule, with validation early stopping for
each. Older fine-tuning reports used different frozen-reference training budgets
and must not be pooled with these runs. Only changing parameters and model
buffers are copied for best-checkpoint selection, avoiding full encoder copies.

Repeat a promising pilot with `SEEDS="43 44" bash scripts/run_650m_pilot.sh`.
GPU scaling should follow repeatable improvement over the matched frozen model
and shortcut controls, with evidence that the question matters. This benchmark
uses presumed negatives and holds out tasks, while peptides can recur across
splits; it does not establish generalization to unseen peptide families.

### Unfreezing the text tower too

`--unfreeze-text` puts the text encoder in the graph as well:

```bash
uv run python scripts/run_amp_finetune.py --esm 8M --trainable-blocks 2 \
    --unfreeze-text --text-trainable-blocks 1 --negative-policy matched
```

It is a separate flag on purpose. Run it *after* a protein-only run and read the
two as a ladder — frozen, protein-live, both-live — because if both towers move
at once and the number goes up, nothing in the report says which one did it.
`compare_amp_runs.py` warns when you mix the two.

Turning it on adds `question_only_live` to the comparison, and that is the point
rather than a side effect: `question_only` is the control for "answers from the
prompt alone", so giving the hypothesis model trainable text blocks while denying
them to that control inflates the hypothesis by exactly the withheld capacity —
the same argument that puts a live protein encoder in `entity_only`. If
`question_only_live` gains as much, the text tower learned the label prior, not
the question. Drift is reported per tower, so a run where only one moved is
visible rather than inferred.

The cost is small: there is one question per task and two answer strings in the
whole benchmark, so a step forwards about 23 short sequences and gathers per row
instead of forwarding a sequence per row. With text live, the batch carries
question and answer *indices* rather than cached vectors, since a cached vector
cannot carry a gradient — the mechanism-swap ablation swaps the index instead,
and there is a test for that path.

## Does the prompt work as biology, or as a name?

Both AMP runners now score every question-reading model a second time with a
prompt from a *different* mechanism family — the fungal chitin/ergosterol prompt
for a Gram-negative target, the LPS prompt for the fungal one. Donors are drawn
from the training tasks, so the swapped prompt is text the model has seen and the
result cannot be dismissed as out-of-distribution.

If accuracy survives the swap, the mechanism language bought nothing: the prompt
is a task identifier that happens to be spelled in English. That reading is in
`aster/real/ablation.py` as `read_swap`, written before any numbers, so it cannot
be re-decided afterwards. A drop is a necessary condition for the mechanism
mattering, not a sufficient one — sensitivity to prompt text is not correct use
of mechanism.

The ablation is eval-only, so it costs one extra forward pass. `entity_only` and
`task_id` are excluded because neither reads the question.

## Comparing runs

One run is one seed and no idea how much of the number is the seed.

```bash
for s in 42 43 44; do
  uv run python scripts/run_amp_finetune.py --seed $s --out reports/ft_s$s.json
done
uv run python scripts/compare_amp_runs.py reports/ft_s4*.json
```

`compare_amp_runs.py` prints, per model, the spread across runs, whether the lift
over the ceiling is positive in every run, and whether it exceeds the seed spread
— a lift smaller than the spread is not a finding about the model. It also
refuses to average runs that are not comparable: a report with no
`label_semantics_version` predates the answer-label fix, and differing negative
policy or encoder means the runs measure different things. Those are listed as
incomparable rather than folded into a mean. It flags the case that matters most
on the fine-tuning path: `entity_only` gaining as much as the hypothesis model.

## Training on experimental data

```bash
uv run python scripts/train.py data/experimental_rows.jsonl --out checkpoints/experiment
```

Supply JSONL records with:

- `split`: `train`, `validation`, or `test`.
- `intervention_group`: a stable group identifier shared by all rows for the intervention; use homology clusters when required by the evaluation.
- `evidence_id`: traceable experimental source identifier.
- `intervention_sequence` and `readout_sequence`: amino-acid sequences.
- `cell_state`: a fixed-order numerical vector derived from **control/pre-intervention** cells only.
- `question`, `options`, and integer `label`: the experimentally defined categorical task.

Feature order, normalization, label definitions and gene/protein mapping must be fixed and documented before training. Do not populate `cell_state` from treated cells: that leaks the outcome. The trainer checks group separation but cannot establish correct scientific provenance or independently detect all homology/batch leakage. It rejects overlong sequences instead of silently truncating them. Domain selection/windowing needs an explicit protocol before using long proteins.

Validation loss is macro-averaged across intervention groups to select a checkpoint. The trainer leaves test rows untouched. Full held-out evaluation and calibration are still required before publishing biological performance.

## Data plan and limitations

Our official Arc sample contains 600 cells, 1,000 measured genes, five interventions, controls, and multiple experimental batches. It is sufficient for pipeline checks only. The descriptive baseline bins observed normalized expression changes; these bins are not significance or equivalence tests. Cells are not independent experimental replicates. See [data audit](reports/arc_sample_audit.json).

The [full Arc Virtual Cell Atlas](https://github.com/ArcInstitute/arc-virtual-cell-atlas) currently uses Google Cloud Marketplace / Requester Pays access. No cloud subscription or billed download has been initiated. Arc also hosts a roughly 30 GB [filtered Replogle dataset](https://huggingface.co/datasets/arcinstitute/State-Replogle-Filtered); its metadata, provenance and license need auditing before selecting it as an alternative. Neither full dataset has been downloaded or used to train Aster.

[Arc State Embedding](https://huggingface.co/arcinstitute/SE-600M) and [scGPT](https://github.com/bowang-lab/scGPT) remain candidate pretrained cell encoders. Arc State weights carry noncommercial restrictions and are not included.

## Evaluation commitments

- Hold out entire interventions and, when supported, protein families and cellular contexts.
- Compare with global and per-readout priors, linear models, and no-ESM/no-text ablations.
- Evaluate batch effects, label stability, class imbalance, calibration and abstention.
- Use experimental replicates or studies as uncertainty units when available.
- Report negative results and distinguish engineering tests from biological validation.

## Inspiration and license

Inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): focused, typed decisions with measurable uncertainty. Aster is independent, is not a reproduction of Jev, and does not claim to implement its undisclosed RLCD recipe. Our starting approach is supervised learning on experimental outcomes.

Original code and documentation are MIT licensed. Third-party data, weights and software retain their own licenses. Source datasets and pretrained weights are not redistributed in this repository.
