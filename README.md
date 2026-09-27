# Aster

**An open research project for typed biological decisions using protein, cell, and language representations.**

Give Aster a biological context and candidate answers; its proposed role is to return structured predictions that researchers can evaluate against experiments.

> 📄 **Reports:** [The text encoder isn't doing anything (September 2026)](https://sathvikask0.github.io/aster/)

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

The default compact configuration uses ESM-2 8M and BERT-tiny for integration tests. The last two ESM blocks and last text block are trainable. `configs/research.json` specifies ESM-2 650M and ModernBERT-large, with their last three blocks trainable; that larger configuration has not yet been run here. The cell MLP, projections and temperature are trainable. Encoder outputs are recomputed during training; only tokenization is cached.

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

Reports are in [`reports/`](reports/):

| Report | What it says |
|---|---|
| [MILESTONE_01](reports/MILESTONE_01.md) | First working pipeline. |
| [CONTROL_RIG](reports/CONTROL_RIG.md) | The evaluation, tested on synthetic data whose ground truth we control, before it was pointed at biology. |
| [RESOLUTION_01](reports/RESOLUTION_01.md) | v0.2 multi-task benchmark. The text encoder stops being a lookup table; it still does not clear the composition shortcut. Corrected 27 Sep 2026 — see the note at the top. |
| [IN_PROGRESS_01](reports/IN_PROGRESS_01.md) | The experiment being set up now: composition-matched negatives, so the shortcut cannot exist. Pre-registered, not yet run. |

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
uv run python scripts/recompute_ceilings.py reports/amp_multitask_results.json
```

Negatives in the AMP benchmark are **presumed, not measured**: the source tables
record reported activity and never recorded inactivity, so a negative means "no
record". `--negative-policy` selects how that is handled — `random` (reproduces
the published v0.2 run), `covered` (only well-assayed peptides), or `matched`
(composition- and length-matched to the positives, which drives the shortcut
ceiling to chance).

## Training on experimental data

```bash
uv run python scripts/train.py data/experimental_rows.jsonl --out checkpoints/experiment
# Larger, not yet validated configuration:
uv run python scripts/train.py data/experimental_rows.jsonl --config configs/research.json
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
