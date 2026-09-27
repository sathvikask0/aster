# Milestone 01: executable prototype and data audit

Aster's first release verifies the joint-training implementation. It does **not** establish predictive biological accuracy.

## Evidence

- Official Arc sample: 600 cells, 1000 genes, five interventions plus controls, 48 experimental batches.
- Unique cell and gene identifiers, finite nonnegative integer counts, no missing observational metadata: verified in `arc_sample_audit.json`.
- Pretrained encoder smoke test ran on `mps` for 3 updates, with synthetic sequences, features and labels.
- Updated parameter tensors: ESM 34; text encoder 16.
- Every frozen parameter stayed unchanged. Both selected encoder suffixes had finite nonzero gradients.
- Tests cover frozen/trainable boundaries, candidate permutation, masked options, checkpoint replay and intervention leakage rejection.

The downloaded source sample is identified by SHA-256 `eb36c766cbf76353f9981cb3a3aa32137622d1de53b29d861c483742bcd4dec7` and pinned upstream commit `5e64833518a6603a0301cbe28185d49c30f4a986`. The pretrained smoke report pins both model revisions.

## What the sample cannot tell us

Repeated cells do not provide five hundred independent intervention experiments. There are only five target genes. The sample contains only 1,000 genes, so normalization uses subset totals rather than full transcriptome library sizes. Its pooled-batch descriptive baselines are vulnerable to batch composition differences. The within-threshold label does not establish equivalence or absence of an effect. No generalization, calibration or longevity claim follows from these checks.

The baseline holds out one intervention at a time, using the other interventions' class frequencies. This verifies split and metric plumbing; it is not a competitive full-cohort benchmark. Baseline numbers are retained in `arc_sample_audit.json` for inspection, not promoted as a model result.

## Next scientific milestone

Acquire and audit a full experimental cohort; define batch-aware labels and control features; map target and readout genes to verified protein sequences; establish intervention/family-disjoint splits; then jointly train and compare the model against simple baselines and modality ablations. Choose a pretrained cell encoder only after checking its license and data overlap.

Full Arc VCC data access is unresolved. No cloud billing was enabled and no biological Aster checkpoint was trained. The compact cell MLP is intentionally a baseline, not a pretrained cell foundation model.

## Sources

- [Arc sample and tutorial](https://github.com/ArcInstitute/cell-eval2/tree/5e64833518a6603a0301cbe28185d49c30f4a986/docs): inspected and downloaded September 27, 2026.
- [Arc data access](https://github.com/ArcInstitute/arc-virtual-cell-atlas): official full-data access instructions inspected September 27, 2026.
- [ESM](https://github.com/facebookresearch/esm) and [BERT-tiny](https://huggingface.co/prajjwal1/bert-tiny): pretrained weights used only in the integration test.
