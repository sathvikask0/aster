# Aster

**An open research project for biological decisions with protein, cell, and language models.**

Aster aims to combine what a protein is, the current state of a cell, and a focused biological question to predict experimentally measurable outcomes.

**Status: design stage. No Aster model has been trained or validated yet.**

## Proposed architecture

```text
Protein sequence       → ESM-2 protein encoder ──┐
Starting gene activity → cell-state encoder ────┼→ fusion + decision heads
Question / assay       → text encoder ──────────┘
                                                ↓
                                 choices, scores, probabilities
```

We plan to jointly fine-tune the last few layers of the encoders and train the fusion and decision heads. Earlier layers remain frozen. Protein candidates can share an ESM-2 backbone. Independent branches can be batched or run concurrently at inference; performance will be measured rather than assumed.

Dot-product scores with softmax can support candidate ranking. They are not automatically calibrated probabilities of biological events. Binary or categorical outcome heads require experimental labels and separate calibration and evaluation.

## First research question

> Given a cell's starting state, how does reducing the activity of one gene change the activity of other genes?

Arc Institute's released 2025 Virtual Cell Challenge dataset is a candidate starting point. The initial benchmark would predict measured changes relative to control cells. Labels, thresholds, preprocessing, data access, and splits remain to be finalized. Gene activity is an RNA measurement, not a direct measurement of protein activity or lifespan.

## Evaluation commitments

- Hold out complete interventions; use independent cellular contexts when the data supports this.
- Account for related proteins, experimental batches, and repeated cell measurements when defining splits.
- Compare against simple baselines and ablations without the protein or text branch.
- Measure calibration, generalization, and the trade-off between abstaining and making predictions.
- Publish reproducible configurations, provenance, limitations, and negative results.

Large cell counts do not imply equally many independent experiments. A model predicting gene-expression changes does not establish a treatment's safety, efficacy, or effect on longevity.

## Candidate components and data

- [ESM-2](https://github.com/facebookresearch/esm): protein sequence encoder.
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large): candidate question/assay text encoder.
- [Arc State Embedding](https://huggingface.co/arcinstitute/SE-600M) or [scGPT](https://github.com/bowang-lab/scGPT): candidate cell encoder.
- [Arc Virtual Cell Atlas](https://github.com/ArcInstitute/arc-virtual-cell-atlas): candidate experimental data.

The cell encoder has not been selected. Arc State model weights carry noncommercial restrictions; they are not included here. Each dependency, dataset, and checkpoint retains its own license. Data-access requirements and permitted redistribution will be checked before inclusion.

## Inspiration

Aster is inspired by the idea of typed, probabilistic decisions described by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). It is an independent biological research project, not a reproduction of Jev's undisclosed architecture or training algorithm. Our starting approach is supervised learning on experimental outcomes; no RLCD implementation is claimed.

## Roadmap

1. Audit a public dataset and establish leakage-resistant evaluation splits.
2. Implement and evaluate simple baselines.
3. Train the joint encoder prototype and measure the contribution of each branch.
4. Release verified training/inference code, evaluation results, and compatible model artifacts.

## Contributing

Contributions to experimental design, data curation, baseline models, calibration, and reproducibility are welcome. Please open an issue before undertaking a substantial change.

## License

Original repository code and documentation are available under the MIT License. This does not relicense third-party data, models, or software.
