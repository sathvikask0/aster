# Property datasets (measured negatives)

These two files are not in the repository: they carry their own citation terms
and are fetched by hand. Neither the cloud container nor the desktop sandbox can
reach these hosts under the current egress allowlist.

## hemolytic — HemoPI2

    https://webs.iiitd.edu.in/raghava/hemopi2/download/cross_val_dataset.csv   -> hemolytic_crossval.csv
    https://webs.iiitd.edu.in/raghava/hemopi2/download/independent_dataset.csv -> hemolytic_independent.csv

Columns: `SEQUENCE`, `µM` (HC50), `label`. The loader reads the concentration and
derives the label itself; it ignores the supplied `label` column on purpose, so
the cutoff is a recorded parameter rather than an inherited convention.

Roughly 1,926 experimentally validated peptides: 891 hemolytic (HC50 <= 100 uM)
and 1,035 non-hemolytic (HC50 > 100 uM). The non-hemolytic rows are **measured**,
which is the entire reason this source was chosen.

Cite: Prediction of hemolytic peptides and their hemolytic concentration,
*Communications Biology* (2025). https://doi.org/10.1038/s42003-025-07615-w

## antimicrobial — DBAASP v3

Export sequence and MIC (uM) per peptide, including weak/high-MIC rows, to
`antimicrobial_mic.csv` with a `SEQUENCE` column and a MIC column whose name
mentions `MIC` or `uM`. The REST API's `ranking_search` filters quantitative
metrics with `>`, `<`, `>=`, `<=`, `=`, which is how the high-MIC (measured
negative) rows are obtained.

    https://dbaasp.org/api

Cite: Pirtskhalava et al., DBAASP v3, *Nucleic Acids Research* 49(D1), 2021.
https://doi.org/10.1093/nar/gkaa991

## Checks to run once the files are here

    uv run python -c "
    from aster.real.properties import *
    t = {'hemolytic': load_property_table('hemolytic', 'data/properties/hemolytic_crossval.csv'),
         'antimicrobial': load_property_table('antimicrobial', 'data/properties/antimicrobial_mic.csv')}
    print(task_overlap(t)); print(label_agreement(t))
    print(threshold_sensitivity(t['hemolytic']))"

`task_overlap` is the gating number. Cross-task transfer is only measurable if a
useful number of peptides are measured for both properties; if the overlap is
near zero, a cross-task result describes two populations rather than transfer.
