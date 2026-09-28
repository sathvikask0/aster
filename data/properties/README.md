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

## antimicrobial — GRAMPA

Prepared by script, not by hand:

    uv run python scripts/fetch_property_data.py

GRAMPA is a published aggregation of MIC measurements from DBAASP, DRAMP,
YADAMP, APD and DADP, stored as log10(MIC in uM). The script converts to uM,
drops peptides carrying unusual chemical modifications (the encoder sees only
the amino-acid string, so their activity is partly produced by something it
cannot see), takes the minimum MIC across organisms as the peptide's best
result, and writes `antimicrobial_mic.csv` with `SEQUENCE` and `uM`.

6,169 peptides, of which 1,195 (19.4%) have a best MIC above 32 uM: tested and
found weak, which is a measured negative rather than a missing row.

    https://github.com/zswitten/Antimicrobial-Peptides

Cite: Witten & Witten, Deep learning regression model for antimicrobial peptide
design, *bioRxiv* (2019). https://doi.org/10.1101/692681

The DBAASP REST API remains the route to per-organism MIC if the per-pathogen
framing is ever revisited; `ranking_search` filters quantitative metrics with
`>`, `<`, `>=`, `<=`, `=`. https://dbaasp.org/api
Cite: Pirtskhalava et al., DBAASP v3, *Nucleic Acids Research* 49(D1), 2021.
https://doi.org/10.1093/nar/gkaa991

## Checks to run once the files are here

    uv run python -c "
    from aster.real.properties import *
    t = {'hemolytic': load_property_table('hemolytic', 'data/properties/hemolytic_crossval.csv'),
         'antimicrobial': load_property_table('antimicrobial', 'data/properties/antimicrobial_mic.csv')}
    print(task_overlap(t)); print(label_agreement(t))
    print(threshold_sensitivity(t['hemolytic']))"

### What those checks returned, 28 September 2026

897 peptides are measured for both properties, 58% of the smaller set, so
cross-task transfer is measurable rather than a comparison of two populations.

The two labels disagree for 413 of those 897 (46%):

| antimicrobial | hemolytic | n | reading |
| --- | --- | --- | --- |
| 1 | 0 | 390 | kills microbes, spares blood cells — the therapeutic-index case |
| 1 | 1 | 361 | kills microbes and lyses blood cells |
| 0 | 0 | 123 | does neither |
| 0 | 1 | 23 | lyses blood cells without killing microbes |

That disagreement is the property this design needed and the pathogen tasks
lacked: there, one peptide carried both labels for near-identical questions;
here it carries different labels for genuinely different questions.

Two cautions. The antimicrobial positive rate is 0.84 at the 32 uM cutoff, so
the loader's `balance_tasks` is doing real work; a stricter cutoff (0.67 at
10 uM) balances better and `threshold_sensitivity` is there to check that a
result is not an artefact of where the line sits. And the labels are derived
from the concentration, never from a supplied label column — on the HemoPI2 file
the derived labels agree with the published column on all 1,540 rows, which is
the check that the direction and cutoff are right.

`task_overlap` is the gating number. Cross-task transfer is only measurable if a
useful number of peptides are measured for both properties; if the overlap is
near zero, a cross-task result describes two populations rather than transfer.
