#!/usr/bin/env python
"""Fetch and prepare the property datasets with measured negatives.

The hemolysis file is downloaded by hand (see data/properties/README.md); this
script prepares the antimicrobial half from GRAMPA, a published aggregation of
MIC measurements drawn from DBAASP, DRAMP, YADAMP, APD and DADP.

Why GRAMPA rather than the pathogen tables already in data/amp. Those tables are
presence-only: a row means "reported active", and absence is not a measurement.
GRAMPA carries the MIC *value* in uM, so a peptide that needed 500 uM was tested
and found weak. That is a recorded negative, which is the whole point of moving
to property tasks.

Two choices worth arguing with rather than accepting.

**Aggregation.** A peptide is assayed against many organisms. This takes the
minimum MIC across organisms: "is this peptide antimicrobial at all", answered by
its best result. That is deliberately the property-level question, not the
per-pathogen one -- the per-pathogen framing is what failed, because most AMPs
are broad-spectrum and the tasks collapsed into each other.

**Modifications.** Over 40% of GRAMPA rows are chemically modified peptides
(disulfide bonds, C-terminal amidation). The protein encoder sees only the amino
acid string, so a modified peptide's activity is partly produced by something the
model cannot see. `--keep-modified` includes them; the default drops the ones
carrying unusual modifications, and records the choice in the output header.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

GRAMPA_URL = ("https://raw.githubusercontent.com/zswitten/"
              "Antimicrobial-Peptides/master/data/grampa.csv")
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "properties"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=GRAMPA_URL)
    p.add_argument("--cache", default=None,
                   help="Read a previously downloaded grampa.csv instead of fetching")
    p.add_argument("--out", default=str(OUT_DIR / "antimicrobial_mic.csv"))
    p.add_argument("--keep-modified", action="store_true",
                   help="Keep peptides with unusual chemical modifications")
    p.add_argument("--aggregate", default="min", choices=["min", "median"],
                   help="How to reduce many organisms to one MIC per peptide")
    args = p.parse_args()

    if args.cache:
        frame = pd.read_csv(args.cache)
    else:
        print(f"downloading {args.url}", flush=True)
        with urllib.request.urlopen(args.url) as response:
            frame = pd.read_csv(response)
    print(f"  {len(frame):,} measurements")

    units = set(frame["unit"].unique())
    if units != {"uM"}:
        raise SystemExit(f"expected uM throughout, found {sorted(units)}")

    # GRAMPA stores log10(MIC in uM).
    frame["mic_um"] = 10 ** frame["value"]

    if not args.keep_modified:
        before = len(frame)
        frame = frame[~frame["has_unusual_modification"].astype(bool)]
        print(f"  dropped {before - len(frame):,} rows with unusual modifications")

    agg = frame.groupby("sequence")["mic_um"].agg(args.aggregate).reset_index()
    agg.columns = ["SEQUENCE", "uM"]
    agg = agg.sort_values("SEQUENCE").reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    weak = (agg["uM"] > 32).sum()
    print(f"  wrote {out} | {len(agg):,} peptides | "
          f"{weak:,} ({weak / len(agg):.1%}) with MIC > 32 uM, i.e. measured negatives")
    print("  aggregate=%s keep_modified=%s" % (args.aggregate, args.keep_modified))
    return 0


if __name__ == "__main__":
    sys.exit(main())
