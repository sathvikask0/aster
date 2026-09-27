"""Fetch and prepare the 56-pathogen antimicrobial peptide benchmark dataset.
Curated from 7 databases (APD3, CAMP, DBAMP, DRAMP, SATPdb, YADAMP, LAMP).
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path
import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/wyky481l/KPPepGen/main/data/source/amp"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "amp"

FILES = {
    "peptide_pathogen_triple.csv": f"{RAW_BASE}/peptide_pathogen_triple.csv",
    "pathogen_description.csv": f"{RAW_BASE}/pathogen_description.csv",
    "amp_peptide.fasta": f"{RAW_BASE}/amp_peptide.fasta",
}


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        dst = DATA_DIR / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[exists] {fname} ({dst.stat().st_size:,} bytes)")
            continue
        print(f"[downloading] {fname} from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
        dst.write_bytes(content)
        print(f"[saved] {fname} ({len(content):,} bytes)")

    # Load and print summary
    triples_path = DATA_DIR / "peptide_pathogen_triple.csv"
    desc_path = DATA_DIR / "pathogen_description.csv"

    df_triples = pd.read_csv(triples_path)
    df_desc = pd.read_csv(desc_path)

    print("\n--- Dataset Summary ---")
    print(f"Total peptide-pathogen associations: {len(df_triples):,}")
    print(f"Unique peptide sequences: {df_triples['sequence'].nunique():,}")
    print(f"Unique pathogens: {df_triples['pathogen'].nunique():,}")
    print(f"Pathogen descriptions: {len(df_desc):,} categories")


if __name__ == "__main__":
    download()
