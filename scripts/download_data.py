"""
Data download helper for rxgnn-tcm-tox.

Automates what can be automated; prints clear instructions for
gated sources that require registration or licence.

Usage
-----
python scripts/download_data.py --data_dir data/raw
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         rxgnn-tcm-tox  —  Data Download Helper              ║
╚══════════════════════════════════════════════════════════════╝
"""

# Databases that can be downloaded programmatically
AUTO_SOURCES: list[dict] = [
    # TOXRIC DILI endpoint (public, no login)
    {
        "name": "TOXRIC DILI",
        "url":  "https://toxric.bioinformatics.ac.cn/static/download/DILI.csv",
        "dest": "toxric_dili.csv",
        "note": "Drug-Induced Liver Injury labels (~1k compounds)",
    },
]

# Databases requiring manual steps
MANUAL_SOURCES: list[dict] = [
    {
        "name": "HERB 2.0 ingredients",
        "url":  "http://herb.ac.cn/Download/",
        "files": ["HERB_ingredient_info.txt", "HERB_herb_ingredient.txt"],
        "note": "Free after registration. Download 'Ingredient info' and 'Herb-ingredient link'.",
    },
    {
        "name": "SuperCYP",
        "url":  "https://bioinformatics.charite.de/supercyp/",
        "files": ["supercyp_interactions.csv"],
        "note": (
            "Email authors (supercyp@charite.de) with institution + purpose. "
            "Response typically within 48h. Ask for the 'interactions CSV export'."
        ),
    },
    {
        "name": "DrugBank metabolites",
        "url":  "https://go.drugbank.com/releases/latest",
        "files": ["drugbank_metabolites.csv"],
        "note": (
            "Register for academic licence (free). Download 'All Drug Links' CSV, "
            "then filter for metabolite relationships."
        ),
    },
]


def download_file(url: str, dest: Path, verbose: bool = True) -> bool:
    try:
        if verbose:
            print(f"  Downloading {dest.name} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size / 1024
        if verbose:
            print(f"OK ({size:.0f} KB)")
        return True
    except Exception as e:
        if verbose:
            print(f"FAILED ({e})")
        return False


def check_existing(data_dir: Path) -> dict[str, bool]:
    expected = [
        "HERB_ingredient_info.txt",
        "HERB_herb_ingredient.txt",
        "supercyp_interactions.csv",
        "drugbank_metabolites.csv",
        "toxric_dili.csv",
    ]
    return {f: (data_dir / f).exists() for f in expected}


def main():
    p = argparse.ArgumentParser(description="Download rxgnn-tcm-tox data sources")
    p.add_argument("--data_dir", default="data/raw", help="Destination directory")
    p.add_argument("--skip_manual", action="store_true", help="Only show status, skip instructions")
    args = p.parse_args()

    print(BANNER)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Status check
    status = check_existing(data_dir)
    print("Current status:")
    for fname, present in status.items():
        mark = "✅" if present else "❌"
        print(f"  {mark}  {fname}")
    print()

    # Auto-download
    print("Auto-downloadable sources:")
    for src in AUTO_SOURCES:
        dest = data_dir / src["dest"]
        if dest.exists():
            print(f"  ✅ {src['name']} already present, skipping.")
            continue
        download_file(src["url"], dest)

    print()

    if not args.skip_manual:
        print("Manual download required (gated sources):")
        print("─" * 62)
        for src in MANUAL_SOURCES:
            print(f"
📦 {src['name']}")
            print(f"   URL   : {src['url']}")
            print(f"   Files : {', '.join(src['files'])}")
            print(f"   Steps : {src['note']}")
            print(f"   Dest  : {data_dir}/")

        print("
" + "─" * 62)
        print("Once all files are in data/raw/, run:")
        print("  python scripts/train.py --config configs/default.yaml")

    # Final status
    status = check_existing(data_dir)
    ready  = sum(status.values())
    print(f"
Data readiness: {ready}/{len(status)} files present.")
    if ready < len(status):
        sys.exit(1)


if __name__ == "__main__":
    main()