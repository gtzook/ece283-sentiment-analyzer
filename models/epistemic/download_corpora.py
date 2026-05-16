#!/usr/bin/env python3
"""
Download epistemic certainty corpora to /mldata/ece283-sentiment-analyzer/epistemic/raw/.

Corpora downloaded
------------------
uncertainty.zip  (Szeged NLP Group, CC-BY)
  wiki.xml       – Wikipedia weasel sentences, sentence-level + ccue token annotations
  factbank.xml   – FactBank NYT documents with the same ccue scheme
  bio_*.xml      – Biomedical text (BMC, FlyBase, HBC) with ccue annotations

bioscope.zip     (Szeged NLP Group, CC-BY-2.0)
  abstracts.xml  – BioScope biomedical abstracts, richer xcope+cue annotations
  full_papers.xml– BioScope full papers

mpqa_lexicon     – MPQA Subjectivity Lexicon
  Not reachable from this cluster. A data-derived hedge word lexicon will be
  built from the ccue/cue tokens in the corpora above and saved to
  hedge_lexicon/hedge_words.tsv by data.py at load time.

Usage
-----
    python models/epistemic/download_corpora.py [--target DIR]

Default target: /mldata/ece283-sentiment-analyzer/epistemic/raw
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

_UNCERTAINTY_URL = (
    "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/uncertainty.zip"
)
_BIOSCOPE_URL = (
    "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/bioscope.zip"
)

_UNCERTAINTY_FILES = [
    "wiki.xml",
    "factbank.xml",
    "bio_bmc.xml",
    "bio_fly.xml",
    "bio_hbc.xml",
    "Uncertainty.dtd",
]

_BIOSCOPE_FILES = [
    "abstracts.xml",
    "full_papers.xml",
    "BioScope.dtd",
    "clinical_merger/clinical_records_anon.xml",
]


def _download_bytes(url: str, desc: str) -> bytes:
    print(f"  Downloading {desc} …", flush=True)
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    chunks = []
    total = int(r.headers.get("content-length", 0))
    received = 0
    for chunk in r.iter_content(chunk_size=65536):
        chunks.append(chunk)
        received += len(chunk)
        if total:
            pct = received * 100 // total
            print(f"\r    {pct}% ({received:,}/{total:,} bytes)", end="", flush=True)
    print()
    return b"".join(chunks)


def _extract_subset(zip_bytes: bytes, members: list[str], dest: Path) -> None:
    """Extract only the listed members (by exact name) from a zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        available = {info.filename for info in zf.infolist()}
        for name in members:
            if name not in available:
                print(f"  WARNING: {name!r} not found in archive (skipping)")
                continue
            out_path = dest / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(name)
            out_path.write_bytes(data)
            print(f"  Extracted {name}  ({len(data):,} bytes)")


def download_uncertainty(dest: Path, force: bool = False) -> bool:
    """Download and extract uncertainty.zip (wiki + factbank + biomedical)."""
    sentinel = dest / "wiki.xml"
    if sentinel.exists() and not force:
        print(f"  uncertainty corpus already present at {dest}, skipping.")
        return True
    try:
        data = _download_bytes(_UNCERTAINTY_URL, "uncertainty.zip (~2.4 MB)")
        _extract_subset(data, _UNCERTAINTY_FILES, dest)
        return True
    except Exception as exc:
        print(f"  ERROR downloading uncertainty.zip: {exc}", file=sys.stderr)
        return False


def download_bioscope(dest: Path, force: bool = False) -> bool:
    """Download and extract bioscope.zip (BioScope corpus)."""
    sentinel = dest / "abstracts.xml"
    if sentinel.exists() and not force:
        print(f"  BioScope corpus already present at {dest}, skipping.")
        return True
    try:
        data = _download_bytes(_BIOSCOPE_URL, "bioscope.zip (~1.1 MB)")
        _extract_subset(data, _BIOSCOPE_FILES, dest)
        return True
    except Exception as exc:
        print(f"  ERROR downloading bioscope.zip: {exc}", file=sys.stderr)
        return False


def verify(raw_dir: Path) -> None:
    """Print a quick sanity check on the downloaded files."""
    import xml.etree.ElementTree as ET

    print("\n--- Verification ---")
    checks = [
        ("uncertainty/wiki.xml", "Sentence", "certain/uncertain Wikipedia"),
        ("uncertainty/factbank.xml", "Sentence", "certain/uncertain FactBank NYT"),
        ("uncertainty/bio_bmc.xml", "Sentence", "biomedical (BMC)"),
        ("uncertainty/bio_fly.xml", "Sentence", "biomedical (FlyBase)"),
        ("uncertainty/bio_hbc.xml", "Sentence", "biomedical (HBC)"),
        ("bioscope/abstracts.xml", "sentence", "BioScope abstracts"),
        ("bioscope/full_papers.xml", "sentence", "BioScope papers"),
    ]
    for rel, tag, label in checks:
        path = raw_dir / rel
        if not path.exists():
            print(f"  MISSING  {rel}")
            continue
        try:
            root = ET.parse(path).getroot()
            count = sum(1 for _ in root.iter(tag))
            print(f"  OK  {rel:<45}  {count:6,} <{tag}> elements  [{label}]")
        except Exception as exc:
            print(f"  ERROR parsing {rel}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="/mldata/ece283-sentiment-analyzer/epistemic/raw",
        help="Root directory for raw corpus files",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    raw_dir = Path(args.target)

    print("=== Downloading uncertainty corpus (wiki + factbank + biomedical) ===")
    download_uncertainty(raw_dir / "uncertainty", force=args.force)

    print("\n=== Downloading BioScope corpus ===")
    download_bioscope(raw_dir / "bioscope", force=args.force)

    print("\n=== MPQA Subjectivity Lexicon ===")
    print("  mpqa.cs.pitt.edu is not reachable from this cluster.")
    print("  A data-derived hedge lexicon will be built from ccue/cue tokens at load time.")

    verify(raw_dir)
    print("\nDone. Raw files at:", raw_dir)


if __name__ == "__main__":
    main()
