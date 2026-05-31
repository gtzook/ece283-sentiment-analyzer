#!/usr/bin/env python3
"""
Dump N sentences per cue type to a CSV for manual review.

Columns:
  global_id, cue_type, rule_label, cue_words, text_plain, text_highlighted
  (text_highlighted wraps each cue word as <<word>>)

Usage:
    python dump_cue_review.py [--n 20] [--seed 42] [--out cue_review.csv]
"""

from __future__ import annotations

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

WIKI_PATH = "/mldata/ece283-sentiment-analyzer/epistemic/raw/uncertainty/wiki.xml"

TARGET_TYPES = [
    "speculation_hypo_investigation _",
    "speculation_hypo_doxastic _",
    "speculation_hypo_condition _",
    "speculation_modal_probable_",   # include control group for comparison
]

CUE_DISPLAY = {
    "speculation_hypo_investigation _": "hypo_investigation",
    "speculation_hypo_doxastic _":      "hypo_doxastic",
    "speculation_hypo_condition _":     "hypo_condition",
    "speculation_modal_probable_":      "modal_probable",
}

RULE_LABEL = {
    "speculation_hypo_investigation _": "speculative",
    "speculation_hypo_doxastic _":      "hedged",
    "speculation_hypo_condition _":     "speculative",
    "speculation_modal_probable_":      "hedged",
}


@dataclass
class Row:
    global_id: int
    cue_type: str
    rule_label: str
    cue_words: str          # comma-separated
    text_plain: str
    text_highlighted: str   # cue words wrapped in <<...>>
    is_mixed: bool          # has additional cue types beyond the primary


def _extract(sent_el: ET.Element, primary_type: str) -> Row | None:
    ccues = sent_el.findall("ccue")
    all_types = [c.get("type", "").strip() for c in ccues]
    if primary_type not in all_types:
        return None

    # Build plain text and highlighted text in one pass
    plain_parts: list[str] = []
    hl_parts: list[str] = []
    cue_words: list[str] = []

    if sent_el.text:
        plain_parts.append(sent_el.text)
        hl_parts.append(sent_el.text)

    for child in sent_el:
        if child.tag != "ccue":
            continue
        word = child.text or ""
        ctype = child.get("type", "").strip()
        plain_parts.append(word)
        if ctype == primary_type:
            hl_parts.append(f"<<{word}>>")
            cue_words.append(word)
        else:
            hl_parts.append(f"[{word}]")   # secondary cue in square brackets
        if child.tail:
            plain_parts.append(child.tail)
            hl_parts.append(child.tail)

    other_types = [t for t in set(all_types) if t != primary_type]
    return Row(
        global_id=0,            # filled in by caller
        cue_type=CUE_DISPLAY[primary_type],
        rule_label=RULE_LABEL[primary_type],
        cue_words=", ".join(cue_words),
        text_plain="".join(plain_parts).strip(),
        text_highlighted="".join(hl_parts).strip(),
        is_mixed=bool(other_types),
    )


def collect(wiki_path: str, target_types: list[str]) -> dict[str, list[Row]]:
    root = ET.parse(wiki_path).getroot()
    pools: dict[str, list[Row]] = {t: [] for t in target_types}

    for sent_el in root.iter("Sentence"):
        ccues = sent_el.findall("ccue")
        all_types = {c.get("type", "").strip() for c in ccues}
        for primary in target_types:
            if primary in all_types:
                row = _extract(sent_el, primary)
                if row and row.text_plain:
                    pools[primary].append(row)

    return pools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Sentences per cue type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "cue_review.csv"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pools = collect(WIKI_PATH, TARGET_TYPES)

    rows: list[Row] = []
    gid = 1
    for t in TARGET_TYPES:
        pool = pools[t]
        # Prefer clean (single-type) examples first, then mixed
        clean = [r for r in pool if not r.is_mixed]
        mixed = [r for r in pool if r.is_mixed]
        n_clean = min(args.n, len(clean))
        n_mixed = max(0, args.n - n_clean)
        chosen = rng.sample(clean, n_clean) + rng.sample(mixed, min(n_mixed, len(mixed)))
        rng.shuffle(chosen)
        for row in chosen:
            row.global_id = gid
            gid += 1
            rows.append(row)
        display = CUE_DISPLAY[t]
        print(f"  {display:<25} {len(chosen):3d} rows  "
              f"({n_clean} clean, {len(chosen)-n_clean} mixed)")

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "cue_type", "rule_label", "is_mixed",
                "cue_words", "text_highlighted", "text_plain",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id":               row.global_id,
                "cue_type":         row.cue_type,
                "rule_label":       row.rule_label,
                "is_mixed":         row.is_mixed,
                "cue_words":        row.cue_words,
                "text_highlighted": row.text_highlighted,
                "text_plain":       row.text_plain,
            })

    print(f"\nWrote {len(rows)} rows → {out_path}")
    print("Legend: <<word>> = primary cue  |  [word] = co-occurring cue of other type")


if __name__ == "__main__":
    main()
