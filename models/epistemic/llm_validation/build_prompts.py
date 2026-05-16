#!/usr/bin/env python3
"""
Build the LLM validation prompt suite for epistemic certainty label mapping.

Samples 250 sentences from wiki.xml (50 per cue category), shuffles them,
splits into 5 prompt batches of 50 sentences each, and writes:

  sentences.jsonl        -- gold-rule labels + metadata for all 250 sentences
  prompts/batch_NN.txt   -- prompt files to paste into each LLM
  outputs/README.txt     -- instructions for naming LLM output files

Usage:
    python build_prompts.py [--wiki PATH] [--seed INT] [--batch-size INT]
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_WIKI = (
    "/mldata/ece283-sentiment-analyzer/epistemic/raw/uncertainty/wiki.xml"
)
OUT_DIR = Path(__file__).parent

# Maps cue type (stripped) → rule label
CUE_TYPE_TO_LABEL: dict[str, str] = {
    "none": "asserted",
    "speculation_modal_probable_": "hedged",
    "speculation_hypo_doxastic _": "hedged",
    "speculation_hypo_condition _": "speculative",   # <-- debatable, under review
    "speculation_hypo_investigation _": "speculative",
}

# Display-friendly names used in prompt metadata comments
CUE_DISPLAY: dict[str, str] = {
    "none": "no_ccue",
    "speculation_modal_probable_": "modal_probable",
    "speculation_hypo_doxastic _": "hypo_doxastic",
    "speculation_hypo_condition _": "hypo_condition",
    "speculation_hypo_investigation _": "hypo_investigation",
}

PROMPT_TEMPLATE = textwrap.dedent("""\
    You're labeling sentences for epistemic certainty in news/encyclopedia text.
    Pick exactly one label per sentence:

    - asserted: the writer states the content as fact, with confidence.
      Example: "The company reported a 12% increase in revenue."
    - hedged: the writer signals partial uncertainty or qualification,
      but is making a claim. Example: "The policy may have contributed to the decline."
    - speculative: the writer presents the content as unverified, hypothetical,
      attributed-but-unconfirmed, or conditional on something unknown.
      Example: "Reportedly, the suspect fled the country."

    Output one JSON object per sentence, on its own line:
    {{"id": 1, "label": "asserted"}}

    Sentences:
    {sentences}
""")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SentenceRecord:
    global_id: int
    batch: int
    local_id: int
    text: str
    rule_label: str
    primary_cue_type: str          # the type used to select this sentence
    all_cue_types: list[str]       # all ccue types present (may differ if mixed)
    cue_words: list[str]
    is_mixed: bool                  # True if sentence has >1 distinct cue type


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _sentence_text(sent_el: ET.Element) -> str:
    """Return plain text of a <Sentence> element (cue words in place, no tags)."""
    return "".join(sent_el.itertext()).strip()


def _sentence_cue_info(sent_el: ET.Element) -> tuple[list[str], list[str]]:
    """Return (list_of_distinct_types, list_of_cue_words) for a <Sentence>."""
    ccues = sent_el.findall("ccue")
    types = sorted({c.get("type", "").strip() for c in ccues})
    words = [c.text or "" for c in ccues]
    return types, words


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _collect_pools(
    wiki_path: str,
) -> dict[str, list[tuple[str, list[str], list[str]]]]:
    """
    Parse wiki.xml and return pools of (text, cue_types, cue_words) per category.

    Pool keys: "none" + each cue type string.
    Within each pool, clean (single-type) entries come before mixed entries.
    """
    root = ET.parse(wiki_path).getroot()

    pools: dict[str, list] = {k: [] for k in CUE_TYPE_TO_LABEL}

    for sent_el in root.iter("Sentence"):
        text = _sentence_text(sent_el)
        if not text:
            continue
        types, words = _sentence_cue_info(sent_el)

        if not types:
            pools["none"].append((text, [], []))
        else:
            for t in types:
                if t in pools:
                    pools[t].append((text, types, words))

    # Sort each uncertain pool so clean entries come first (single-type)
    for t in list(CUE_TYPE_TO_LABEL.keys()):
        if t == "none":
            continue
        pools[t].sort(key=lambda x: (len(x[1]) > 1, 0))  # clean first

    return pools


def sample_sentences(
    wiki_path: str,
    n_per_category: int = 50,
    seed: int = 42,
) -> list[SentenceRecord]:
    """Sample n_per_category sentences from each cue-type category."""
    rng = random.Random(seed)
    pools = _collect_pools(wiki_path)

    records: list[SentenceRecord] = []
    global_id = 1

    categories = list(CUE_TYPE_TO_LABEL.keys())
    for cat in categories:
        pool = pools[cat]
        if len(pool) < n_per_category:
            print(
                f"  WARNING: only {len(pool)} sentences available for {cat!r} "
                f"(wanted {n_per_category}); using all."
            )
            chosen = pool
        else:
            chosen = rng.sample(pool, n_per_category)

        for text, types, words in chosen:
            records.append(
                SentenceRecord(
                    global_id=global_id,
                    batch=-1,          # filled in after shuffle
                    local_id=-1,
                    text=text,
                    rule_label=CUE_TYPE_TO_LABEL[cat],
                    primary_cue_type=cat,
                    all_cue_types=types,
                    cue_words=words,
                    is_mixed=(len(types) > 1),
                )
            )
            global_id += 1

    return records


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def assign_batches(records: list[SentenceRecord], batch_size: int, seed: int) -> None:
    """Shuffle records in-place and assign batch / local_id fields."""
    rng = random.Random(seed + 1)
    rng.shuffle(records)
    for i, rec in enumerate(records):
        rec.batch = i // batch_size + 1
        rec.local_id = i % batch_size + 1
        rec.global_id = i + 1   # re-assign sequential global IDs after shuffle


def build_prompt(batch_records: list[SentenceRecord]) -> str:
    lines = []
    for rec in batch_records:
        lines.append(f"{rec.global_id}. {rec.text}")
    return PROMPT_TEMPLATE.format(sentences="\n".join(lines))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_sentences_jsonl(records: list[SentenceRecord], out_dir: Path) -> None:
    path = out_dir / "sentences.jsonl"
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")
    print(f"Wrote {len(records)} sentences → {path}")


def write_prompts(records: list[SentenceRecord], out_dir: Path) -> None:
    batches: dict[int, list[SentenceRecord]] = {}
    for rec in records:
        batches.setdefault(rec.batch, []).append(rec)

    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    for batch_num in sorted(batches):
        batch_recs = batches[batch_num]
        prompt_text = build_prompt(batch_recs)
        path = prompts_dir / f"batch_{batch_num:02d}.txt"
        path.write_text(prompt_text)
        label_dist = {}
        for rec in batch_recs:
            label_dist[rec.rule_label] = label_dist.get(rec.rule_label, 0) + 1
        print(f"  Wrote batch {batch_num:02d} ({len(batch_recs)} sentences, "
              f"rule-label dist: {label_dist}) → {path}")


def write_outputs_readme(out_dir: Path, n_sentences: int) -> None:
    path = out_dir / "outputs" / "README.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(textwrap.dedent(f"""\
        Drop LLM output files here.

        Expected format
        ---------------
        One JSONL file per model, containing ALL {n_sentences} labeled sentences
        (across all batches) concatenated together:

            {{"id": 1, "label": "asserted"}}
            {{"id": 2, "label": "hedged"}}
            ...

        Naming convention:  <model_name>.jsonl
        Examples:
            gpt4o.jsonl
            claude_sonnet.jsonl
            gemini_pro.jsonl

        If you have per-batch output files, concatenate them:
            cat batch_01_output.txt batch_02_output.txt ... > gpt4o.jsonl

        Then run:
            python compute_agreement.py
    """))
    print(f"Wrote outputs README → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki", default=DEFAULT_WIKI)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--n-per-category", type=int, default=50)
    args = parser.parse_args()

    print(f"Sampling {args.n_per_category} sentences per category (seed={args.seed})...")
    records = sample_sentences(args.wiki, n_per_category=args.n_per_category, seed=args.seed)

    print(f"Shuffling and assigning batches (batch_size={args.batch_size})...")
    assign_batches(records, batch_size=args.batch_size, seed=args.seed)

    total = len(records)
    n_batches = max(r.batch for r in records)
    print(f"\nTotal sentences: {total}  |  Batches: {n_batches}")
    print(f"Category breakdown:")
    from collections import Counter
    ctr = Counter(r.primary_cue_type for r in records)
    for cat, count in ctr.items():
        mixed_count = sum(1 for r in records if r.primary_cue_type == cat and r.is_mixed)
        print(f"  {CUE_DISPLAY[cat]:<25} {count:3d}  ({mixed_count} mixed)")

    print()
    write_sentences_jsonl(records, OUT_DIR)
    write_prompts(records, OUT_DIR)
    write_outputs_readme(OUT_DIR, total)
    print("\nDone. Paste prompt files into each LLM, then drop output files in outputs/.")


if __name__ == "__main__":
    main()
