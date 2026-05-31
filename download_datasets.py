"""
Download all MAGPIE datasets to a local directory.

Edit DOWNLOAD_DIR below to set the destination folder.
Run from the repo root:  python download_datasets.py
"""

from __future__ import annotations

import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Configure this ────────────────────────────────────────────────────────────
DOWNLOAD_DIR = "/mldata/ece283-sentiment-analyzer/"
# ─────────────────────────────────────────────────────────────────────────────

# Optional filters (set to True to restrict the download set)
NEWS_ONLY = False          # only datasets flagged news_relevant=True
SKIP_SEQUENCE_LABEL = True # skip pure sequence-labelling tasks (different format)
FORCE = False              # re-download even if file already exists

_GITHUB_BASE = (
    "https://raw.githubusercontent.com/Media-Bias-Group/"
    "magpie-multi-task/main/datasets/{dataset_id}/preprocessed.csv"
)

# All dataset IDs from registry.py
_ALL_IDS = [
    # News-domain
    "10_BABE", "9_BASIL", "12_PHEME", "19_MultiDimNews", "22_NewsWCL50",
    "25_FakeNewsNet", "26_neutralizing-bias", "72_LIAR", "96_Bu-NEMO",
    "42_GoodNewsEveryone", "120_SemEval2023Task3", "03_CW_HARD",
    # General sentiment
    "99_SST2", "63_semeval2014", "100_Amazon_reviews", "101_IMDB",
    "103_MPQA", "31_SUBJ",
    # Emotion
    "84_emotion_tweets",
    # Bias / stereotype
    "33_CrowSPairs", "64_StereoSet", "109_stereotype",
    # Gender bias
    "18_GAP", "105_RtGender", "116_MDGender", "117_Funpedia",
    "118_WizardsOfWikipedia", "91_WikiMadlibs",
    # Hate speech / toxicity
    "40_JIGSAW", "86_OffensiveLanguage", "87_OnlineHarassmentDataset",
    "88_HatespeechTwitter", "92_HateXplain", "891_WikiDetoxToxicity",
    "892_WikiDetoxAggressionAndAttack",
    # Stance
    "119_SemEval2023Task4", "124_SemEval2016Task6", "125_MultiTargetStance",
    "126_WTWT", "127_VaccineLies",
    # Other
    "38_starbucks", "75_RedditBias", "80_DebateEffects", "104_TRAC2",
    "108_MeTooMA", "128_GWSD",
]

# Datasets that are purely sequence-labelling tasks
_SEQUENCE_LABEL_IDS = {"42_GoodNewsEveryone"}

# Datasets flagged as news-relevant
_NEWS_IDS = {
    "10_BABE", "9_BASIL", "12_PHEME", "19_MultiDimNews", "22_NewsWCL50",
    "25_FakeNewsNet", "26_neutralizing-bias", "72_LIAR", "96_Bu-NEMO",
    "42_GoodNewsEveryone", "120_SemEval2023Task3", "03_CW_HARD",
}


def _select_ids() -> list[str]:
    ids = list(_ALL_IDS)
    if NEWS_ONLY:
        ids = [i for i in ids if i in _NEWS_IDS]
    if SKIP_SEQUENCE_LABEL:
        ids = [i for i in ids if i not in _SEQUENCE_LABEL_IDS]
    return ids


def download_one(dataset_id: str, dest_root: Path) -> Path | None:
    dest = dest_root / dataset_id / "preprocessed.csv"
    if dest.exists() and not FORCE:
        print(f"  [cached]  {dataset_id}")
        return dest

    url = _GITHUB_BASE.format(dataset_id=dataset_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [fetch]   {dataset_id} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f" done ({dest.stat().st_size // 1024} KB)")
        return dest
    except urllib.error.HTTPError as exc:
        print(f" FAILED (HTTP {exc.code})")
        print(
            f"           {dataset_id} may require separate licensing — see "
            f"https://github.com/Media-Bias-Group/magpie-multi-task/tree/main/datasets/{dataset_id}"
        )
        return None


def main() -> None:
    dest_root = Path(DOWNLOAD_DIR).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading datasets to: {dest_root}\n")

    ids = _select_ids()
    print(f"Datasets to fetch: {len(ids)}\n")

    ok, failed = [], []
    for dataset_id in ids:
        path = download_one(dataset_id, dest_root)
        (ok if path else failed).append(dataset_id)

    print(f"\nDone. {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
