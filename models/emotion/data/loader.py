"""
Dataset loading for the emotional framing floor model.

Four data sources merged before the 80/10/10 stratified split:

  1. MAGPIE 84_emotion_tweets (local, /mldata/)
     ~196k tweets, 8-class Plutchik multiclass → 11-d multi-hot.
     Covers: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.
     Missing: love, optimism, pessimism.

  2. GoEmotions simplified (HuggingFace: google-research-datasets/go_emotions)
     ~35k Reddit comments after filtering, 27 emotions → 11-label multi-hot.
     Conservative mapping; neutral-only and fully-unmapped examples dropped.
     Adds coverage for: love, optimism, pessimism (via disappointment/grief).

  3. dair-ai/emotion (HuggingFace)
     ~20k sentences, 6 single-label classes → 11-d one-hot.
     Covers: sadness, joy, love, anger, fear, surprise.
     Primary value: clean love signal.

  4. TweetEval emotion (HuggingFace: cardiffnlp/tweet_eval, config=emotion)
     ~5k tweets, 4 single-label classes → 11-d one-hot.
     Covers: anger, joy, optimism, sadness.
     Primary value: clean optimism signal.

NOTE: SemEval-2018 Task 1 E-c is unavailable via HF datasets v5 (deprecated
dataset scripts). All 11 labels are covered across the four sources above.
Weakest coverage: trust (GoEmotions approval/admiration only).

Split: 80/10/10 using iterative stratification (scikit-multilearn) on the
merged label matrix to preserve per-label prevalence across splits.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EmotionalFramingConfig, EMOTION_LABELS, MAGPIE_LABEL_TO_SEMEVAL_IDX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GoEmotions → SemEval label mapping
# ---------------------------------------------------------------------------

# GoEmotions simplified label order — index matches the integer in 'labels' column
_GOEMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Conservative mapping: only include labels with clear semantic overlap.
# Ambiguous labels (confusion, disapproval, embarrassment, neutral) are omitted.
_GOEMOTION_TO_SEMEVAL: dict[str, str] = {
    "anger":          "anger",
    "annoyance":      "anger",        # mild anger
    "curiosity":      "anticipation",
    "excitement":     "anticipation",
    "desire":         "anticipation",
    "disgust":        "disgust",
    "fear":           "fear",
    "nervousness":    "fear",
    "amusement":      "joy",
    "joy":            "joy",
    "pride":          "joy",
    "relief":         "joy",
    "admiration":     "love",
    "caring":         "love",
    "gratitude":      "love",
    "love":           "love",
    "optimism":       "optimism",     # direct match
    "disappointment": "pessimism",
    "grief":          "pessimism",
    "sadness":        "sadness",
    "remorse":        "sadness",
    "surprise":       "surprise",
    "realization":    "surprise",
    "approval":       "trust",
}

_GOEMOTION_TO_SEMEVAL_IDX: dict[str, int] = {
    ge: EMOTION_LABELS.index(sem)
    for ge, sem in _GOEMOTION_TO_SEMEVAL.items()
}

# ---------------------------------------------------------------------------
# Single-label dataset schemas
# ---------------------------------------------------------------------------

# tweet_eval emotion: ClassLabel(names=['anger', 'joy', 'optimism', 'sadness'])
_TWEET_EVAL_LABELS = ["anger", "joy", "optimism", "sadness"]

# dair-ai/emotion: ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
_DAIRAI_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def _single_label_to_multihot(int_label: int, source_labels: list[str]) -> list[int]:
    """Convert a single-class integer label to an 11-d multi-hot vector."""
    vec = [0] * len(EMOTION_LABELS)
    name = source_labels[int_label]
    if name in EMOTION_LABELS:
        vec[EMOTION_LABELS.index(name)] = 1
    return vec


# ---------------------------------------------------------------------------
# Source loaders — each returns (texts, labels, sources) or None on failure
# ---------------------------------------------------------------------------

def _load_magpie_emotion(data_dir: str) -> Optional[Tuple[list, list, list]]:
    """
    Load 84_emotion_tweets/preprocessed.csv (local).

    CSV: text, label (0-indexed integer 0–7, Plutchik alphabetical).
    Converted to 11-d multi-hot; love/optimism/pessimism are always 0 here.
    """
    import csv

    path = Path(data_dir) / "84_emotion_tweets" / "preprocessed.csv"
    if not path.exists():
        logger.warning("MAGPIE emotion data not found at %s", path)
        return None

    texts, labels, sources = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "").strip()
            if not text:
                continue
            vec = [0] * len(EMOTION_LABELS)
            idx = MAGPIE_LABEL_TO_SEMEVAL_IDX.get(int(row["label"]))
            if idx is not None:
                vec[idx] = 1
            texts.append(text)
            labels.append(vec)
            sources.append("magpie")

    logger.info("MAGPIE 84_emotion_tweets: %d examples", len(texts))
    return texts, labels, sources


def _load_goemotion() -> Optional[Tuple[list, list, list]]:
    """
    Load GoEmotions simplified from HF Hub (~35k after filtering).

    27 GoEmotions labels → 11 SemEval labels via conservative mapping.
    Neutral-only and fully-unmapped examples are dropped.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception as e:
        logger.warning("Could not load GoEmotions: %s — skipping", e)
        return None

    neutral_idx = _GOEMOTION_LABELS.index("neutral")
    texts, labels, sources = [], [], []
    dropped_neutral = dropped_unmapped = 0

    for split_name, split_ds in ds.items():
        for ex in split_ds:
            active: list[int] = ex["labels"]
            if active == [neutral_idx]:
                dropped_neutral += 1
                continue
            vec = [0] * len(EMOTION_LABELS)
            for ge_idx in active:
                sem_idx = _GOEMOTION_TO_SEMEVAL_IDX.get(_GOEMOTION_LABELS[ge_idx])
                if sem_idx is not None:
                    vec[sem_idx] = 1
            if sum(vec) == 0:
                dropped_unmapped += 1
                continue
            texts.append(ex["text"].strip())
            labels.append(vec)
            sources.append("goemotion")

    logger.info(
        "GoEmotions: %d examples | %d neutral-only dropped | %d unmapped dropped",
        len(texts), dropped_neutral, dropped_unmapped,
    )
    return texts, labels, sources


def _load_dairai_emotion() -> Optional[Tuple[list, list, list]]:
    """
    Load dair-ai/emotion from HF Hub (~20k).

    6 single-label classes: sadness, joy, love, anger, fear, surprise.
    Primary value: love signal (absent from MAGPIE).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("dair-ai/emotion")
    except Exception as e:
        logger.warning("Could not load dair-ai/emotion: %s — skipping", e)
        return None

    texts, labels, sources = [], [], []
    for split_name, split_ds in ds.items():
        for ex in split_ds:
            text = ex["text"].strip()
            if not text:
                continue
            vec = _single_label_to_multihot(ex["label"], _DAIRAI_LABELS)
            texts.append(text)
            labels.append(vec)
            sources.append("dairai")

    logger.info("dair-ai/emotion: %d examples", len(texts))
    return texts, labels, sources


def _load_tweet_eval_emotion() -> Optional[Tuple[list, list, list]]:
    """
    Load TweetEval emotion from HF Hub (~5k).

    4 single-label classes: anger, joy, optimism, sadness.
    Primary value: optimism signal (absent from MAGPIE).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
    except Exception as e:
        logger.warning("Could not load tweet_eval emotion: %s — skipping", e)
        return None

    texts, labels, sources = [], [], []
    for split_name, split_ds in ds.items():
        for ex in split_ds:
            text = ex["text"].strip()
            if not text:
                continue
            vec = _single_label_to_multihot(ex["label"], _TWEET_EVAL_LABELS)
            texts.append(text)
            labels.append(vec)
            sources.append("tweeteval")

    logger.info("TweetEval emotion: %d examples", len(texts))
    return texts, labels, sources


# ---------------------------------------------------------------------------
# Iterative stratification split
# ---------------------------------------------------------------------------

def _iterative_split(
    indices: np.ndarray,
    label_matrix: np.ndarray,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) using multi-label iterative stratification."""
    try:
        from skmultilearn.model_selection import iterative_train_test_split
    except ImportError:
        logger.warning(
            "scikit-multilearn not installed — falling back to random split. "
            "Install with: pip install scikit-multilearn"
        )
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(indices))
        cut = int(len(indices) * (1 - test_size))
        return indices[perm[:cut]], indices[perm[cut:]]

    X = indices.reshape(-1, 1).astype(float)
    y = label_matrix.astype(float)
    X_tr, _, X_te, _ = iterative_train_test_split(X, y, test_size=test_size)
    return X_tr.astype(int).flatten(), X_te.astype(int).flatten()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_split(cfg: EmotionalFramingConfig) -> "DatasetDict":  # noqa: F821
    """
    Load all emotion sources, merge, and return an 80/10/10 stratified split.

    Returns a HuggingFace DatasetDict with keys "train", "dev", "test".
    Each example has:
      text   (str)       — raw input text
      labels (List[int]) — 11-d multi-hot vector (EMOTION_LABELS order)
      source (str)       — dataset provenance tag
    """
    from datasets import Dataset, DatasetDict

    all_texts: list = []
    all_labels: list = []
    all_sources: list = []

    def _extend(result: Optional[Tuple[list, list, list]]) -> None:
        if result is not None:
            all_texts.extend(result[0])
            all_labels.extend(result[1])
            all_sources.extend(result[2])

    # ── 1. MAGPIE (required — local) ─────────────────────────────────────────
    magpie = _load_magpie_emotion(cfg.magpie_data_dir)
    if magpie is None:
        raise RuntimeError(
            f"MAGPIE emotion data not found under {cfg.magpie_data_dir}/84_emotion_tweets/. "
            "Run download_datasets.py to fetch it."
        )
    _extend(magpie)

    # ── 2–4. Supplementary sources (optional — HF Hub) ───────────────────────
    _extend(_load_goemotion())
    _extend(_load_dairai_emotion())
    _extend(_load_tweet_eval_emotion())

    logger.info("Total before filtering: %d examples", len(all_texts))

    # Drop blank texts
    keep = [i for i, t in enumerate(all_texts) if t.strip()]
    all_texts   = [all_texts[i]   for i in keep]
    all_labels  = [all_labels[i]  for i in keep]
    all_sources = [all_sources[i] for i in keep]

    source_counts = Counter(all_sources)
    logger.info("Source counts: %s", dict(source_counts))

    # Cap dataset size (--debug uses debug_samples, --samples/--fast uses max_samples)
    cap: int | None = None
    if cfg.debug:
        cap = cfg.debug_samples
    elif cfg.max_samples is not None:
        cap = cfg.max_samples

    if cap is not None:
        rng = np.random.default_rng(cfg.seed)
        sel = rng.choice(len(all_texts), size=min(cap, len(all_texts)), replace=False)
        all_texts   = [all_texts[i]   for i in sel]
        all_labels  = [all_labels[i]  for i in sel]
        all_sources = [all_sources[i] for i in sel]
        logger.info("Using %d examples (cap=%d)", len(all_texts), cap)

    label_matrix = np.array(all_labels, dtype=np.int32)  # (N, 11)
    indices = np.arange(len(all_texts))

    # 80 / 10 / 10 with iterative stratification
    train_idx, temp_idx = _iterative_split(indices, label_matrix, test_size=0.20)
    dev_idx, test_idx   = _iterative_split(temp_idx, label_matrix[temp_idx], test_size=0.50)

    splits: dict = {}
    for name, idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        splits[name] = Dataset.from_dict({
            "text":   [all_texts[i]   for i in idx],
            "labels": [all_labels[i]  for i in idx],
            "source": [all_sources[i] for i in idx],
        })
        sub = label_matrix[idx]
        prev = sub.mean(axis=0)
        logger.info(
            "%s: %d examples | prevalence: %s",
            name, len(idx),
            {lbl: f"{prev[j]:.3f}" for j, lbl in enumerate(EMOTION_LABELS)},
        )

    return DatasetDict(splits)
