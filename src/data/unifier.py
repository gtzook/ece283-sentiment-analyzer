"""
Label unification: mapping heterogeneous MAGPIE labels to a shared sentiment space.

The output space is a 3-simplex: (neg_score, neu_score, pos_score) with values in [0, 1]
summing to 1. This is treated as a soft label for a 3-class sentiment classifier
(Negative / Neutral / Positive).

Usage
-----
from src.data.unifier import SENTIMENT_PROJECTORS, project_label

scores = project_label("10_BABE", "label", canonical_value=1)
# scores -> SentimentScores(neg=0.8, neu=0.2, pos=0.0)

# Convert to hard label:
hard = scores.argmax()  # 0=neg, 1=neu, 2=pos
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class SentimentScores:
    neg: float
    neu: float
    pos: float

    def __post_init__(self) -> None:
        total = self.neg + self.neu + self.pos
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scores must sum to 1.0, got {total}")

    def argmax(self) -> int:
        """0=negative, 1=neutral, 2=positive."""
        return max(range(3), key=lambda i: [self.neg, self.neu, self.pos][i])

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.neg, self.neu, self.pos)


# Type alias: canonical label int → SentimentScores
Projector = Callable[[int | float], SentimentScores]

NEG = SentimentScores(neg=1.0, neu=0.0, pos=0.0)
NEU = SentimentScores(neg=0.0, neu=1.0, pos=0.0)
POS = SentimentScores(neg=0.0, neu=0.0, pos=1.0)
SOFT_NEG = SentimentScores(neg=0.7, neu=0.3, pos=0.0)
SOFT_POS = SentimentScores(neg=0.0, neu=0.3, pos=0.7)


# ---------------------------------------------------------------------------
# Dataset-specific projectors
# (keyed by (dataset_id, label_col) for precision)
# ---------------------------------------------------------------------------

SENTIMENT_PROJECTORS: dict[tuple[str, str], Projector] = {}


def _proj(dataset_id: str, col: str) -> Callable[[Projector], Projector]:
    def decorator(fn: Projector) -> Projector:
        SENTIMENT_PROJECTORS[(dataset_id, col)] = fn
        return fn
    return decorator


# --- SST2: 0=positive, 1=negative -------------------------------------------
@_proj("99_SST2", "label")
def _sst2(v: int | float) -> SentimentScores:
    return POS if int(v) == 0 else NEG


# --- SemEval2014: 0=neg, 1=neu, 2=pos (after label_map remapping) ----------
@_proj("63_semeval2014", "label")
def _semeval14(v: int | float) -> SentimentScores:
    return [NEG, NEU, POS][int(v)]


# --- IMDB: 0=negative, 1=positive -------------------------------------------
@_proj("101_IMDB", "label")
def _imdb(v: int | float) -> SentimentScores:
    return NEG if int(v) == 0 else POS


# --- Amazon reviews: regression 0.0–1.0 (low=bad, high=good) ----------------
@_proj("100_Amazon_reviews", "label")
def _amazon(v: int | float) -> SentimentScores:
    s = float(v)  # already normalised 0–1
    pos = max(0.0, s - 0.5) * 2
    neg = max(0.0, 0.5 - s) * 2
    neu = 1.0 - pos - neg
    return SentimentScores(neg=round(neg, 4), neu=round(neu, 4), pos=round(pos, 4))


# --- BABE: 0=not biased, 1=biased -------------------------------------------
# Bias ≠ sentiment, but biased news is more likely to carry negative framing.
# We project to neutral-negative space rather than pos-neg to reflect that
# unbiased text is not inherently "positive".
@_proj("10_BABE", "label")
def _babe(v: int | float) -> SentimentScores:
    return SOFT_NEG if int(v) == 1 else NEU


# --- FakeNewsNet: 0=real, 1=fake --------------------------------------------
@_proj("25_FakeNewsNet", "label")
def _fakenews(v: int | float) -> SentimentScores:
    return NEG if int(v) == 1 else NEU


# --- LIAR regression: 0.0=true → 1.0=pants-fire ----------------------------
@_proj("72_LIAR", "label")
def _liar_reg(v: int | float) -> SentimentScores:
    falseness = float(v)  # 0=true, 1=false
    neg = round(falseness, 4)
    neu = round(max(0.0, 1.0 - 2 * abs(falseness - 0.5)), 4)
    pos = round(max(0.0, 1.0 - falseness - neu), 4)
    # renormalise for floating-point safety
    total = neg + neu + pos
    return SentimentScores(neg=neg / total, neu=neu / total, pos=pos / total)


@_proj("72_LIAR", "label_binary")
def _liar_bin(v: int | float) -> SentimentScores:
    return NEG if int(v) == 1 else NEU


# --- Emotion tweets: 8-class Plutchik (0-indexed after label_map) -----------
# Valence mapping: joy/trust/anticipation → positive; anger/disgust/sadness/fear → negative; surprise → neutral
_EMOTION_VALENCE: dict[int, SentimentScores] = {
    0: NEG,       # anger
    1: SOFT_POS,  # anticipation
    2: NEG,       # disgust
    3: NEG,       # fear
    4: POS,       # joy
    5: NEG,       # sadness
    6: NEU,       # surprise
    7: POS,       # trust
}


@_proj("84_emotion_tweets", "label")
def _emotions(v: int | float) -> SentimentScores:
    return _EMOTION_VALENCE[int(v)]


# --- PHEME veracity: 0=false, 1=true, 2=unknown -----------------------------
@_proj("12_PHEME", "veracity_label")
def _pheme_veracity(v: int | float) -> SentimentScores:
    return [NEG, POS, NEU][int(v)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project_label(
    dataset_id: str,
    col: str,
    canonical_value: int | float,
) -> SentimentScores | None:
    """Map a canonical label to (neg, neu, pos) scores.

    Returns None if no projector is registered for this (dataset_id, col) pair,
    meaning the label cannot be meaningfully mapped to a sentiment axis.
    """
    projector = SENTIMENT_PROJECTORS.get((dataset_id, col))
    if projector is None:
        return None
    return projector(canonical_value)


def has_projector(dataset_id: str, col: str) -> bool:
    return (dataset_id, col) in SENTIMENT_PROJECTORS


def projectable_datasets() -> list[tuple[str, str]]:
    """Return all (dataset_id, col) pairs that can be projected to sentiment."""
    return list(SENTIMENT_PROJECTORS.keys())
