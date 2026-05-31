from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# SemEval-2018 Task 1 E-c label schema (alphabetical order, 11 labels)
EMOTION_LABELS: List[str] = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]

# MAGPIE 84_emotion_tweets: 0-indexed raw label → SemEval index in EMOTION_LABELS
# Plutchik 8 emotions in alphabetical order (preprocessed.csv is already 0-indexed)
MAGPIE_LABEL_TO_SEMEVAL_IDX: dict[int, int] = {
    0: 0,   # anger      → anger
    1: 1,   # anticipation → anticipation
    2: 2,   # disgust    → disgust
    3: 3,   # fear       → fear
    4: 4,   # joy        → joy
    5: 8,   # sadness    → sadness
    6: 9,   # surprise   → surprise
    7: 10,  # trust      → trust
    # love(5), optimism(6), pessimism(7) absent in MAGPIE — set to 0
}


@dataclass
class EmotionalFramingConfig:
    # ── Labels ────────────────────────────────────────────────────────────────
    labels: List[str] = field(default_factory=lambda: list(EMOTION_LABELS))
    num_labels: int = 11

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "roberta-base"

    # ── Optimizer — separate LRs for encoder vs. classification head ──────────
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01

    # ── Scheduler ─────────────────────────────────────────────────────────────
    warmup_ratio: float = 0.06   # fraction of total steps used for linear warmup

    # ── Training loop ─────────────────────────────────────────────────────────
    batch_size: int = 32
    max_epochs: int = 10
    patience: int = 3            # early stopping on dev macro-F1
    max_seq_length: int = 128
    threshold: float = 0.5       # per-label sigmoid threshold at inference (tune on dev)

    # ── I/O ───────────────────────────────────────────────────────────────────
    output_dir: str = "./checkpoints/emotional_framing_floor"
    magpie_data_dir: str = "/mldata/ece283-sentiment-analyzer"
    hf_cache_dir: str = "/mldata/ece283-sentiment-analyzer/hf_cache"

    # ── Logging ───────────────────────────────────────────────────────────────
    wandb_project: str = "emotional-framing-floor"
    logging_steps: int = 50

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    fp16: bool = True            # overridden to False at runtime if no CUDA

    # ── Debug mode — 100 samples for fast iteration ───────────────────────────
    debug: bool = False
    debug_samples: int = 100

    # ── Fast mode — distilroberta + short seqs + capped data ─────────────────
    # Activated by --fast flag. Cuts runtime from ~4 h to ~20 min.
    # distilroberta-base: 6 layers vs 12, same hidden size (768), ~2× faster.
    max_samples: int | None = None   # None = use all data
