"""
Metrics computation for multi-label emotion classification.

compute_metrics()  — passed to HuggingFace Trainer (uses logits from eval loop)
full_report()      — called at end of training for a detailed per-label breakdown
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    accuracy_score,
)
from transformers import EvalPrediction

from models.emotion.config import EMOTION_LABELS


def _apply_threshold(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid + threshold → binary predictions."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= threshold).astype(np.int32)


def make_compute_metrics(threshold: float = 0.5):
    """
    Factory that captures the inference threshold.

    Returns a compute_metrics function compatible with HuggingFace Trainer.
    Primary metric for early stopping: eval_macro_f1.
    """
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, label_ids = eval_pred
        preds = _apply_threshold(logits, threshold)

        macro_f1 = f1_score(label_ids, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(label_ids, preds, average="micro", zero_division=0)
        subset_acc = accuracy_score(label_ids, preds)
        h_loss = hamming_loss(label_ids, preds)

        return {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "subset_accuracy": subset_acc,
            "hamming_loss": h_loss,
        }

    return compute_metrics


def full_report(
    logits: np.ndarray,
    label_ids: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "test",
) -> None:
    """
    Print a comprehensive evaluation report to stdout.

    Includes per-label F1/Precision/Recall and aggregate metrics.
    """
    preds = _apply_threshold(logits, threshold)

    print(f"\n{'='*60}")
    print(f"  Evaluation report — {split_name} set")
    print(f"{'='*60}")

    # Per-label breakdown
    print(classification_report(
        label_ids,
        preds,
        target_names=EMOTION_LABELS,
        digits=4,
        zero_division=0,
    ))

    # Aggregate metrics
    macro_f1 = f1_score(label_ids, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(label_ids, preds, average="micro", zero_division=0)
    macro_p = precision_score(label_ids, preds, average="macro", zero_division=0)
    macro_r = recall_score(label_ids, preds, average="macro", zero_division=0)
    subset_acc = accuracy_score(label_ids, preds)
    h_loss = hamming_loss(label_ids, preds)

    print(f"Macro-F1       : {macro_f1:.4f}  (primary metric)")
    print(f"Micro-F1       : {micro_f1:.4f}")
    print(f"Macro-Precision: {macro_p:.4f}")
    print(f"Macro-Recall   : {macro_r:.4f}")
    print(f"Subset Accuracy: {subset_acc:.4f}  (exact match)")
    print(f"Hamming Loss   : {h_loss:.4f}")
    print(f"{'='*60}\n")


def tune_threshold(
    logits: np.ndarray,
    label_ids: np.ndarray,
    candidates: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> float:
    """
    Grid-search over threshold candidates; return the one maximising dev macro-F1.
    Call this on the dev set before final test evaluation.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        preds = _apply_threshold(logits, t)
        f1 = f1_score(label_ids, preds, average="macro", zero_division=0)
        print(f"  threshold={t:.2f}  macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  → best threshold: {best_t:.2f}  (macro-F1={best_f1:.4f})")
    return best_t
