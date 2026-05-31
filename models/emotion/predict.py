"""
Inference interface for the emotional framing floor model.

Usage (programmatic):
    from predict import EmotionalFramingPredictor

    predictor = EmotionalFramingPredictor("./checkpoints/emotional_framing_floor")
    results = predictor.predict(["I am so angry!", "What a joyful day."])

Usage (CLI):
    python predict.py --checkpoint ./checkpoints/emotional_framing_floor \
                      --texts "I am scared" "This is amazing"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch

from models.emotion.config import EMOTION_LABELS, EmotionalFramingConfig
from models.emotion.model import EmotionalFramingClassifier

from transformers import AutoTokenizer, RobertaConfig


class EmotionalFramingPredictor:
    """
    Wraps a saved checkpoint for single-text and batch inference.

    The tuned threshold is read from <checkpoint>/threshold.txt if present;
    otherwise falls back to cfg.threshold (0.5 default).
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        device: str | None = None,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)

        # Load threshold (tuned on dev during training, if available)
        threshold_path = checkpoint_dir / "threshold.txt"
        self.threshold = (
            float(threshold_path.read_text().strip())
            if threshold_path.exists()
            else EmotionalFramingConfig.threshold
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

        model_config = RobertaConfig.from_pretrained(str(checkpoint_dir))
        self.model = EmotionalFramingClassifier.from_pretrained(
            str(checkpoint_dir),
            config=model_config,
        )
        self.model.eval()
        self.model.to(self.device)

        self.max_length = EmotionalFramingConfig.max_seq_length

    @torch.inference_mode()
    def predict(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Classify a list of raw text strings for emotional framing.

        Args:
            texts:      List of input strings (news sentences or passages).
            batch_size: Inference batch size.

        Returns:
            List of dicts, one per input, with structure:
            {
                "anger": 0,        # 0 or 1 per emotion
                "anticipation": 1,
                ...
                "scores": {        # raw sigmoid probabilities
                    "anger": 0.12,
                    "anticipation": 0.87,
                    ...
                }
            }

        MTL HOOK: In the MTL model, predict() would accept a task_name argument
        and route pooled representations to the appropriate task head before
        applying the threshold. The tokenisation and encoding steps are identical.
        """
        all_results: list[dict] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start: batch_start + batch_size]

            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            output = self.model(**enc)
            logits: torch.Tensor = output.logits  # (B, 11)

            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 11)
            preds = (probs >= self.threshold).astype(int)  # (B, 11)

            for i in range(len(batch_texts)):
                result = {lbl: int(preds[i, j]) for j, lbl in enumerate(EMOTION_LABELS)}
                result["scores"] = {lbl: float(round(probs[i, j], 4))
                                    for j, lbl in enumerate(EMOTION_LABELS)}
                all_results.append(result)

        return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run emotional framing inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/emotional_framing_floor",
        help="Path to saved model checkpoint directory",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        required=True,
        help="One or more text strings to classify",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override sigmoid threshold (default: loaded from checkpoint)",
    )
    args = parser.parse_args()

    predictor = EmotionalFramingPredictor(args.checkpoint)
    if args.threshold is not None:
        predictor.threshold = args.threshold

    results = predictor.predict(args.texts)
    print(json.dumps(results, indent=2))
