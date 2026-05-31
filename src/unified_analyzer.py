"""
UnifiedSentimentAnalyzer — ensemble inference across all three task models.

Each model is optional; pass None to skip a checkpoint and that task's
fields will be None in the result.

Usage:
    from src.models.unified_analyzer import UnifiedSentimentAnalyzer

    analyzer = UnifiedSentimentAnalyzer(
        epistemic_checkpoint="runs/epistemic/best.pt",
        bias_checkpoint="runs/10_BABE/label/best.pt",
        emotion_checkpoint="checkpoints/emotional_framing_floor",
    )

    result = analyzer.analyze("The policy may reduce emissions slightly.")
    print(result.epistemic_label)   # "hedged"
    print(result.bias_label)        # "not biased"
    print(result.emotions)          # {"anger": 0, "optimism": 1, ...}
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path when imported as a module
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class UnifiedResult:
    text: str

    # ── Epistemic certainty (models/epistemic/) ───────────────────────────────
    # label: 0=asserted, 1=hedged, 2=speculative
    epistemic_label: Optional[str] = None          # "asserted"|"hedged"|"speculative"
    epistemic_uncertainty: Optional[float] = None  # 0.0 (certain) → 1.0 (speculative)
    epistemic_probs: Optional[list[float]] = None  # [p_asserted, p_hedged, p_speculative]
    hedge_cue_spans: list[tuple[int, int]] = field(default_factory=list)

    # ── Political bias (infer.py / src/models/roberta_classifier.py) ─────────
    bias_label: Optional[str] = None               # "biased"|"not biased" (or task-specific)
    bias_confidence: Optional[float] = None
    bias_probabilities: Optional[dict[str, float]] = None

    # ── Emotional framing (emotional_framing/) ────────────────────────────────
    emotions: Optional[dict[str, int]] = None      # {anger: 0|1, joy: 0|1, ...}
    emotion_scores: Optional[dict[str, float]] = None


class UnifiedSentimentAnalyzer:
    """
    Loads up to three trained checkpoints and runs them on shared input.

    Each model runs independently on the same text; there is no shared
    encoder at this stage (each checkpoint carries its own fine-tuned
    RoBERTa weights).
    """

    def __init__(
        self,
        epistemic_checkpoint: Optional[str | Path] = None,
        bias_checkpoint: Optional[str | Path] = None,
        bias_dataset_id: str = "10_BABE",
        bias_label_col: str = "label",
        emotion_checkpoint: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._epistemic = None
        if epistemic_checkpoint is not None:
            from models.epistemic.predict import load_predictor
            self._epistemic = load_predictor(str(epistemic_checkpoint), device=device)

        self._bias = None
        if bias_checkpoint is not None:
            from models.political_bias.predict import BiasPredictor
            self._bias = BiasPredictor(
                checkpoint=bias_checkpoint,
                dataset_id=bias_dataset_id,
                label_col=bias_label_col,
                device=device,
            )

        self._emotion = None
        if emotion_checkpoint is not None:
            from models.emotion.predict import EmotionalFramingPredictor
            self._emotion = EmotionalFramingPredictor(
                checkpoint_dir=emotion_checkpoint,
                device=device,
            )

    def analyze(self, text: str) -> UnifiedResult:
        result = UnifiedResult(text=text)

        if self._epistemic is not None:
            from models.epistemic.predict import predict as ep_predict
            ep = ep_predict(self._epistemic, text)
            result.epistemic_label = ep["label_name"]
            result.epistemic_uncertainty = ep["uncertainty_score"]
            result.epistemic_probs = ep.get("sent_probs")
            result.hedge_cue_spans = ep.get("cue_spans", [])

        if self._bias is not None:
            bi = self._bias.predict(text)
            if "score" in bi:
                result.bias_label = "regression"
                result.bias_confidence = bi["score"]
            else:
                result.bias_label = bi["prediction"]
                result.bias_confidence = bi["confidence"]
                result.bias_probabilities = bi.get("probabilities")

        if self._emotion is not None:
            em = self._emotion.predict([text])[0]
            result.emotion_scores = em.pop("scores", {})
            result.emotions = em

        return result

    def analyze_batch(self, texts: list[str]) -> list[UnifiedResult]:
        return [self.analyze(t) for t in texts]
