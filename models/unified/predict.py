"""
Inference interface for the UnifiedModel.

A single shared encoder runs once per text; all three task heads execute in
a single forward pass, unlike the ensemble UnifiedSentimentAnalyzer in
src/unified_analyzer.py which runs three independent models sequentially.

Usage:
    from models.unified.predict import load_unified_predictor, predict_all

    predictor = load_unified_predictor(
        checkpoint="runs/unified/best.pt",
        config="models/unified/config.yaml",
    )
    results = predict_all(predictor, ["The drug may help some patients."])
    # [{"text": ..., "epistemic": {...}, "bias": {...}, "emotion": {...}}]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from transformers import RobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).parents[2]))
from models.unified.model import UnifiedModel, TASK_EPISTEMIC, TASK_BIAS, TASK_EMOTION
from src.data.registry import REGISTRY, TaskType


_EPISTEMIC_LABELS = ["asserted", "hedged", "speculative"]
_UNCERTAINTY_W    = torch.tensor([0.0, 0.5, 1.0])

_DEFAULT_BIAS_LABEL_NAMES = {
    "10_BABE":          {0: "not biased",    1: "biased"},
    "72_LIAR":          {0: "true",           1: "false"},
    "03_CW_HARD":       {0: "mainstream",     1: "hyperpartisan"},
    "75_RedditBias":    {0: "unbiased",        1: "biased"},
    "80_DebateEffects": {0: "not persuasive", 1: "persuasive"},
    "9_BASIL":          {0: "lexical bias",   1: "informational bias", 2: "not biased"},
}

_EMOTION_LABELS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]


@dataclass
class UnifiedPredictor:
    model:              UnifiedModel
    tokenizer:          object
    device:             torch.device
    max_len:            int
    emotion_thresholds: list = field(default_factory=lambda: [0.5] * 11)
    bias_dataset:       str = "10_BABE"
    bias_task_type:     TaskType = TaskType.BINARY_CLS
    bias_label_names:   dict = field(default_factory=dict)


def load_unified_predictor(
    checkpoint: str | Path,
    config: str | Path = "models/unified/config.yaml",
    device: Optional[str] = None,
    emotion_threshold: float = 0.5,
) -> UnifiedPredictor:
    """Load a trained UnifiedModel checkpoint and return an inference-ready predictor.

    If emotion_thresholds.json exists alongside the checkpoint it is loaded
    automatically; otherwise every class uses emotion_threshold (default 0.5).
    """
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model"]["name"])

    # Resolve bias dataset for label-name lookup; prefer explicit key, fall back to
    # first entry in bias_datasets list, then hard-coded default.
    bias_dataset = (cfg["model"].get("bias_dataset")
                    or (cfg["model"].get("bias_datasets") or [{}])[0].get("dataset_id", "10_BABE"))
    bias_lc_col  = (cfg["model"].get("bias_label_col")
                    or (cfg["model"].get("bias_datasets") or [{}])[0].get("label_col", "label"))

    if "bias_datasets" in cfg["model"]:
        bias_task_type   = TaskType.BINARY_CLS
        bias_num_classes = 2
    else:
        bias_lc          = next(lc for lc in REGISTRY[bias_dataset].label_columns
                                if lc.col == bias_lc_col)
        bias_task_type   = bias_lc.task_type
        bias_num_classes = bias_lc.num_classes

    model = UnifiedModel(
        model_name         = cfg["model"]["name"],
        dropout            = cfg["model"].get("dropout", 0.1),
        lambda_token       = cfg["model"].get("lambda_token", 0.3),
        bias_task_type     = bias_task_type,
        bias_num_classes   = bias_num_classes,
        emotion_num_labels = cfg["model"].get("emotion_num_labels", 11),
    ).to(dev)
    model.load_state_dict(
        torch.load(checkpoint, map_location=dev, weights_only=True),
        strict=False,
    )
    model.eval()

    # Load per-class thresholds if available, otherwise fall back to fixed threshold
    import json as _json
    thresholds_path = Path(checkpoint).parent / "emotion_thresholds.json"
    if thresholds_path.exists():
        emotion_thresholds = _json.loads(thresholds_path.read_text())["thresholds"]
    else:
        num_labels = cfg["model"].get("emotion_num_labels", 11)
        emotion_thresholds = [emotion_threshold] * num_labels

    return UnifiedPredictor(
        model              = model,
        tokenizer          = tokenizer,
        device             = dev,
        max_len            = cfg["data"]["max_len"],
        emotion_thresholds = emotion_thresholds,
        bias_dataset       = bias_dataset,
        bias_task_type     = bias_task_type,
        bias_label_names   = _DEFAULT_BIAS_LABEL_NAMES.get(bias_dataset, {}),
    )


def _encode(predictor: UnifiedPredictor, texts: list[str]) -> dict:
    enc = predictor.tokenizer(
        texts,
        max_length  = predictor.max_len,
        padding     = "max_length",
        truncation  = True,
        return_tensors = "pt",
    )
    return {k: v.to(predictor.device) for k, v in enc.items()}


@torch.no_grad()
def predict_epistemic(predictor: UnifiedPredictor, texts: list[str]) -> list[dict]:
    """Run only the epistemic head."""
    enc = _encode(predictor, texts)
    out = predictor.model(task=TASK_EPISTEMIC, **enc)

    sent_probs = torch.softmax(out["sent_logits"], dim=-1).cpu()
    weights    = _UNCERTAINTY_W.to(sent_probs.device)
    uncertainty = (sent_probs * weights).sum(dim=-1)

    results = []
    for i in range(len(texts)):
        probs = sent_probs[i]
        label = int(probs.argmax())
        results.append({
            "label":            label,
            "label_name":       _EPISTEMIC_LABELS[label],
            "uncertainty_score": round(uncertainty[i].item(), 4),
            "sent_probs":       [round(p, 4) for p in probs.tolist()],
        })
    return results


@torch.no_grad()
def predict_bias(predictor: UnifiedPredictor, texts: list[str]) -> list[dict]:
    """Run only the bias head."""
    enc = _encode(predictor, texts)
    out = predictor.model(task=TASK_BIAS, **enc)

    if predictor.bias_task_type == TaskType.REGRESSION:
        return [{"score": round(out["bias_logits"][i].item(), 4), "type": "regression"}
                for i in range(len(texts))]

    probs = F.softmax(out["bias_logits"], dim=-1).cpu()
    results = []
    for i in range(len(texts)):
        p      = probs[i]
        label  = int(p.argmax())
        results.append({
            "prediction":    predictor.bias_label_names.get(label, str(label)),
            "confidence":    round(p[label].item(), 4),
            "probabilities": {
                predictor.bias_label_names.get(j, str(j)): round(pj.item(), 4)
                for j, pj in enumerate(p)
            },
        })
    return results


@torch.no_grad()
def predict_emotion(predictor: UnifiedPredictor, texts: list[str]) -> list[dict]:
    """Run only the emotion head, applying per-class thresholds."""
    enc   = _encode(predictor, texts)
    out   = predictor.model(task=TASK_EMOTION, **enc)
    probs = torch.sigmoid(out["emotion_logits"]).cpu().numpy()
    thrs  = np.array(predictor.emotion_thresholds)  # (11,)
    preds = (probs >= thrs).astype(int)              # broadcast over batch dim

    results = []
    for i in range(len(texts)):
        result = {lbl: int(preds[i, j]) for j, lbl in enumerate(_EMOTION_LABELS)}
        result["scores"] = {lbl: round(float(probs[i, j]), 4)
                            for j, lbl in enumerate(_EMOTION_LABELS)}
        results.append(result)
    return results


@torch.no_grad()
def predict_all(predictor: UnifiedPredictor, texts: list[str]) -> list[dict]:
    """
    Run all three heads in a single shared-encoder forward pass per task.
    Returns one dict per text with keys: text, epistemic, bias, emotion.
    """
    ep_results   = predict_epistemic(predictor, texts)
    bias_results = predict_bias(predictor, texts)
    em_results   = predict_emotion(predictor, texts)

    return [
        {
            "text":      text,
            "epistemic": ep,
            "bias":      bias,
            "emotion":   em,
        }
        for text, ep, bias, em in zip(texts, ep_results, bias_results, em_results)
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run unified model inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="models/unified/config.yaml")
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--task",  choices=["all", "epistemic", "bias", "emotion"], default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictor = load_unified_predictor(args.checkpoint, args.config,
                                       emotion_threshold=args.threshold)

    dispatch = {
        "all":       predict_all,
        "epistemic": predict_epistemic,
        "bias":      predict_bias,
        "emotion":   predict_emotion,
    }
    results = dispatch[args.task](predictor, args.texts)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
