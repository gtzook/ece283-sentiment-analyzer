"""
predict.py — inference API for the epistemic certainty model.

Importable:
    from models.epistemic.predict import load_predictor, predict

    predictor = load_predictor("runs/20260530_071702/best.pt")
    result    = predict(predictor, "The drug may help some patients.")
    # result = {
    #   "label": 1,
    #   "label_name": "hedged",
    #   "uncertainty_score": 0.52,
    #   "sent_probs": [0.31, 0.55, 0.14],
    #   "cue_spans": [(9, 12)],   # char offsets of detected hedge cue tokens
    # }

CLI (single sentence):
    python -m models.epistemic.predict \\
        --checkpoint runs/20260530_071702/best.pt \\
        "The drug may help some patients."

CLI (batch — one sentence per line on stdin):
    echo -e "May be true.\\nDefinitely true." | \\
        python -m models.epistemic.predict --checkpoint runs/.../best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from models.epistemic.data import LABEL_NAMES
from models.epistemic.model import EpistemicModel

_DEFAULT_CONFIG = "models/epistemic/config.yaml"
_DEFAULT_CHECKPOINT_GLOB = "runs/*/best.pt"
_CUE_THRESHOLD = 0.5   # P(cue) threshold for reporting a token as a hedge cue


@dataclass
class Predictor:
    model:     EpistemicModel
    tokenizer: object
    device:    torch.device
    max_len:   int


def load_predictor(
    checkpoint: str | Path,
    config: str | Path = _DEFAULT_CONFIG,
    device: str | None = None,
) -> Predictor:
    """
    Load a trained EpistemicModel from a checkpoint.

    Args:
        checkpoint: path to .pt file
        config:     path to config.yaml (for model_name and max_len)
        device:     "cuda", "cpu", or None (auto-detect)
    """
    with open(config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    max_len    = cfg["data"].get("max_len", 128)

    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = EpistemicModel(model_name=model_name).to(dev)
    # strict=False: the checkpoint may contain sent_loss_fn.weight (class weights
    # registered as a buffer during training) which is not needed at inference.
    model.load_state_dict(
        torch.load(checkpoint, map_location=dev, weights_only=True),
        strict=False,
    )
    model.eval()

    return Predictor(model=model, tokenizer=tokenizer, device=dev, max_len=max_len)


def predict(predictor: Predictor, text: str) -> dict:
    """
    Run inference on a single sentence.

    Returns:
        label            int          0=asserted, 1=hedged, 2=speculative
        label_name       str
        uncertainty_score float       0.0 (certain) → 1.0 (speculative)
        sent_probs       list[float]  softmax over [asserted, hedged, speculative]
        cue_spans        list[tuple]  (char_start, char_end) for detected hedge cues
    """
    return predict_batch(predictor, [text])[0]


def predict_batch(predictor: Predictor, texts: list[str]) -> list[dict]:
    """
    Run inference on a list of sentences.  Returns one result dict per input.
    """
    enc = predictor.tokenizer(
        texts,
        max_length=predictor.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    input_ids      = enc["input_ids"].to(predictor.device)
    attention_mask = enc["attention_mask"].to(predictor.device)
    offset_mapping = enc["offset_mapping"]   # (N, L, 2) — stays on CPU

    with torch.no_grad():
        out = predictor.model.predict(input_ids, attention_mask)

    labels     = out["label"].cpu().tolist()
    scores     = out["uncertainty_score"].cpu().tolist()
    sent_probs = out["sent_probs"].cpu().tolist()
    tok_probs  = out["token_probs"].cpu()   # (N, L)

    results = []
    for i, text in enumerate(texts):
        # Find hedge cue character spans where P(cue) > threshold
        cue_spans = []
        for j in range(predictor.max_len):
            tok_start, tok_end = offset_mapping[i, j].tolist()
            if tok_start == 0 and tok_end == 0:
                continue    # special / padding token
            if tok_probs[i, j].item() >= _CUE_THRESHOLD:
                cue_spans.append((tok_start, tok_end))

        results.append({
            "label":             labels[i],
            "label_name":        LABEL_NAMES[labels[i]],
            "uncertainty_score": round(scores[i], 4),
            "sent_probs":        [round(p, 4) for p in sent_probs[i]],
            "cue_spans":         cue_spans,
        })
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint() -> Path | None:
    import glob
    paths = sorted(glob.glob(_DEFAULT_CHECKPOINT_GLOB))
    return Path(paths[-1]) if paths else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run epistemic certainty inference")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to .pt checkpoint (default: latest runs/*/best.pt)",
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="Path to config YAML",
    )
    parser.add_argument(
        "sentence", nargs="?", default=None,
        help="Sentence to classify. If omitted, reads one sentence per line from stdin.",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint or _find_latest_checkpoint()
    if ckpt is None:
        sys.exit("No checkpoint found. Pass --checkpoint or train a model first.")

    predictor = load_predictor(ckpt, config=args.config)

    if args.sentence:
        texts = [args.sentence]
    else:
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    results = predict_batch(predictor, texts)
    for text, result in zip(texts, results):
        print(json.dumps({"text": text, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
