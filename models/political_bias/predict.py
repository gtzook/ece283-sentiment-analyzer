"""
Interactive bias-detection inference.

Usage
-----
# Interactive REPL (paste headlines one at a time)
python infer.py

# Single headline from the command line
python infer.py --text "Left-wing activists demand sweeping reforms"

# Different checkpoint / dataset
python infer.py --checkpoint runs/10_BABE/label/best.pt --dataset 10_BABE
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast

logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.registry import REGISTRY, TaskType
from models.political_bias.model import RoBERTaClassifier

DEFAULT_CHECKPOINT = "runs/10_BABE/label/best.pt"
DEFAULT_DATASET    = "10_BABE"
DEFAULT_LABEL_COL  = "label"
DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_MAX_LENGTH = 128

LABEL_NAMES = {
    "10_BABE":          {0: "not biased",     1: "biased"},
    "72_LIAR":          {0: "true",            1: "false"},
    "03_CW_HARD":       {0: "mainstream",      1: "hyperpartisan"},
    "75_RedditBias":    {0: "unbiased",         1: "biased"},
    "80_DebateEffects": {0: "not persuasive",  1: "persuasive"},
    "9_BASIL":          {0: "lexical bias",    1: "informational bias", 2: "not biased"},
}


class BiasPredictor:
    """Loadable inference wrapper for a trained political-bias checkpoint.

    Usage:
        from infer import BiasPredictor
        predictor = BiasPredictor("runs/10_BABE/label/best.pt")
        result = predictor.predict("Left-wing activists demand sweeping reforms")
        # {"prediction": "biased", "confidence": 0.94, "probabilities": {...}}
    """

    def __init__(
        self,
        checkpoint: str | Path,
        dataset_id: str = DEFAULT_DATASET,
        label_col: str = DEFAULT_LABEL_COL,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.label_names = LABEL_NAMES.get(dataset_id, {})

        self.model, self.lc = load_model(
            checkpoint, dataset_id, label_col, model_name, self.device
        )
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    def predict(self, text: str) -> dict:
        return predict(
            text, self.model, self.tokenizer,
            self.lc, self.device, self.label_names, self.max_length,
        )

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


def load_model(checkpoint, dataset_id, label_col, model_name, device):
    meta  = REGISTRY[dataset_id]
    lc    = next(lc for lc in meta.label_columns if lc.col == label_col)
    model = RoBERTaClassifier(
        task_type=lc.task_type,
        num_classes=lc.num_classes,
        model_name=model_name,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, lc


@torch.no_grad()
def predict(text, model, tokenizer, lc, device, label_names, max_length=DEFAULT_MAX_LENGTH):
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

    if lc.task_type == TaskType.REGRESSION:
        return {"score": round(logits.item(), 4), "type": "regression"}

    probs      = F.softmax(logits, dim=-1).squeeze(0)
    pred_class = probs.argmax().item()
    return {
        "prediction": label_names.get(pred_class, str(pred_class)),
        "confidence": round(probs[pred_class].item(), 4),
        "probabilities": {
            label_names.get(i, str(i)): round(p.item(), 4)
            for i, p in enumerate(probs)
        },
    }


def print_result(text, result):
    bar_width = 30
    print(f"\n  Text        : {text}")
    if "score" in result:
        print(f"  Score       : {result['score']:.4f}")
        return
    print(f"  Prediction  : {result['prediction'].upper()}")
    print(f"  Confidence  : {result['confidence']:.1%}")
    print(f"  Breakdown   :")
    for label, prob in result["probabilities"].items():
        filled = round(prob * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        print(f"    {label:18s} {bar}  {prob:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Political bias inference")
    parser.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset",     default=DEFAULT_DATASET)
    parser.add_argument("--label-col",   default=DEFAULT_LABEL_COL)
    parser.add_argument("--model-name",  default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length",  type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--text",        default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, lc = load_model(
        args.checkpoint, args.dataset, args.label_col, args.model_name, device
    )
    tokenizer   = RobertaTokenizerFast.from_pretrained(args.model_name)
    label_names = LABEL_NAMES.get(args.dataset, {})

    print(f"Model ready  [{args.dataset} / {lc.task_type.value}]  device={device}\n")

    if args.text:
        result = predict(args.text, model, tokenizer, lc, device, label_names, args.max_length)
        print_result(args.text, result)
        return

    print("Paste a headline and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        result = predict(text, model, tokenizer, lc, device, label_names, args.max_length)
        print_result(text, result)
        print()


if __name__ == "__main__":
    main()
