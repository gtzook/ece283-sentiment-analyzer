"""
Evaluation utilities for political-bias models.

Usage:
    python -m models.political_bias.eval \\
        --checkpoint runs/10_BABE/label/best.pt \\
        --dataset 10_BABE --label-col label
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.dataset import MAGPIEDataset
from src.data.splits import stratified_split
from src.data.registry import REGISTRY, TaskType
from models.political_bias.model import RoBERTaClassifier
from models.political_bias.train_baseline import collate_fn

_DEFAULT_CACHE = "/mldata/ece283-sentiment-analyzer"


def _cls_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    acc = (preds == labels).mean()
    results = {"accuracy": float(acc)}
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, g in zip(preds, labels):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1s = []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        results[f"f1_class{c}"] = round(f1, 4)
    results["f1_macro"] = round(float(np.mean(f1s)), 4)
    return results


def _reg_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    r   = float(np.corrcoef(preds, labels)[0, 1]) if preds.std() > 0 else 0.0
    return {"mse": round(mse, 4), "mae": round(mae, 4), "pearson_r": round(r, 4)}


@torch.no_grad()
def evaluate(model: RoBERTaClassifier, loader: DataLoader, device: torch.device) -> dict:
    """Run inference over loader and return classification/regression metrics."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        ttids = batch.get("token_type_ids")
        if ttids is not None:
            ttids = ttids.to(device)

        logits = model(ids, mask, ttids)
        total_loss += model.loss(logits, lbls).item() * len(lbls)

        if model.task_type == TaskType.REGRESSION:
            all_preds.extend(logits.cpu().numpy().tolist())
        else:
            all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        all_labels.extend(lbls.cpu().numpy().tolist())

    preds, labels = np.array(all_preds), np.array(all_labels)
    if model.task_type == TaskType.REGRESSION:
        metrics = _reg_metrics(preds, labels)
    else:
        metrics = _cls_metrics(preds, labels.astype(int), model.num_classes)
    metrics["loss"] = round(total_loss / len(labels), 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a political-bias checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset",    default="10_BABE")
    parser.add_argument("--label-col",  default=None)
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-dir",  default=_DEFAULT_CACHE)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset not in REGISTRY:
        print(f"ERROR: '{args.dataset}' not in registry."); sys.exit(1)

    meta      = REGISTRY[args.dataset]
    label_col = args.label_col or meta.label_columns[0].col
    lc        = next((lc for lc in meta.label_columns if lc.col == label_col), None)
    if lc is None:
        print(f"ERROR: label column '{label_col}' not found."); sys.exit(1)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    full_ds   = MAGPIEDataset(
        dataset_id=args.dataset, cache_dir=args.cache_dir, tokenizer=tokenizer,
        max_length=args.max_length, download_if_missing=True, label_col_filter=[label_col],
    )
    train_ds, val_ds, test_ds = stratified_split(full_ds, label_col=label_col, seed=args.seed)
    split_ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    _col    = collate_fn(label_col)
    loader  = DataLoader(split_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, collate_fn=_col, pin_memory=True)

    model = RoBERTaClassifier(
        task_type=lc.task_type, num_classes=lc.num_classes, model_name=args.model_name,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    metrics = evaluate(model, loader, device)
    print(f"\n{args.split} metrics — {args.dataset}/{label_col}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out = {"dataset": args.dataset, "label_col": label_col, "split": args.split, **metrics}
    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
