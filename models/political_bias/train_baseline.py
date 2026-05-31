"""
RoBERTa baseline for MAGPIE political-bias datasets.

Usage examples
--------------
# Fine-tune on BABE (binary news bias)
python train_baseline.py --dataset 10_BABE

# Fine-tune on LIAR binary label
python train_baseline.py --dataset 72_LIAR --label-col label_binary

# Evaluate only (skip training)
python train_baseline.py --dataset 10_BABE --eval-only --checkpoint runs/10_BABE/best.pt

Political-bias relevant dataset IDs
------------------------------------
  10_BABE          News sentence bias          (binary)
  72_LIAR          PolitiFact truthfulness     (binary / regression)
  03_CW_HARD       Hyperpartisan news          (binary)
  75_RedditBias    Reddit ideological bias     (binary)
  80_DebateEffects Political debate persuasion (binary)
  9_BASIL          News bias type              (3-class)
  19_MultiDimNews  Multi-dim news bias         (binary x4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.dataset import MAGPIEDataset
from src.data.splits import stratified_split
from src.data.registry import REGISTRY, TaskType
from models.political_bias.classifier import RoBERTaClassifier

_DEFAULT_CACHE = "/mldata/ece283-sentiment-analyzer"

POLITICAL_DATASETS = {
    "10_BABE", "72_LIAR", "03_CW_HARD",
    "75_RedditBias", "80_DebateEffects", "9_BASIL", "19_MultiDimNews",
}


# ── metrics ───────────────────────────────────────────────────────────────────

def _cls_metrics(preds, labels, num_classes):
    from collections import defaultdict
    acc = (preds == labels).mean()
    results = {"accuracy": float(acc)}
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, g in zip(preds, labels):
        if p == g: tp[g] += 1
        else:      fp[p] += 1; fn[g] += 1
    f1s = []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        results[f"f1_class{c}"] = round(f1, 4)
    results["f1_macro"] = round(float(np.mean(f1s)), 4)
    return results


def _reg_metrics(preds, labels):
    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    r   = float(np.corrcoef(preds, labels)[0, 1]) if preds.std() > 0 else 0.0
    return {"mse": round(mse, 4), "mae": round(mae, 4), "pearson_r": round(r, 4)}


# ── collation ─────────────────────────────────────────────────────────────────

def collate_fn(label_col):
    def _collate(batch):
        input_ids      = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        raw_labels     = [b["labels"][label_col] for b in batch]
        labels = torch.stack(raw_labels) if isinstance(raw_labels[0], torch.Tensor) \
                 else torch.tensor(raw_labels)
        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "token_type_ids" in batch[0]:
            out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
        return out
    return _collate


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        ttids = batch.get("token_type_ids")
        if ttids is not None: ttids = ttids.to(device)

        logits = model(ids, mask, ttids)
        total_loss += model.loss(logits, lbls).item() * len(lbls)

        if model.task_type == TaskType.REGRESSION:
            all_preds.extend(logits.cpu().numpy().tolist())
        else:
            all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        all_labels.extend(lbls.cpu().numpy().tolist())

    preds, labels = np.array(all_preds), np.array(all_labels)
    metrics = _reg_metrics(preds, labels) if model.task_type == TaskType.REGRESSION \
              else _cls_metrics(preds, labels.astype(int), model.num_classes)
    metrics["loss"] = round(total_loss / len(labels), 4)
    return metrics


# ── training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, device, args, run_dir):
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_metric    = float("inf") if model.task_type == TaskType.REGRESSION else -1.0
    patience_count = 0
    history        = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)
            ttids = batch.get("token_type_ids")
            if ttids is not None: ttids = ttids.to(device)

            optimizer.zero_grad()
            logits = model(ids, mask, ttids)
            loss   = model.loss(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % max(1, len(train_loader) // 4) == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}  loss={loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        avg_train   = epoch_loss / len(train_loader)

        if model.task_type == TaskType.REGRESSION:
            primary  = val_metrics["mse"]
            improved = primary < best_metric
        else:
            primary  = val_metrics["f1_macro"]
            improved = primary > best_metric

        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train:.4f}  "
              + "  ".join(f"val_{k}={v}" for k, v in val_metrics.items()))

        if improved:
            best_metric    = primary
            patience_count = 0
            ckpt = run_dir / "best.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ saved checkpoint → {ckpt}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        history.append({"epoch": epoch, "train_loss": avg_train, **val_metrics})

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoBERTa political-bias baseline")
    parser.add_argument("--dataset",    default="10_BABE")
    parser.add_argument("--label-col",  default=None)
    parser.add_argument("--cache-dir",  default=_DEFAULT_CACHE)
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--max-length", type=int,   default=128)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--patience",   type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--run-dir",    default=None)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--list-datasets", action="store_true")
    args = parser.parse_args()

    if args.list_datasets:
        print("Political-bias relevant MAGPIE datasets:")
        for did in sorted(POLITICAL_DATASETS):
            if did in REGISTRY:
                print(f"  {did:30s} {REGISTRY[did].name}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset not in REGISTRY:
        print(f"ERROR: '{args.dataset}' not in registry."); sys.exit(1)

    meta      = REGISTRY[args.dataset]
    label_col = args.label_col or meta.label_columns[0].col
    lc_meta   = next((lc for lc in meta.label_columns if lc.col == label_col), None)
    if lc_meta is None:
        print(f"ERROR: label column '{label_col}' not found."); sys.exit(1)

    print(f"\nDataset : {args.dataset} — {meta.name}")
    print(f"Task    : {lc_meta.task_type.value}  label='{label_col}'")

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)

    full_ds = MAGPIEDataset(
        dataset_id=args.dataset,
        cache_dir=args.cache_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        download_if_missing=True,
        label_col_filter=[label_col],
    )
    print(f"  {len(full_ds):,} samples")

    train_ds, val_ds, test_ds = stratified_split(full_ds, label_col=label_col, seed=args.seed)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    _col = collate_fn(label_col)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=_col, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.workers, collate_fn=_col, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.workers, collate_fn=_col, pin_memory=True)

    print(f"\nBuilding RoBERTa classifier ({args.model_name})")
    model = RoBERTaClassifier(
        task_type=lc_meta.task_type,
        num_classes=lc_meta.num_classes,
        model_name=args.model_name,
        dropout=args.dropout,
    ).to(device)

    run_dir = Path(args.run_dir or f"runs/{args.dataset}/{label_col}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        ckpt = args.checkpoint or run_dir / "best.pt"
        print(f"Loading checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        print(f"\n{'='*60}")
        print(f"Training  lr={args.lr}  batch={args.batch_size}  epochs={args.epochs}")
        print(f"Run dir: {run_dir}")
        print(f"{'='*60}")
        train(model, train_loader, val_loader, device, args, run_dir)
        best_ckpt = run_dir / "best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    print(f"\n{'='*60}\nTest set evaluation\n{'='*60}")
    test_metrics = evaluate(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    out_path = run_dir / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "label_col": label_col, **test_metrics}, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
