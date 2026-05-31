"""
Improved RoBERTa training — two enhancements over train_baseline.py:

  1. Layerwise LR decay  (--lr-decay, default 0.9)
     Each transformer layer gets LR multiplied by decay^depth, so lower layers
     (general syntax) update more slowly than upper layers (task-specific).

  2. Auxiliary pre-fine-tuning  (--aux-dataset, default 25_FakeNewsNet)
     Before training on the target dataset, fine-tune for a few epochs on a
     larger news-domain dataset to stabilise the encoder. The classifier head
     is then re-initialised for the main task.

Usage examples
--------------
# Both improvements, BABE target
python train_improved.py --dataset 10_BABE

# Tune the decay multiplier
python train_improved.py --dataset 10_BABE --lr-decay 0.8

# Skip auxiliary pre-fine-tuning
python train_improved.py --dataset 10_BABE --no-aux

# Different auxiliary source
python train_improved.py --dataset 10_BABE --aux-dataset 03_CW_HARD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset import MAGPIEDataset
from src.data.splits import stratified_split
from src.data.registry import REGISTRY, TaskType
from src.models.roberta_classifier import RoBERTaClassifier

_DEFAULT_CACHE = "/mldata/ece283-sentiment-analyzer"
_DEFAULT_AUX   = "25_FakeNewsNet"   # 21k news headlines, same domain as BABE


# ── layerwise LR decay ────────────────────────────────────────────────────────

def build_param_groups(model, base_lr, lr_decay, weight_decay):
    """Return AdamW parameter groups with per-layer learning rates.

    Depth 0  = classifier head  → base_lr
    Depth 1  = encoder layer 11 → base_lr * decay
    ...
    Depth 13 = embeddings       → base_lr * decay^13
    """
    inner      = model._cls_model
    num_layers = inner.roberta.config.num_hidden_layers
    no_decay   = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    def _group(params_iter, lr):
        decay_p, nodecay_p = [], []
        for name, p in params_iter:
            (nodecay_p if any(nd in name for nd in no_decay) else decay_p).append(p)
        groups = []
        if decay_p:   groups.append({"params": decay_p,   "lr": lr, "weight_decay": weight_decay})
        if nodecay_p: groups.append({"params": nodecay_p, "lr": lr, "weight_decay": 0.0})
        return groups

    groups = _group(inner.classifier.named_parameters(), base_lr)
    for layer_idx in reversed(range(num_layers)):
        depth = num_layers - layer_idx
        groups += _group(inner.roberta.encoder.layer[layer_idx].named_parameters(),
                         base_lr * (lr_decay ** depth))
    groups += _group(inner.roberta.embeddings.named_parameters(),
                     base_lr * (lr_decay ** (num_layers + 1)))
    return groups


# ── metrics / collation ───────────────────────────────────────────────────────

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

def run_training(model, train_loader, val_loader, device, args, run_dir,
                 tag="", max_epochs=None, patience=None, base_lr=None):
    epochs  = max_epochs or args.epochs
    pat     = patience   or args.patience
    lr      = base_lr    or args.lr
    prefix  = f"[{tag}] " if tag else ""

    optimizer = torch.optim.AdamW(build_param_groups(model, lr, args.lr_decay, 0.01))
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_metric    = float("inf") if model.task_type == TaskType.REGRESSION else -1.0
    patience_count = 0
    history        = []

    for epoch in range(1, epochs + 1):
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
                print(f"  {prefix}epoch {epoch} step {step}/{len(train_loader)}  loss={loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        avg_train   = epoch_loss / len(train_loader)

        if model.task_type == TaskType.REGRESSION:
            primary  = val_metrics["mse"];   improved = primary < best_metric
        else:
            primary  = val_metrics["f1_macro"]; improved = primary > best_metric

        print(f"{prefix}Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}  "
              + "  ".join(f"val_{k}={v}" for k, v in val_metrics.items()))

        if improved:
            best_metric    = primary
            patience_count = 0
            if run_dir and not tag:
                ckpt = run_dir / "best.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"  ✓ saved checkpoint → {ckpt}")
        else:
            patience_count += 1
            if patience_count >= pat:
                print(f"{prefix}Early stopping at epoch {epoch}")
                break

        history.append({"epoch": epoch, "train_loss": avg_train, **val_metrics})

    return history


# ── auxiliary pre-fine-tuning ─────────────────────────────────────────────────

def pretrain_auxiliary(model, aux_id, tokenizer, device, args):
    if aux_id not in REGISTRY:
        print(f"WARNING: '{aux_id}' not in registry, skipping."); return

    aux_meta = REGISTRY[aux_id]
    aux_lc   = aux_meta.label_columns[0]
    if aux_lc.task_type == TaskType.SEQUENCE_LABEL:
        print(f"WARNING: '{aux_id}' is sequence-labeling, skipping."); return

    print(f"\n{'='*60}")
    print(f"Auxiliary pre-fine-tuning on {aux_id} ({aux_meta.name})")
    print(f"  task: {aux_lc.task_type.value}  classes: {aux_lc.num_classes}")
    print(f"{'='*60}")

    original_head = None
    if aux_lc.num_classes != model.num_classes:
        from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
        original_head = model._cls_model.classifier
        aux_cfg = type(model._cls_model.config)(**model._cls_model.config.to_dict())
        aux_cfg.num_labels = aux_lc.num_classes
        model._cls_model.classifier = RobertaClassificationHead(aux_cfg).to(device)

    aux_ds = MAGPIEDataset(
        dataset_id=aux_id, cache_dir=args.cache_dir, tokenizer=tokenizer,
        max_length=args.max_length, download_if_missing=True,
        label_col_filter=[aux_lc.col],
    )
    print(f"  {len(aux_ds):,} samples")

    train_ds, val_ds, _ = stratified_split(aux_ds, label_col=aux_lc.col, seed=args.seed)
    _col = collate_fn(aux_lc.col)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=_col, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.workers, collate_fn=_col, pin_memory=True)

    orig_task, orig_nc = model.task_type, model.num_classes
    model.task_type, model.num_classes = aux_lc.task_type, aux_lc.num_classes

    run_training(model, train_loader, val_loader, device, args,
                 run_dir=None, tag="aux", max_epochs=args.aux_epochs,
                 patience=args.aux_epochs, base_lr=args.lr)

    model.task_type, model.num_classes = orig_task, orig_nc

    if original_head is not None:
        model._cls_model.classifier = original_head
    else:
        for layer in model._cls_model.classifier.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    print(f"\nAuxiliary pre-training complete. Classifier head re-initialised.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RoBERTa with layerwise LR decay + auxiliary pre-fine-tuning"
    )
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
    parser.add_argument("--lr-decay",   type=float, default=0.9,
                        help="Per-layer LR multiplier (1.0=uniform, 0.9=10%% decay per layer)")
    parser.add_argument("--aux-dataset", default=_DEFAULT_AUX,
                        help="MAGPIE dataset ID for auxiliary pre-training (default: 25_FakeNewsNet)")
    parser.add_argument("--aux-epochs",  type=int, default=2)
    parser.add_argument("--no-aux",      action="store_true",
                        help="Skip auxiliary pre-fine-tuning")
    args = parser.parse_args()

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
    print(f"LR decay: {args.lr_decay}  (top={args.lr:.1e}, "
          f"bottom={args.lr * args.lr_decay**12:.2e}, emb={args.lr * args.lr_decay**13:.2e})")

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)

    full_ds = MAGPIEDataset(
        dataset_id=args.dataset, cache_dir=args.cache_dir, tokenizer=tokenizer,
        max_length=args.max_length, download_if_missing=True, label_col_filter=[label_col],
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
        task_type=lc_meta.task_type, num_classes=lc_meta.num_classes,
        model_name=args.model_name, dropout=args.dropout,
    ).to(device)

    run_dir = Path(args.run_dir or f"runs_improved/{args.dataset}/{label_col}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        ckpt = args.checkpoint or run_dir / "best.pt"
        print(f"Loading checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        if not args.no_aux:
            pretrain_auxiliary(model, args.aux_dataset, tokenizer, device, args)

        print(f"\n{'='*60}")
        print(f"Main training  lr={args.lr}  decay={args.lr_decay}  "
              f"batch={args.batch_size}  epochs={args.epochs}")
        print(f"Run dir: {run_dir}")
        print(f"{'='*60}")
        history = run_training(model, train_loader, val_loader, device, args, run_dir)

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        best_ckpt = run_dir / "best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    print(f"\n{'='*60}\nTest set evaluation\n{'='*60}")
    test_metrics = evaluate(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    out = {"dataset": args.dataset, "label_col": label_col,
           "lr_decay": args.lr_decay,
           "aux_dataset": args.aux_dataset if not args.no_aux else None,
           **test_metrics}
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {run_dir}/test_metrics.json")


if __name__ == "__main__":
    main()
