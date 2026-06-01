"""
Multi-task training loop for the UnifiedModel.

Trains a single shared RoBERTa encoder jointly on epistemic, political-bias,
and emotional-framing datasets. Each training step processes one batch from one
task; tasks are round-robined so all data is seen each epoch.

Usage:
    python -m models.unified.train --config models/unified/config.yaml

    # Dry-run (2 batches per task, no checkpoint)
    python -m models.unified.train --config models/unified/config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

try:
    import wandb as _wandb_module
except ImportError:
    _wandb_module = None

sys.path.insert(0, str(Path(__file__).parents[2]))
from models.unified.model import UnifiedModel, TASK_EPISTEMIC, TASK_BIAS, TASK_EMOTION
from models.epistemic.data import SentDataset, TokenDataset, load_szeged_wiki, load_szeged_factbank
from models.epistemic.train import load_all_data, compute_sent_class_weights, doc_level_split
from models.epistemic.eval import compute_ece
from src.data.dataset import MAGPIEDataset
from src.data.splits import stratified_split
from src.data.registry import REGISTRY, TaskType
from models.political_bias.train_baseline import collate_fn as bias_collate_fn, _cls_metrics, _reg_metrics

logger = logging.getLogger(__name__)


# ── Emotion PyTorch adapter ───────────────────────────────────────────────────

class EmotionTorchDataset(Dataset):
    """Wraps a HuggingFace Dataset of pre-tokenized emotion examples."""

    def __init__(self, hf_dataset) -> None:
        self._ds = hf_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        ex = self._ds[idx]
        def _to(t, dtype):
            return t.detach().clone().to(dtype) if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=dtype)
        return {
            "input_ids":      _to(ex["input_ids"],      torch.long),
            "attention_mask": _to(ex["attention_mask"], torch.long),
            "emotion_labels": _to(ex["labels"],         torch.float32),
        }


def _emotion_collate(batch: list[dict]) -> dict:
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "emotion_labels": torch.stack([b["emotion_labels"] for b in batch]),
    }


# ── Multi-task batch iterator ─────────────────────────────────────────────────

class MultiTaskBatchIterator:
    """
    Yields (task_name, batch) tuples by round-robining three DataLoaders.

    The loader with the most batches determines one epoch's length; shorter
    loaders restart from the beginning when exhausted.
    """

    def __init__(self, loaders: dict[str, DataLoader]) -> None:
        self._loaders    = loaders
        self.epoch_steps = max(len(v) for v in loaders.values())
        self.task_keys   = list(loaders.keys())

    def __iter__(self):
        iters = {k: iter(v) for k, v in self._loaders.items()}
        for _ in range(self.epoch_steps):
            for task in self.task_keys:
                try:
                    batch = next(iters[task])
                except StopIteration:
                    iters[task] = iter(self._loaders[task])
                    batch = next(iters[task])
                yield task, batch


# ── Optimiser ─────────────────────────────────────────────────────────────────

def build_optimizer(
    model: UnifiedModel,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    no_decay = {"bias", "LayerNorm.weight"}

    encoder_decay   = [p for n, p in model.encoder.named_parameters()
                       if not any(nd in n for nd in no_decay)]
    encoder_nodecay = [p for n, p in model.encoder.named_parameters()
                       if any(nd in n for nd in no_decay)]
    head_params     = (
        list(model.sent_head.parameters())
        + list(model.token_head.parameters())
        + list(model.bias_head.parameters())
        + list(model.emotion_head.parameters())
    )
    return torch.optim.AdamW([
        {"params": encoder_decay,   "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": encoder_nodecay, "lr": encoder_lr, "weight_decay": 0.0},
        {"params": head_params,     "lr": head_lr,    "weight_decay": weight_decay},
    ])


# ── Data loading ──────────────────────────────────────────────────────────────

def _epistemic_sent_sampler(examples) -> torch.utils.data.WeightedRandomSampler:
    """
    Inverse-frequency sampler so each batch sees minority classes
    (hedged, speculative) at roughly equal frequency to the majority (asserted),
    rather than at their natural ~2% / <1% rates.
    """
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    labels       = [ex.label for ex in examples]
    class_counts = Counter(labels)
    total        = len(labels)
    weights      = [total / class_counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=total, replacement=True)


def _load_epistemic_loaders(cfg: dict, tokenizer, batch_size: int, workers: int) -> dict:
    """Return {'sent': DataLoader, 'tok': DataLoader} for the epistemic task."""
    ep_data = load_all_data(cfg)
    sent_class_weights = compute_sent_class_weights(ep_data["sent_train"])

    sent_train = SentDataset(ep_data["sent_train"], tokenizer, max_len=cfg["data"]["max_len"])
    tok_train  = TokenDataset(
        ep_data["tok_news_train"] + ep_data.get("tok_bio_train", []),
        tokenizer,
        max_len=cfg["data"]["max_len"],
    )

    # Balanced sampler: over-samples hedged/speculative so every batch has
    # a representative mix rather than ~85% asserted examples.
    sent_sampler = _epistemic_sent_sampler(ep_data["sent_train"])

    return {
        "sent": DataLoader(sent_train, batch_size=batch_size, shuffle=True,
                           num_workers=workers, persistent_workers=workers > 0),
        "tok":  DataLoader(tok_train,  batch_size=batch_size, shuffle=True,
                           num_workers=workers, persistent_workers=workers > 0),
        "sent_class_weights": sent_class_weights,
    }


def _load_bias_loaders(cfg: dict, tokenizer, batch_size: int, workers: int) -> dict:
    """Return {'train': DataLoader} for the political bias task."""
    dataset_id = cfg["model"].get("bias_dataset", "10_BABE")
    label_col  = cfg["model"].get("bias_label_col", "label")

    full_ds = MAGPIEDataset(
        dataset_id=dataset_id,
        cache_dir=cfg["data"]["cache_dir"],
        tokenizer=tokenizer,
        max_length=cfg["data"]["max_len"],
        download_if_missing=True,
        label_col_filter=[label_col],
    )
    train_ds, _, _ = stratified_split(full_ds, label_col=label_col, seed=cfg["data"]["seed"])
    _col = bias_collate_fn(label_col)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=workers, collate_fn=_col, pin_memory=True,
                            persistent_workers=workers > 0),
    }


def _load_emotion_loaders(cfg: dict, tokenizer, batch_size: int, workers: int) -> dict:
    """Return {'train': DataLoader} for the emotion task."""
    from models.emotion.config import EmotionalFramingConfig
    from models.emotion.data import load_and_split

    em_cfg = EmotionalFramingConfig()
    em_cfg.magpie_data_dir = cfg["data"].get("magpie_data_dir", em_cfg.magpie_data_dir)
    em_cfg.hf_cache_dir    = cfg["data"].get("hf_cache_dir", em_cfg.hf_cache_dir)
    em_cfg.max_seq_length  = cfg["data"]["max_len"]
    em_cfg.seed            = cfg["data"]["seed"]

    dataset_dict = load_and_split(em_cfg)

    def _tokenize(batch):
        enc = tokenizer(
            batch["text"],
            max_length=em_cfg.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        enc["labels"] = [list(map(float, lv)) for lv in batch["labels"]]
        return enc

    tokenized = dataset_dict.map(_tokenize, batched=True, remove_columns=["text", "source"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_torch = EmotionTorchDataset(tokenized["train"])
    return {
        "train": DataLoader(train_torch, batch_size=batch_size, shuffle=True,
                            num_workers=workers, collate_fn=_emotion_collate, pin_memory=True,
                            persistent_workers=workers > 0),
        "dev": EmotionTorchDataset(tokenized["dev"]),
        "test": EmotionTorchDataset(tokenized["test"]),
    }


# ── Batch → model kwargs ──────────────────────────────────────────────────────

def _batch_to_kwargs(task: str, batch: dict, device: torch.device) -> dict:
    """Convert a raw DataLoader batch to UnifiedModel.forward() kwargs."""
    kwargs = {
        "input_ids":      batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "task":           task,
    }
    if task == TASK_EPISTEMIC:
        if "sent_label" in batch:
            kwargs["sent_label"] = batch["sent_label"].to(device)
        if "token_labels" in batch:
            kwargs["token_labels"] = batch["token_labels"].to(device)
    elif task == TASK_BIAS:
        if "labels" in batch:
            kwargs["bias_label"] = batch["labels"].to(device)
    elif task == TASK_EMOTION:
        if "emotion_labels" in batch:
            kwargs["emotion_labels"] = batch["emotion_labels"].to(device)
    return kwargs


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: UnifiedModel,
    ep_data: dict,
    bias_val_loader: DataLoader,
    em_val_dataset,
    tokenizer,
    cfg: dict,
    device: torch.device,
    task_weights: dict,
) -> dict:
    model.eval()
    results = {}

    # Epistemic
    ep_cfg = cfg.copy()
    sent_val = SentDataset(ep_data["sent_val"], tokenizer, max_len=cfg["data"]["max_len"])
    val_loader = DataLoader(sent_val, batch_size=32, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    for batch in val_loader:
        kwargs = _batch_to_kwargs(TASK_EPISTEMIC, batch, device)
        out = model(**kwargs)
        probs = torch.softmax(out["sent_logits"], dim=-1).cpu()
        preds = probs.argmax(-1).numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["sent_label"].numpy().tolist())
        all_probs.append(probs.numpy())
    preds_a  = np.array(all_preds)
    labels_a = np.array(all_labels)
    from collections import defaultdict
    from models.epistemic.eval import compute_ece
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, g in zip(preds_a, labels_a):
        if p == g: tp[g] += 1
        else:      fp[p] += 1; fn[g] += 1
    f1s = []
    for c in range(3):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    results["epistemic_macro_f1"] = float(np.mean(f1s))

    # Bias
    bias_preds, bias_labels = [], []
    for batch in bias_val_loader:
        kwargs = _batch_to_kwargs(TASK_BIAS, batch, device)
        out = model(**kwargs)
        bias_preds.extend(out["bias_logits"].argmax(-1).cpu().numpy().tolist())
        bias_labels.extend(batch["labels"].numpy().tolist())
    bp = np.array(bias_preds)
    bl = np.array(bias_labels)
    from models.political_bias.train_baseline import _cls_metrics as _bm
    bias_m = _bm(bp, bl, model.bias_head.num_classes)
    results["bias_macro_f1"] = bias_m["f1_macro"]

    # Emotion
    em_loader = DataLoader(em_val_dataset, batch_size=64, shuffle=False,
                           collate_fn=_emotion_collate)
    em_logits, em_labels = [], []
    for batch in em_loader:
        kwargs = _batch_to_kwargs(TASK_EMOTION, batch, device)
        out = model(**kwargs)
        em_logits.append(out["emotion_logits"].cpu().numpy())
        em_labels.append(batch["emotion_labels"].numpy())
    em_logits = np.concatenate(em_logits, axis=0)
    em_labels = np.concatenate(em_labels, axis=0)
    from sklearn.metrics import f1_score
    em_preds = (1.0 / (1.0 + np.exp(-em_logits)) >= 0.5).astype(int)
    results["emotion_macro_f1"] = float(f1_score(em_labels, em_preds, average="macro", zero_division=0))

    # Composite score (mean of normalised per-task F1s)
    results["composite"] = float(np.mean([
        results["epistemic_macro_f1"],
        results["bias_macro_f1"],
        results["emotion_macro_f1"],
    ]))
    return results


# ── W&B helpers ───────────────────────────────────────────────────────────────

def _init_wandb(cfg: dict, run_dir: Path, dry_run: bool):
    """
    Initialize a W&B run. Returns the active run object, or None if wandb is
    unavailable, not logged in, or this is a dry-run.
    """
    if dry_run or _wandb_module is None:
        return None
    log_cfg = cfg.get("logging", {})
    project = log_cfg.get("wandb_project", "ece283-unified")
    entity  = log_cfg.get("wandb_entity") or None
    try:
        run = _wandb_module.init(
            project = project,
            entity  = entity,
            name    = run_dir.name,
            config  = cfg,
            dir     = str(run_dir),
            resume  = "allow",
        )
        logger.info("W&B run started: %s", run.url)
        return run
    except Exception as exc:
        logger.warning("W&B init failed (%s) — training without W&B logging.", exc)
        return None


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, dry_run: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model"]["name"])
    batch_size = cfg["training"]["batch_size"]
    workers    = 0 if dry_run else cfg["training"].get("num_workers", 2)

    logger.info("Loading epistemic data …")
    ep_loaders = _load_epistemic_loaders(cfg, tokenizer, batch_size, workers)

    logger.info("Loading political-bias data …")
    bias_loaders = _load_bias_loaders(cfg, tokenizer, batch_size, workers)

    logger.info("Loading emotion data …")
    em_loaders = _load_emotion_loaders(cfg, tokenizer, batch_size, workers)

    # Three-task DataLoaders for training
    task_loaders = {
        TASK_EPISTEMIC: ep_loaders["sent"],   # sentence head drives epoch length
        TASK_BIAS:      bias_loaders["train"],
        TASK_EMOTION:   em_loaders["train"],
    }
    iterator = MultiTaskBatchIterator(task_loaders)

    # Model
    bias_meta    = REGISTRY[cfg["model"].get("bias_dataset", "10_BABE")]
    bias_lc      = next(lc for lc in bias_meta.label_columns
                        if lc.col == cfg["model"].get("bias_label_col", "label"))
    sent_weights = ep_loaders["sent_class_weights"].to(device) if ep_loaders["sent_class_weights"] is not None else None

    model = UnifiedModel(
        model_name         = cfg["model"]["name"],
        dropout            = cfg["model"].get("dropout", 0.1),
        lambda_token       = cfg["model"].get("lambda_token", 0.3),
        bias_task_type     = bias_lc.task_type,
        bias_num_classes   = bias_lc.num_classes,
        emotion_num_labels = cfg["model"].get("emotion_num_labels", 11),
        sent_class_weights = sent_weights,
    ).to(device)

    task_weights = cfg["training"].get("task_weights", {TASK_EPISTEMIC: 1.0, TASK_BIAS: 1.0, TASK_EMOTION: 0.5})
    optimizer    = build_optimizer(
        model,
        encoder_lr   = cfg["training"]["encoder_lr"],
        head_lr      = cfg["training"]["head_lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    epochs            = 1 if dry_run else cfg["training"]["epochs"]
    grad_accum_steps  = cfg["training"].get("grad_accum_steps", 1)
    total_steps       = (iterator.epoch_steps * len(task_loaders) * epochs) // grad_accum_steps
    warmup_steps      = int(cfg["training"].get("warmup_frac", 0.10) * total_steps)
    scheduler         = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    grad_clip         = cfg["training"].get("grad_clip", 1.0)

    # Bias val loader for validation
    bias_full = MAGPIEDataset(
        dataset_id       = cfg["model"].get("bias_dataset", "10_BABE"),
        cache_dir        = cfg["data"]["cache_dir"],
        tokenizer        = tokenizer,
        max_length       = cfg["data"]["max_len"],
        download_if_missing = True,
        label_col_filter = [cfg["model"].get("bias_label_col", "label")],
    )
    _, bias_val_ds, _ = stratified_split(
        bias_full,
        label_col = cfg["model"].get("bias_label_col", "label"),
        seed      = cfg["data"]["seed"],
    )
    bias_val_loader = DataLoader(
        bias_val_ds, batch_size=64, shuffle=False,
        collate_fn=bias_collate_fn(cfg["model"].get("bias_label_col", "label")),
    )

    ep_data = load_all_data(cfg)  # needed for val

    run_dir = Path(cfg["training"].get("checkpoint_dir", "runs/unified")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    step_log_path = run_dir / "step_losses.jsonl"

    wb_run = _init_wandb(cfg, run_dir, dry_run)

    log_every  = cfg.get("logging", {}).get("log_every_n_steps", 50)
    
    best_composite = -1.0
    history        = []
    global_step    = 0

    total_steps_per_epoch = iterator.epoch_steps * len(task_loaders)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses   = {TASK_EPISTEMIC: [], TASK_BIAS: [], TASK_EMOTION: []}
        recent_losses  = {TASK_EPISTEMIC: 0.0, TASK_BIAS: 0.0, TASK_EMOTION: 0.0}

        bar = tqdm(
            total     = total_steps_per_epoch,
            desc      = f"Epoch {epoch}/{epochs}",
            unit      = "step",
            dynamic_ncols = True,
            leave     = True,
        )

        for step_in_epoch, (task, batch) in enumerate(iterator, 1):
            if dry_run and step_in_epoch > 2 * len(task_loaders):
                logger.info("Dry-run: stopping early after %d steps", step_in_epoch)
                bar.close()
                break

            kwargs = _batch_to_kwargs(task, batch, device)
            out    = model(**kwargs)

            if "loss" not in out:
                bar.update(1)
                continue

            raw_loss = out["loss"].item()
            loss = task_weights.get(task, 1.0) * out["loss"] / grad_accum_steps
            loss.backward()
            epoch_losses[task].append(raw_loss)
            recent_losses[task] = raw_loss

            with open(step_log_path, "a") as _f:
                _f.write(json.dumps({"step": global_step, "epoch": epoch, "task": task, "loss": raw_loss}) + "\n")

            if (global_step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ── Per-step W&B logging ──────────────────────────────────────────
            if wb_run is not None and global_step % log_every == 0:
                wb_run.log({
                    f"train/{task}_loss": raw_loss,
                    "train/lr":           scheduler.get_last_lr()[0],
                    "train/epoch":        epoch,
                }, step=global_step)

            global_step += 1
            bar.update(1)
            bar.set_postfix({
                "task": task[:3],
                "ep":   f"{recent_losses[TASK_EPISTEMIC]:.3f}",
                "bi":   f"{recent_losses[TASK_BIAS]:.3f}",
                "em":   f"{recent_losses[TASK_EMOTION]:.3f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
            })

        else:
            bar.close()

        avg_losses = {t: (float(np.mean(v)) if v else 0.0) for t, v in epoch_losses.items()}
        logger.info(
            "Epoch %d/%d  epistemic=%.4f  bias=%.4f  emotion=%.4f",
            epoch, epochs, avg_losses[TASK_EPISTEMIC], avg_losses[TASK_BIAS], avg_losses[TASK_EMOTION],
        )

        if not dry_run:
            val_metrics = validate(
                model, ep_data, bias_val_loader, em_loaders["dev"],
                tokenizer, cfg, device, task_weights,
            )
            logger.info(
                "  Val — ep_f1=%.4f  bias_f1=%.4f  em_f1=%.4f  composite=%.4f",
                val_metrics["epistemic_macro_f1"],
                val_metrics["bias_macro_f1"],
                val_metrics["emotion_macro_f1"],
                val_metrics["composite"],
            )

            if val_metrics["composite"] > best_composite:
                best_composite = val_metrics["composite"]
                ckpt = run_dir / "best.pt"
                torch.save(model.state_dict(), ckpt)
                logger.info("  ✓ New best checkpoint → %s", ckpt)

            epoch_record = {"epoch": epoch, **avg_losses, **val_metrics}

            # ── Per-epoch W&B logging ─────────────────────────────────────────
            if wb_run is not None:
                wb_run.log({
                    "epoch/epistemic_loss":     avg_losses[TASK_EPISTEMIC],
                    "epoch/bias_loss":          avg_losses[TASK_BIAS],
                    "epoch/emotion_loss":       avg_losses[TASK_EMOTION],
                    "epoch/epistemic_macro_f1": val_metrics["epistemic_macro_f1"],
                    "epoch/bias_macro_f1":      val_metrics["bias_macro_f1"],
                    "epoch/emotion_macro_f1":   val_metrics["emotion_macro_f1"],
                    "epoch/composite":          val_metrics["composite"],
                }, step=global_step)
        else:
            epoch_record = {"epoch": epoch, **avg_losses}

        history.append(epoch_record)
        # Flush after every epoch so a crash doesn't lose prior data
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save(model.state_dict(), run_dir / "last.pt")
    if wb_run is not None:
        wb_run.finish()
    logger.info("Training complete. Outputs in %s", run_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified multi-task model")
    parser.add_argument("--config",   default="models/unified/config.yaml")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run 2 batches per task per epoch to verify the data pipeline")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
