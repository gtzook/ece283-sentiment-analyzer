"""
Training script for the emotional framing floor model.

Usage:
    python train.py                   # full training run
    python train.py --debug           # 100-sample smoke test
    python train.py --threshold 0.4   # override inference threshold
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from models.emotion.config import EmotionalFramingConfig, EMOTION_LABELS
from models.emotion.data import load_and_split
from models.emotion.eval import full_report, make_compute_metrics, tune_threshold
from models.emotion.model import EmotionalFramingClassifier

import torch.optim

from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

AdamW = torch.optim.AdamW

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers


# ---------------------------------------------------------------------------
# Custom Trainer with per-param-group learning rates
# ---------------------------------------------------------------------------

class EmotionalFramingTrainer(Trainer):
    """
    Subclasses HF Trainer only to set encoder_lr / head_lr param groups.
    Everything else (eval loop, checkpointing, logging) is standard Trainer.

    MTL HOOK: In the multi-task model, create_optimizer would add param groups
    for each task head, each potentially with its own LR.
    """

    def __init__(self, *args, encoder_lr: float, head_lr: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder_lr = encoder_lr
        self._head_lr = head_lr

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        no_decay = {"bias", "LayerNorm.weight"}

        # Encoder params — lower LR, weight decay only on non-bias/norm params
        encoder_decay = [
            p for n, p in self.model.roberta.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        encoder_nodecay = [
            p for n, p in self.model.roberta.named_parameters()
            if any(nd in n for nd in no_decay)
        ]
        # Classification head — higher LR
        head_params = list(self.model.classifier.parameters())

        # MTL HOOK: for MTL, add a param group per task head in head_params
        optimizer_grouped_parameters = [
            {"params": encoder_decay,  "lr": self._encoder_lr, "weight_decay": self.args.weight_decay},
            {"params": encoder_nodecay, "lr": self._encoder_lr, "weight_decay": 0.0},
            {"params": head_params,    "lr": self._head_lr,    "weight_decay": self.args.weight_decay},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def make_tokenize_fn(tokenizer, max_length: int):
    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        enc["labels"] = [list(map(float, lv)) for lv in batch["labels"]]
        return enc
    return tokenize


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: EmotionalFramingConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    seed_everything(cfg.seed)

    # ── CUDA / fp16 ──────────────────────────────────────────────────────────
    has_cuda = torch.cuda.is_available()
    cfg.fp16 = cfg.fp16 and has_cuda
    if not has_cuda:
        logger.warning("No CUDA detected — running on CPU, fp16 disabled")

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    dataset_dict = load_and_split(cfg)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenize_fn = make_tokenize_fn(tokenizer, cfg.max_seq_length)

    logger.info("Tokenizing …")
    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "source"],
        desc="Tokenizing",
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = RobertaConfig.from_pretrained(cfg.model_name)
    model_config.num_labels = cfg.num_labels
    model = EmotionalFramingClassifier.from_pretrained(
        cfg.model_name,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    # ── Warmup steps (warmup_ratio is deprecated in transformers ≥ v5.2) ─────
    steps_per_epoch = math.ceil(len(tokenized["train"]) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    logger.info("Total training steps: %d | warmup steps: %d", total_steps, warmup_steps)

    # ── wandb — verify with a real API call before enabling ──────────────────
    # wandb.login() only checks that a key exists locally; it does NOT verify it
    # server-side. We ping wandb.Api().viewer() to catch invalid/corrupted keys.
    report_to: list[str] = []
    try:
        import wandb
        wandb.Api().viewer()   # raises AuthenticationError if key is bad
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
        report_to = ["wandb"]
        logger.info("wandb authenticated — logging to project '%s'", cfg.wandb_project)
    except Exception as e:
        logger.warning(
            "wandb auth failed (%s). Run `wandb login` to fix. Training without W&B.", e
        )

    # ── Training arguments ───────────────────────────────────────────────────
    # learning_rate here is a placeholder; actual per-group LRs are in create_optimizer
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.max_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.encoder_lr,          # reference; overridden by custom optimizer
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=cfg.fp16,
        seed=cfg.seed,
        report_to=report_to,
        run_name="emotional-framing-floor" + ("-debug" if cfg.debug else ""),
        logging_steps=cfg.logging_steps,
        dataloader_pin_memory=has_cuda,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = EmotionalFramingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        compute_metrics=make_compute_metrics(cfg.threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
        encoder_lr=cfg.encoder_lr,
        head_lr=cfg.head_lr,
    )

    logger.info("Starting training …")
    trainer.train()

    # ── Threshold tuning on dev ──────────────────────────────────────────────
    logger.info("Tuning threshold on dev set …")
    dev_pred = trainer.predict(tokenized["dev"])
    best_threshold = tune_threshold(
        dev_pred.predictions,
        dev_pred.label_ids,
    )

    # ── Final evaluation on test ──────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    test_pred = trainer.predict(tokenized["test"])
    full_report(test_pred.predictions, test_pred.label_ids,
                threshold=best_threshold, split_name="test")

    # Also report dev for comparison
    full_report(dev_pred.predictions, dev_pred.label_ids,
                threshold=best_threshold, split_name="dev")

    # ── Save best checkpoint ──────────────────────────────────────────────────
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Best checkpoint saved to %s", out)

    # Persist the tuned threshold alongside the checkpoint
    (out / "threshold.txt").write_text(str(best_threshold))
    logger.info("Tuned threshold: %.2f → saved to %s/threshold.txt", best_threshold, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train emotional framing floor model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speed presets:
  --debug        100 samples, roberta-base, seq_len=128  (~2 min, sanity check only)
  --fast         20k samples, distilroberta-base, seq_len=64, 3 epochs  (~20 min)
  --samples N    N samples, full roberta-base  (scale as needed)
  (none)         Full run: ~256k samples, roberta-base, 10 epochs  (~4 h)
        """,
    )
    parser.add_argument("--debug", action="store_true",
                        help="100-sample smoke test (roberta-base, seq_len=128)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick test run: distilroberta-base, seq_len=64, 3 epochs, 20k samples")
    parser.add_argument("--samples", type=int, default=None, metavar="N",
                        help="Cap training+eval data at N samples (random stratified subset)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override sigmoid threshold (default: tuned on dev set)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override checkpoint output directory")
    parser.add_argument("--magpie-dir", type=str, default=None,
                        help="Override path to MAGPIE data directory")
    args = parser.parse_args()

    cfg = EmotionalFramingConfig()

    if args.fast:
        cfg.model_name = "distilroberta-base"
        cfg.max_seq_length = 64
        cfg.max_epochs = 3
        cfg.max_samples = 20_000
        cfg.output_dir = "./checkpoints/emotional_framing_floor_fast"
        logger.info(
            "Fast mode: distilroberta-base | seq_len=64 | 3 epochs | 20k samples"
        )

    cfg.debug = args.debug
    if args.samples is not None:
        cfg.max_samples = args.samples
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.magpie_dir is not None:
        cfg.magpie_data_dir = args.magpie_dir

    main(cfg)
