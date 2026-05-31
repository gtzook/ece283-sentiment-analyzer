"""
train.py — training loop for the epistemic certainty model.

Usage:
    python -m models.epistemic.train                        # uses config.yaml
    python -m models.epistemic.train --config path/to.yaml
    python -m models.epistemic.train --bio_token_weight 0.25  # override any key

Joint training strategy:
    Each step draws one sentence-head batch and one token-head batch.
    Loss = sent_loss + lambda_token * (news_tok_loss + bio_weight * bio_tok_loss)

    The sentence and token dataloaders cycle independently; the shorter one
    wraps around rather than truncating the epoch.

Checkpointing:
    Best model by validation sentence-head macro-F1 is saved to
    runs/<timestamp>/best.pt.  Final model saved as last.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.epistemic.data import (
    SentDataset,
    SentExample,
    TokenDataset,
    TokenExample,
    load_bioscope,
    load_factbank_ldc,
    load_szeged_bio,
    load_szeged_factbank,
    load_szeged_wiki,
)
from models.epistemic.model import EpistemicModel


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Document-level split ──────────────────────────────────────────────────────

def _doc_id(example) -> str:
    """Return the document identifier for an example."""
    return example.doc_id or example.source


def doc_level_split(
    examples: list,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    Split examples by document ID so no document appears in multiple splits.
    Returns (train, val, test).
    """
    doc_to_examples: dict[str, list] = defaultdict(list)
    for ex in examples:
        doc_to_examples[_doc_id(ex)].append(ex)

    docs = sorted(doc_to_examples.keys())
    rng  = random.Random(seed)
    rng.shuffle(docs)

    n         = len(docs)
    n_train   = math.floor(n * train_frac)
    n_val     = math.floor(n * val_frac)

    train_docs = set(docs[:n_train])
    val_docs   = set(docs[n_train : n_train + n_val])
    # test = remainder

    train, val, test = [], [], []
    for ex in examples:
        d = _doc_id(ex)
        if d in train_docs:
            train.append(ex)
        elif d in val_docs:
            val.append(ex)
        else:
            test.append(ex)

    return train, val, test


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_data(cfg: dict) -> dict:
    """
    Load all corpora and return split example lists.

    Returns a dict with keys:
        sent_train, sent_val, sent_test          (SentExample lists)
        tok_news_train, tok_news_val             (TokenExample lists, news domain)
        tok_bio_train                            (TokenExample lists, bio domain)
    """
    dc   = cfg["data"]
    seed = cfg["training"]["seed"]
    tf   = dc["train_frac"]
    vf   = dc["val_frac"]

    # ── Sentence head sources ─────────────────────────────────────────────────
    sent_all: list[SentExample] = []
    tok_news_all: list[TokenExample] = []

    for path, loader in [
        (dc["wiki_xml"],            load_szeged_wiki),
        (dc["szeged_factbank_xml"], load_szeged_factbank),
    ]:
        s, t = loader(path)
        sent_all.extend(s)
        tok_news_all.extend(t)

    fb_sent, fb_tok = load_factbank_ldc(dc["factbank_ldc_ann"])
    sent_all.extend(fb_sent)
    tok_news_all.extend(fb_tok)

    # ── Token head: bio domain ────────────────────────────────────────────────
    tok_bio_all: list[TokenExample] = []
    for path in dc.get("bio_xmls", []):
        tok_bio_all.extend(load_szeged_bio(path))
    for path in dc.get("bioscope_xmls", []):
        tok_bio_all.extend(load_bioscope(path))

    # ── Document-level splits ─────────────────────────────────────────────────
    # Sentence and news-token examples share document IDs so their splits align.
    # Bio examples are biomedical — not split with news docs; use same fractions
    # applied to a shuffled list.
    sent_train, sent_val, sent_test     = doc_level_split(sent_all,     tf, vf, seed)
    tok_news_train, tok_news_val, tok_news_test = doc_level_split(tok_news_all, tf, vf, seed)

    rng = random.Random(seed)
    rng.shuffle(tok_bio_all)
    n_bio      = len(tok_bio_all)
    nb_train   = math.floor(n_bio * tf)
    tok_bio_train = tok_bio_all[:nb_train]
    # bio val/test not used for checkpointing; omit for now

    print(
        f"Data split:"
        f"\n  sent:      train={len(sent_train):>6}  val={len(sent_val):>5}  test={len(sent_test):>5}"
        f"\n  tok news:  train={len(tok_news_train):>6}  val={len(tok_news_val):>5}  test={len(tok_news_test):>5}"
        f"\n  tok bio:   train={len(tok_bio_train):>6}  (bio, down-weighted)"
    )

    return {
        "sent_train":     sent_train,
        "sent_val":       sent_val,
        "sent_test":      sent_test,
        "tok_news_train": tok_news_train,
        "tok_news_val":   tok_news_val,
        "tok_news_test":  tok_news_test,
        "tok_bio_train":  tok_bio_train,
    }


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_sent_class_weights(examples: list[SentExample]) -> torch.Tensor:
    """Inverse-frequency weights for the 3-class sentence loss."""
    counts = torch.zeros(3)
    for ex in examples:
        counts[ex.label] += 1
    weights = counts.sum() / (3 * counts.clamp(min=1))
    return weights


# ── Logging ───────────────────────────────────────────────────────────────────

def make_run_dir(base: str) -> Path:
    ts      = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def try_init_wandb(cfg: dict, run_dir: Path):
    lc = cfg.get("logging", {})
    try:
        import wandb
        import os
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY not set")
        wandb.init(
            project=lc.get("wandb_project", "ece283-epistemic"),
            entity=lc.get("wandb_entity") or None,
            config=cfg,
            dir=str(run_dir),
        )
        return wandb
    except Exception:
        return None


class JsonLogger:
    def __init__(self, path: Path) -> None:
        self.path    = path
        self.records: list[dict] = []

    def log(self, d: dict) -> None:
        self.records.append(d)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: EpistemicModel,
    sent_loader: DataLoader,
    tok_loader: DataLoader | None,
    device: torch.device,
    lambda_token: float,
) -> dict:
    model.eval()
    all_sent_true, all_sent_pred = [], []
    total_loss = 0.0
    n_steps    = 0

    for batch in sent_loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        out     = model(**batch)
        total_loss += out["loss"].item()
        preds = out["sent_logits"].argmax(dim=-1).cpu().tolist()
        all_sent_pred.extend(preds)
        all_sent_true.extend(batch["sent_label"].cpu().tolist())
        n_steps += 1

    sent_macro_f1 = f1_score(all_sent_true, all_sent_pred, average="macro", zero_division=0)

    metrics = {
        "val_loss":       total_loss / max(n_steps, 1),
        "val_macro_f1":   sent_macro_f1,
    }

    # Token head val macro-F1 (optional, only when news token val data present)
    if tok_loader is not None:
        all_tok_true, all_tok_pred = [], []
        for batch in tok_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_labels=batch["token_labels"],
            )
            mask  = batch["token_labels"].view(-1) != -100
            preds = out["token_logits"].view(-1, 2).argmax(dim=-1)[mask].cpu().tolist()
            true  = batch["token_labels"].view(-1)[mask].cpu().tolist()
            all_tok_pred.extend(preds)
            all_tok_true.extend(true)
        metrics["val_token_macro_f1"] = f1_score(
            all_tok_true, all_tok_pred, average="macro", zero_division=0
        )

    model.train()
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    tc = cfg["training"]
    dc = cfg["data"]
    mc = cfg["model"]

    set_seed(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = tc.get("fp16", False) and device.type == "cuda"

    run_dir    = make_run_dir(tc.get("checkpoint_dir", "runs/"))
    wandb_run  = try_init_wandb(cfg, run_dir)
    json_log   = JsonLogger(run_dir / "metrics.json")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(mc["name"])

    # ── Load data ─────────────────────────────────────────────────────────────
    splits = load_all_data(cfg)

    sent_class_weights = compute_sent_class_weights(splits["sent_train"]).to(device)

    max_len = dc.get("max_len", 128)
    bs      = tc["batch_size"]

    sent_train_ds  = SentDataset(splits["sent_train"],     tokenizer, max_len)
    sent_val_ds    = SentDataset(splits["sent_val"],       tokenizer, max_len)
    tok_news_ds    = TokenDataset(splits["tok_news_train"], tokenizer, max_len)
    tok_bio_ds     = TokenDataset(splits["tok_bio_train"],  tokenizer, max_len)
    tok_news_val_ds = TokenDataset(splits["tok_news_val"],  tokenizer, max_len)

    sent_train_dl  = DataLoader(sent_train_ds,   batch_size=bs, shuffle=True,  drop_last=True)
    sent_val_dl    = DataLoader(sent_val_ds,     batch_size=bs, shuffle=False)
    tok_news_dl    = DataLoader(tok_news_ds,     batch_size=bs, shuffle=True,  drop_last=True)
    tok_bio_dl     = DataLoader(tok_bio_ds,      batch_size=bs, shuffle=True,  drop_last=True)
    tok_news_val_dl = DataLoader(tok_news_val_ds, batch_size=bs, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EpistemicModel(
        model_name=mc["name"],
        dropout=mc.get("dropout", 0.1),
        lambda_token=mc.get("lambda_token", 0.3),
        sent_class_weights=sent_class_weights,
    ).to(device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc.get("weight_decay", 0.01),
    )

    steps_per_epoch = max(len(sent_train_dl), len(tok_news_dl))
    total_steps     = steps_per_epoch * tc["epochs"]
    warmup_steps    = math.ceil(total_steps * tc.get("warmup_frac", 0.10))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    bio_weight = dc.get("bio_token_weight", 0.5)

    # Cycle the shorter token loaders so they wrap rather than truncate the epoch
    tok_news_iter = cycle(tok_news_dl)
    tok_bio_iter  = cycle(tok_bio_dl) if len(tok_bio_ds) > 0 else None

    best_val_f1  = -1.0
    global_step  = 0

    step_log_path = run_dir / "step_losses.jsonl"

    print(
        f"\nTraining on {device}  |  fp16={use_fp16}  |  "
        f"total_steps={total_steps}  |  warmup={warmup_steps}"
    )

    from tqdm import tqdm

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        epoch_loss           = 0.0
        epoch_sent_losses    = []
        epoch_tok_news_losses = []
        epoch_tok_bio_losses  = []

        pbar = tqdm(
            sent_train_dl,
            desc=f"Epoch {epoch}/{tc['epochs']}",
            unit="batch",
            dynamic_ncols=True,
        )
        for sent_batch in pbar:
            sent_batch = {k: v.to(device) for k, v in sent_batch.items()}
            tok_news_batch = {k: v.to(device) for k, v in next(tok_news_iter).items()}

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_fp16):
                # Sentence head loss
                sent_out   = model(**sent_batch)
                sent_loss  = sent_out["loss"]
                loss       = sent_loss

                # News token head loss (full weight)
                tok_out      = model(
                    input_ids=tok_news_batch["input_ids"],
                    attention_mask=tok_news_batch["attention_mask"],
                    token_labels=tok_news_batch["token_labels"],
                )
                tok_news_loss = tok_out["loss"]
                loss = loss + tok_news_loss

                # Bio token head loss (down-weighted)
                tok_bio_loss = None
                if tok_bio_iter is not None and bio_weight > 0.0:
                    tok_bio_batch = {k: v.to(device) for k, v in next(tok_bio_iter).items()}
                    bio_out = model(
                        input_ids=tok_bio_batch["input_ids"],
                        attention_mask=tok_bio_batch["attention_mask"],
                        token_labels=tok_bio_batch["token_labels"],
                    )
                    tok_bio_loss = bio_out["loss"]
                    loss = loss + bio_weight * tok_bio_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance the LR schedule when the optimizer actually stepped.
            # scaler.step() is a no-op (skips optimizer) when gradients contain
            # inf/nan; in that case the scale is reduced and we must not step
            # the scheduler, or the warmup schedule drifts ahead of reality.
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            sent_loss_val     = sent_loss.item()
            tok_news_loss_val = tok_news_loss.item()
            tok_bio_loss_val  = tok_bio_loss.item() if tok_bio_loss is not None else None
            total_loss_val    = loss.item()

            epoch_sent_losses.append(sent_loss_val)
            epoch_tok_news_losses.append(tok_news_loss_val)
            if tok_bio_loss_val is not None:
                epoch_tok_bio_losses.append(tok_bio_loss_val)

            step_record = {
                "step": global_step, "epoch": epoch,
                "sent_loss": sent_loss_val,
                "tok_news_loss": tok_news_loss_val,
                "total_loss": total_loss_val,
            }
            if tok_bio_loss_val is not None:
                step_record["tok_bio_loss"] = tok_bio_loss_val
            with open(step_log_path, "a") as _f:
                _f.write(json.dumps(step_record) + "\n")

            epoch_loss  += total_loss_val
            global_step += 1
            pbar.set_postfix(loss=f"{total_loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        pbar.close()
        avg_loss = epoch_loss / max(len(sent_train_dl), 1)

        val_metrics = validate(model, sent_val_dl, tok_news_val_dl, device, mc.get("lambda_token", 0.3))
        val_metrics["epoch"]              = epoch
        val_metrics["train_loss"]         = avg_loss
        val_metrics["train_sent_loss"]    = float(np.mean(epoch_sent_losses)) if epoch_sent_losses else 0.0
        val_metrics["train_tok_news_loss"] = float(np.mean(epoch_tok_news_losses)) if epoch_tok_news_losses else 0.0
        val_metrics["train_tok_bio_loss"] = float(np.mean(epoch_tok_bio_losses)) if epoch_tok_bio_losses else 0.0
        val_metrics["step"]               = global_step

        print(
            f"Epoch {epoch}/{tc['epochs']}  "
            f"train_loss={avg_loss:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_macro_f1={val_metrics['val_macro_f1']:.4f}"
            + (f"  val_tok_f1={val_metrics['val_token_macro_f1']:.4f}"
               if "val_token_macro_f1" in val_metrics else "")
        )

        json_log.log(val_metrics)
        if wandb_run:
            wandb_run.log(val_metrics)

        if val_metrics["val_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["val_macro_f1"]
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  → new best val macro-F1: {best_val_f1:.4f}  saved to {run_dir}/best.pt")

    torch.save(model.state_dict(), run_dir / "last.pt")
    print(f"\nTraining complete. Run dir: {run_dir}")
    print(f"Best val macro-F1: {best_val_f1:.4f}")

    if wandb_run:
        wandb_run.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _deep_set(d: dict, key: str, value) -> None:
    """Set a dotted key like 'data.bio_token_weight' into a nested dict."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the epistemic certainty model")
    parser.add_argument(
        "--config", default="models/epistemic/config.yaml",
        help="Path to config YAML (default: models/epistemic/config.yaml)",
    )
    parser.add_argument(
        "--set", nargs="*", metavar="KEY=VALUE",
        help="Override config values, e.g. --set data.bio_token_weight=0.25 training.epochs=5",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for kv in (args.set or []):
        k, _, v = kv.partition("=")
        # Try int → float → string
        for cast in (int, float, lambda x: x):
            try:
                v = cast(v)
                break
            except (ValueError, TypeError):
                pass
        _deep_set(cfg, k.strip(), v)

    train(cfg)


if __name__ == "__main__":
    main()
