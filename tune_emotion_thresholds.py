"""
Compute per-class F1-optimal thresholds for the emotion head on the dev split.

Run from the project root:
    python3 tune_emotion_thresholds.py \
        --checkpoint runs/unified/seed42-5ep/best.pt \
        --config     models/unified/config.yaml

Outputs runs/unified/seed42-5ep/emotion_thresholds.json and prints a summary table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))

from models.unified.model import TASK_EMOTION
from models.unified.predict import load_unified_predictor
from models.unified.train import EmotionTorchDataset, _emotion_collate

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]


@torch.no_grad()
def _score_dev_set(predictor, dev_dataset, batch_size=64):
    """Return raw sigmoid scores (N, 11) and binary labels (N, 11) for the dev split."""
    loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=_emotion_collate)
    all_logits, all_labels = [], []
    predictor.model.eval()
    for batch in tqdm(loader, desc="Scoring dev set", unit="batch"):
        ids  = batch["input_ids"].to(predictor.device)
        mask = batch["attention_mask"].to(predictor.device)
        out  = predictor.model(input_ids=ids, attention_mask=mask, task=TASK_EMOTION)
        all_logits.append(out["emotion_logits"].cpu().numpy())
        all_labels.append(batch["emotion_labels"].numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    return scores, labels


def _find_thresholds(scores, labels, n_steps=80, default=0.5):
    """For each class, sweep [0.05, 0.95] and pick the threshold maximising binary F1."""
    thresholds = []
    for j in range(scores.shape[1]):
        if int(labels[:, j].sum()) == 0:
            thresholds.append(default)
            continue
        best_t, best_f1 = default, 0.0
        for t in np.linspace(0.05, 0.95, n_steps):
            f1 = f1_score(labels[:, j], scores[:, j] >= t, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        thresholds.append(best_t)
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/unified/seed42-5ep/best.pt")
    parser.add_argument("--config",     default="models/unified/config.yaml")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model…")
    predictor = load_unified_predictor(str(ckpt_path), args.config, device=args.device)
    print(f"Device: {predictor.device}")

    # ── Rebuild emotion dev split (same config / seed as training) ────────────
    print("Loading emotion dev split…")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from models.emotion.config import EmotionalFramingConfig
    from models.emotion.data import load_and_split

    em_cfg = EmotionalFramingConfig()
    em_cfg.magpie_data_dir = cfg["data"].get("magpie_data_dir", em_cfg.magpie_data_dir)
    em_cfg.hf_cache_dir    = cfg["data"].get("hf_cache_dir",    em_cfg.hf_cache_dir)
    em_cfg.max_seq_length  = cfg["data"]["max_len"]
    em_cfg.seed            = cfg["data"]["seed"]

    tokenizer    = RobertaTokenizerFast.from_pretrained(cfg["model"]["name"])
    dataset_dict = load_and_split(em_cfg)

    def _tokenize(batch):
        enc = tokenizer(batch["text"], max_length=em_cfg.max_seq_length,
                        padding="max_length", truncation=True)
        enc["labels"] = [list(map(float, lv)) for lv in batch["labels"]]
        return enc

    tokenized = dataset_dict.map(_tokenize, batched=True, remove_columns=["text", "source"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset = EmotionTorchDataset(tokenized["dev"])
    print(f"Dev examples: {len(dev_dataset):,}")

    # ── Score dev set ─────────────────────────────────────────────────────────
    scores, labels = _score_dev_set(predictor, dev_dataset)

    # ── Find optimal thresholds ───────────────────────────────────────────────
    print("Finding per-class optimal thresholds…")
    thresholds = _find_thresholds(scores, labels)

    # ── Report ────────────────────────────────────────────────────────────────
    fixed_preds = (scores >= 0.5).astype(int)
    opt_preds   = np.stack(
        [(scores[:, j] >= thresholds[j]).astype(int) for j in range(len(EMOTIONS))],
        axis=1,
    )

    f1_fixed = f1_score(labels, fixed_preds, average=None, zero_division=0)
    f1_opt   = f1_score(labels, opt_preds,   average=None, zero_division=0)

    print(f"\n{'Emotion':<15}  {'Threshold':>9}  {'F1@0.5':>8}  {'F1@opt':>8}  {'Pos%':>6}")
    print("─" * 58)
    for j, emotion in enumerate(EMOTIONS):
        pos_pct = 100.0 * labels[:, j].mean()
        delta   = f1_opt[j] - f1_fixed[j]
        marker  = " ▲" if delta > 0.01 else ("  " if delta >= 0 else " ▼")
        print(f"{emotion:<15}  {thresholds[j]:>9.3f}  {f1_fixed[j]:>8.3f}  "
              f"{f1_opt[j]:>8.3f}{marker}  {pos_pct:>5.1f}%")

    old_macro = float(f1_score(labels, fixed_preds, average="macro", zero_division=0))
    new_macro = float(f1_score(labels, opt_preds,   average="macro", zero_division=0))
    print(f"\nMacro-F1:  {old_macro:.4f}  →  {new_macro:.4f}  (+{new_macro - old_macro:+.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ckpt_path.parent / "emotion_thresholds.json"
    out_path.write_text(json.dumps({
        "emotions":             EMOTIONS,
        "thresholds":           thresholds,
        "dev_macro_f1_fixed":   round(old_macro, 6),
        "dev_macro_f1_optimal": round(new_macro, 6),
    }, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
