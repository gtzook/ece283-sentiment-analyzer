"""
Cross-task evaluation: unified model vs. three individual baselines.

Loads a trained UnifiedModel checkpoint and the three baseline checkpoints,
runs each on their respective test split, and prints a comparison table.

Usage:
    python -m models.unified.eval \\
        --unified-checkpoint  runs/unified/20260601_120000/best.pt \\
        --epistemic-checkpoint runs/20260530_071702/best.pt \\
        --bias-checkpoint      runs/10_BABE/label/best.pt \\
        --emotion-checkpoint   checkpoints/emotional_framing_floor \\
        --config               models/unified/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).parents[2]))
from models.unified.model import UnifiedModel, TASK_EPISTEMIC, TASK_BIAS, TASK_EMOTION
from models.unified.train import EmotionTorchDataset, _emotion_collate, _batch_to_kwargs
from models.epistemic.data import SentDataset, TokenDataset
from models.epistemic.train import load_all_data, doc_level_split
from models.epistemic.eval import compute_ece
from models.epistemic.model import EpistemicModel
from models.epistemic.predict import load_predictor as load_ep_predictor
from models.political_bias.model import RoBERTaClassifier
from models.political_bias.train_baseline import collate_fn as bias_collate_fn, _cls_metrics
from models.political_bias.eval import evaluate as bias_evaluate
from models.emotion.config import EmotionalFramingConfig, EMOTION_LABELS
from models.emotion.data import load_and_split as emotion_load_and_split
from models.emotion.eval import _apply_threshold, tune_threshold
from src.data.dataset import MAGPIEDataset
from src.data.splits import stratified_split
from src.data.registry import REGISTRY, TaskType


def _macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, g in zip(preds, labels):
        if p == g: tp[g] += 1
        else:      fp[p] += 1; fn[g] += 1
    f1s = []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


# ── Epistemic evaluation helpers ──────────────────────────────────────────────

@torch.no_grad()
def _eval_epistemic_unified(
    model: UnifiedModel,
    sent_test: SentDataset,
    device: torch.device,
) -> dict:
    model.eval()
    loader = DataLoader(sent_test, batch_size=64, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        kwargs = _batch_to_kwargs(TASK_EPISTEMIC, batch, device)
        out    = model(**kwargs)
        probs  = torch.softmax(out["sent_logits"], dim=-1).cpu().numpy()
        all_preds.extend(probs.argmax(-1).tolist())
        all_labels.extend(batch["sent_label"].numpy().tolist())
        all_probs.append(probs)
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.concatenate(all_probs, axis=0)
    return {
        "sent_macro_f1": _macro_f1(preds, labels, 3),
        "ece":           compute_ece(probs, labels),
    }


@torch.no_grad()
def _eval_epistemic_baseline(
    predictor,
    sent_examples: list,
    tokenizer,
    device: torch.device,
) -> dict:
    from models.epistemic.predict import predict_batch
    texts  = [ex.text for ex in sent_examples]
    labels = [ex.label for ex in sent_examples]
    results = predict_batch(predictor, texts)
    preds  = np.array([r["label"] for r in results])
    probs  = np.array([r["sent_probs"] for r in results])
    lbls   = np.array(labels)
    return {
        "sent_macro_f1": _macro_f1(preds, lbls, 3),
        "ece":           compute_ece(probs, lbls),
    }


# ── Bias evaluation helpers ───────────────────────────────────────────────────

@torch.no_grad()
def _eval_bias_unified(
    model: UnifiedModel,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in test_loader:
        kwargs = _batch_to_kwargs(TASK_BIAS, batch, device)
        out    = model(**kwargs)
        all_preds.extend(out["bias_logits"].argmax(-1).cpu().numpy().tolist())
        all_labels.extend(batch["labels"].numpy().tolist())
    return {
        "macro_f1": _macro_f1(np.array(all_preds), np.array(all_labels), num_classes),
        "accuracy": float((np.array(all_preds) == np.array(all_labels)).mean()),
    }


# ── Emotion evaluation helpers ────────────────────────────────────────────────

@torch.no_grad()
def _eval_emotion_unified(
    model: UnifiedModel,
    test_dataset: EmotionTorchDataset,
    threshold: float,
    device: torch.device,
) -> dict:
    model.eval()
    loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=_emotion_collate)
    logits_list, labels_list = [], []
    for batch in loader:
        kwargs = _batch_to_kwargs(TASK_EMOTION, batch, device)
        out    = model(**kwargs)
        logits_list.append(out["emotion_logits"].cpu().numpy())
        labels_list.append(batch["emotion_labels"].numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    preds  = (probs >= threshold).astype(int)
    return {
        "macro_f1":    float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "micro_f1":    float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds)),
    }


@torch.no_grad()
def _eval_emotion_baseline(
    baseline_model,
    test_dataset: EmotionTorchDataset,
    threshold: float,
    device: torch.device,
) -> dict:
    baseline_model.eval()
    loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=_emotion_collate)
    logits_list, labels_list = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = baseline_model(input_ids=ids, attention_mask=mask)
        logits_list.append(out.logits.cpu().numpy())
        labels_list.append(batch["emotion_labels"].numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    preds  = (probs >= threshold).astype(int)
    return {
        "macro_f1":    float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "micro_f1":    float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds)),
    }


# ── Table printing ────────────────────────────────────────────────────────────

def _print_table(results: dict) -> None:
    W = 20

    def row(task, metric, baseline, unified):
        b = f"{baseline:.4f}" if isinstance(baseline, float) else str(baseline)
        u = f"{unified:.4f}"  if isinstance(unified,  float) else str(unified)
        delta = unified - baseline if isinstance(baseline, float) else 0.0
        sign  = "+" if delta >= 0 else ""
        return f"  {task:<14s}  {metric:<22s}  {b:<12s}  {u:<12s}  {sign}{delta:.4f}"

    print("\n" + "=" * 70)
    print(f"  {'Task':<14s}  {'Metric':<22s}  {'Baseline':<12s}  {'Unified':<12s}  {'Delta'}")
    print("=" * 70)
    for task, metrics in results.items():
        first = True
        for metric, (baseline_val, unified_val) in metrics.items():
            t_label = task if first else ""
            print(row(t_label, metric, baseline_val, unified_val))
            first = False
        print("-" * 70)
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unified model against three baselines")
    parser.add_argument("--unified-checkpoint",   required=True)
    parser.add_argument("--epistemic-checkpoint",  default=None)
    parser.add_argument("--bias-checkpoint",       default=None)
    parser.add_argument("--emotion-checkpoint",    default=None,
                        help="Path to HF checkpoint directory (not a .pt file)")
    parser.add_argument("--config",   default="models/unified/config.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model"]["name"])

    # ── Load unified model ────────────────────────────────────────────────────
    if "bias_datasets" in cfg["model"]:
        bias_task_type   = TaskType.BINARY_CLS
        bias_num_classes = 2
    else:
        bias_meta        = REGISTRY[cfg["model"].get("bias_dataset", "10_BABE")]
        bias_lc          = next(lc for lc in bias_meta.label_columns
                                if lc.col == cfg["model"].get("bias_label_col", "label"))
        bias_task_type   = bias_lc.task_type
        bias_num_classes = bias_lc.num_classes

    unified = UnifiedModel(
        model_name         = cfg["model"]["name"],
        dropout            = cfg["model"].get("dropout", 0.1),
        lambda_token       = cfg["model"].get("lambda_token", 0.3),
        bias_task_type     = bias_task_type,
        bias_num_classes   = bias_num_classes,
        emotion_num_labels = cfg["model"].get("emotion_num_labels", 11),
    ).to(device)
    unified.load_state_dict(
        torch.load(args.unified_checkpoint, map_location=device, weights_only=True)
    )
    unified.eval()
    print(f"Loaded unified model from {args.unified_checkpoint}")

    results = {}

    # ── Epistemic ─────────────────────────────────────────────────────────────
    print("\nEvaluating epistemic task …")
    ep_data   = load_all_data(cfg)
    sent_test = SentDataset(ep_data["sent_test"], tokenizer, max_len=cfg["data"]["max_len"])

    ep_unified = _eval_epistemic_unified(unified, sent_test, device)
    ep_metrics = {"sent_macro_f1": ep_unified["sent_macro_f1"], "ece": ep_unified["ece"]}

    ep_baseline = {}
    if args.epistemic_checkpoint:
        ep_predictor = load_ep_predictor(args.epistemic_checkpoint, device=str(device))
        ep_base_raw  = _eval_epistemic_baseline(ep_predictor, ep_data["sent_test"], tokenizer, device)
        ep_baseline  = {"sent_macro_f1": ep_base_raw["sent_macro_f1"], "ece": ep_base_raw["ece"]}
    else:
        ep_baseline = {k: float("nan") for k in ep_metrics}

    results["Epistemic"] = {
        metric: (ep_baseline[metric], ep_metrics[metric]) for metric in ep_metrics
    }

    # ── Bias ──────────────────────────────────────────────────────────────────
    print("Evaluating bias task …")
    from torch.utils.data import ConcatDataset
    from models.unified.train import _make_bias_transform

    if "bias_datasets" in cfg["model"]:
        specs = cfg["model"]["bias_datasets"]
    else:
        specs = [{"dataset_id": cfg["model"].get("bias_dataset", "10_BABE"),
                  "label_col":  cfg["model"].get("bias_label_col", "label")}]

    test_subsets = []
    for spec in specs:
        ds_id     = spec["dataset_id"]
        label_col = spec["label_col"]
        transform = _make_bias_transform(label_col, spec.get("remap"))
        full_ds = MAGPIEDataset(
            dataset_id          = ds_id,
            cache_dir           = cfg["data"]["cache_dir"],
            tokenizer           = tokenizer,
            max_length          = cfg["data"]["max_len"],
            download_if_missing = True,
            label_col_filter    = [label_col],
            transform           = transform,
        )
        _, _, test_ds = stratified_split(
            full_ds,
            label_col  = label_col,
            train_frac = cfg["data"].get("train_frac", 0.80),
            val_frac   = cfg["data"].get("val_frac",   0.10),
            seed       = cfg["data"]["seed"],
        )
        test_subsets.append(test_ds)

    primary_label_col = specs[0]["label_col"]
    bias_test_loader = DataLoader(
        ConcatDataset(test_subsets), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        collate_fn=bias_collate_fn(primary_label_col),
    )

    bias_unified = _eval_bias_unified(unified, bias_test_loader, bias_num_classes, device)

    if args.bias_checkpoint:
        bias_model = RoBERTaClassifier(
            task_type=bias_task_type, num_classes=bias_num_classes,
            model_name=cfg["model"]["name"],
        ).to(device)
        bias_model.load_state_dict(
            torch.load(args.bias_checkpoint, map_location=device, weights_only=True)
        )
        bias_base = bias_evaluate(bias_model, bias_test_loader, device)
        bias_baseline = {"macro_f1": bias_base["f1_macro"], "accuracy": bias_base["accuracy"]}
    else:
        bias_baseline = {k: float("nan") for k in bias_unified}

    results["Bias"] = {
        metric: (bias_baseline[metric], bias_unified[metric]) for metric in bias_unified
    }

    # ── Emotion ───────────────────────────────────────────────────────────────
    print("Evaluating emotion task …")
    em_cfg = EmotionalFramingConfig()
    em_cfg.magpie_data_dir = cfg["data"].get("magpie_data_dir", em_cfg.magpie_data_dir)
    em_cfg.hf_cache_dir    = cfg["data"].get("hf_cache_dir", em_cfg.hf_cache_dir)
    em_cfg.max_seq_length  = cfg["data"]["max_len"]
    em_cfg.seed            = cfg["data"]["seed"]

    dataset_dict = emotion_load_and_split(em_cfg)

    def _tokenize(batch):
        enc = tokenizer(
            batch["text"], max_length=em_cfg.max_seq_length,
            padding="max_length", truncation=True,
        )
        enc["labels"] = [list(map(float, lv)) for lv in batch["labels"]]
        return enc

    tokenized = dataset_dict.map(_tokenize, batched=True, remove_columns=["text", "source"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    em_test_torch = EmotionTorchDataset(tokenized["test"])
    em_dev_torch  = EmotionTorchDataset(tokenized["dev"])

    # Tune threshold on dev set for unified model
    dev_loader     = DataLoader(em_dev_torch, batch_size=64, shuffle=False, collate_fn=_emotion_collate)
    dev_logits_list, dev_labels_list = [], []
    with torch.no_grad():
        for batch in dev_loader:
            kwargs = _batch_to_kwargs(TASK_EMOTION, batch, device)
            out    = unified(**kwargs)
            dev_logits_list.append(out["emotion_logits"].cpu().numpy())
            dev_labels_list.append(batch["emotion_labels"].numpy())
    dev_logits = np.concatenate(dev_logits_list, axis=0)
    dev_labels = np.concatenate(dev_labels_list, axis=0)
    unified_threshold = tune_threshold(dev_logits, dev_labels)

    em_unified = _eval_emotion_unified(unified, em_test_torch, unified_threshold, device)

    if args.emotion_checkpoint:
        from models.emotion.model import EmotionalFramingClassifier
        from transformers import RobertaConfig
        model_config = RobertaConfig.from_pretrained(args.emotion_checkpoint)
        em_baseline_model = EmotionalFramingClassifier.from_pretrained(
            args.emotion_checkpoint, config=model_config,
        ).to(device)
        threshold_path = Path(args.emotion_checkpoint) / "threshold.txt"
        em_threshold = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.5
        em_base = _eval_emotion_baseline(em_baseline_model, em_test_torch, em_threshold, device)
        em_baseline = em_base
    else:
        em_baseline = {k: float("nan") for k in em_unified}

    results["Emotion"] = {
        metric: (em_baseline[metric], em_unified[metric]) for metric in em_unified
    }

    # ── Print and save ────────────────────────────────────────────────────────
    _print_table(results)

    out_path = Path(args.unified_checkpoint).parent / "eval_comparison.json"
    flat_results = {
        task: {m: {"baseline": b, "unified": u} for m, (b, u) in metrics.items()}
        for task, metrics in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(flat_results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
