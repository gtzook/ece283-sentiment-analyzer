#!/usr/bin/env python3
"""
generate_confusion_matrices.py — confusion matrices for the "Frozen",
"Task weights", and "TF-IDF + LR Baseline" runs shown in the report figures.

For the two unified checkpoints (Frozen = unified_exp7_staged,
Task weights = unified_exp4_weights) this loads the trained model, runs it
over each task's held-out test split, and saves a confusion matrix (PNG + CSV)
per task — a 3x3 for epistemic certainty, a 2x2 for political bias, and one
binary (present/absent) matrix per emotion label for the multi-label emotion
task, mirroring the layout already produced for the TF-IDF + LR baseline at
baselines/tfidf_lr/confusion_matrices/.

The TF-IDF + LR baseline matrices already exist (produced by
baselines/tfidf_lr/run_baseline.py on the same seed=42 test split) — this
script copies them alongside the unified ones so all three runs land in one
place: figures/confusion_matrices/<model_id>/.

Usage:
    python generate_confusion_matrices.py
"""

from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, ConcatDataset
from transformers import RobertaTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified.model import UnifiedModel, TASK_EPISTEMIC, TASK_BIAS, TASK_EMOTION
from models.unified.train import (
    EmotionTorchDataset, _emotion_collate, _batch_to_kwargs, _make_bias_transform,
)
from models.epistemic.data import LABEL_NAMES as EPISTEMIC_LABELS, SentDataset
from models.epistemic.train import load_all_data
from models.emotion.config import EmotionalFramingConfig, EMOTION_LABELS
from models.emotion.data import load_and_split as emotion_load_and_split
from models.political_bias.train_baseline import collate_fn as bias_collate_fn
from src.data.dataset import MAGPIEDataset
from src.data.splits import stratified_split
from src.data.registry import TaskType

BIAS_LABELS = ["neutral", "biased"]

OUT_ROOT = PROJECT_ROOT / "figures" / "confusion_matrices"

RUNS = [
    # {
    #     "model_id": "unified_staged",
    #     "display":  "Frozen",
    #     "config":   "models/unified/config_exp7_staged.yaml",
    #     "checkpoint": "/mldata/ece283-sentiment-analyzer/runs/unified_exp7_staged/20260602_043318/best.pt",
    # },
    # {
    #     "model_id": "unified_weights",
    #     "display":  "Task weights",
    #     "config":   "models/unified/config_exp4_task_weights.yaml",
    #     "checkpoint": "/mldata/ece283-sentiment-analyzer/runs/unified_exp4_weights/20260602_010857/best.pt",
    # },
    {
        "model_id": "unified_regularization",
        "display":  "Regularized",
        "config":   "models/unified/config_exp2regularization.yaml",
        "checkpoint": "/mldata/ece283-sentiment-analyzer/runs/unified_exp2_reg/20260602_011413/best.pt",
    },
]

TFIDF_SRC = PROJECT_ROOT / "baselines" / "tfidf_lr" / "confusion_matrices"
TFIDF_DST_ID = "tfidf_lr"


# ── Plotting helpers (mirrors baselines/tfidf_lr/run_baseline.py style) ──────

def save_cm_png(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="True", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=max(6, 10 - n))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cm_csv(cm: np.ndarray, labels: list[str], path: Path) -> None:
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.index.name = "true\\pred"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def write_cm(cm: np.ndarray, labels: list[str], out_dir: Path, name: str, title: str) -> None:
    save_cm_png(cm, labels, out_dir / f"{name}.png", title)
    save_cm_csv(cm, labels, out_dir / f"{name}.csv")
    print(f"  wrote {out_dir / name}.{{png,csv}}")


# ── Per-task prediction collection ───────────────────────────────────────────

@torch.no_grad()
def collect_epistemic(model: UnifiedModel, sent_test: SentDataset, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(sent_test, batch_size=64, shuffle=False)
    preds, labels = [], []
    for batch in loader:
        kwargs = _batch_to_kwargs(TASK_EPISTEMIC, batch, device)
        out    = model(**kwargs)
        preds.extend(out["sent_logits"].argmax(-1).cpu().numpy().tolist())
        labels.extend(batch["sent_label"].numpy().tolist())
    return np.array(labels), np.array(preds)


@torch.no_grad()
def collect_bias(model: UnifiedModel, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        kwargs = _batch_to_kwargs(TASK_BIAS, batch, device)
        out    = model(**kwargs)
        preds.extend(out["bias_logits"].argmax(-1).cpu().numpy().tolist())
        labels.extend(batch["labels"].numpy().tolist())
    return np.array(labels), np.array(preds)


@torch.no_grad()
def collect_emotion(model: UnifiedModel, dataset: EmotionTorchDataset,
                    device) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce the multi-label emotion task to a single predicted/true label per
    example: predicted = the emotion the model is most confident about
    (argmax over logits); true = argmax over the (mostly single-hot, ~98% of
    test examples carry exactly one Plutchik label) ground-truth vector.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=_emotion_collate)
    logits_list, labels_list = [], []
    for batch in loader:
        kwargs = _batch_to_kwargs(TASK_EMOTION, batch, device)
        out    = model(**kwargs)
        logits_list.append(out["emotion_logits"].cpu().numpy())
        labels_list.append(batch["emotion_labels"].numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return labels.argmax(axis=1), logits.argmax(axis=1)


# ── Per-run pipeline ──────────────────────────────────────────────────────────

def run_for_checkpoint(spec: dict, device: torch.device) -> None:
    model_id = spec["model_id"]
    out_dir  = OUT_ROOT / model_id
    print(f"\n=== {spec['display']} ({model_id}) ===")

    with open(PROJECT_ROOT / spec["config"]) as f:
        cfg = yaml.safe_load(f)

    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model"]["name"])

    bias_task_type, bias_num_classes = TaskType.BINARY_CLS, 2

    model = UnifiedModel(
        model_name         = cfg["model"]["name"],
        dropout            = cfg["model"].get("dropout", 0.1),
        lambda_token       = cfg["model"].get("lambda_token", 0.3),
        bias_task_type     = bias_task_type,
        bias_num_classes   = bias_num_classes,
        emotion_num_labels = cfg["model"].get("emotion_num_labels", 11),
    ).to(device)
    model.load_state_dict(
        torch.load(spec["checkpoint"], map_location=device, weights_only=True),
        strict=False,
    )
    model.eval()
    print(f"Loaded checkpoint from {spec['checkpoint']}")

    # ── Epistemic (3-class) ───────────────────────────────────────────────────
    print("Epistemic …")
    ep_data   = load_all_data(cfg)
    sent_test = SentDataset(ep_data["sent_test"], tokenizer, max_len=cfg["data"]["max_len"])
    y_true, y_pred = collect_epistemic(model, sent_test, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EPISTEMIC_LABELS))))
    write_cm(cm, EPISTEMIC_LABELS, out_dir, "epistemic",
             f"{spec['display']} — epistemic certainty (test)")

    # ── Bias (binary) ─────────────────────────────────────────────────────────
    print("Bias …")
    specs = cfg["model"]["bias_datasets"]
    test_subsets = []
    for bspec in specs:
        ds_id, label_col = bspec["dataset_id"], bspec["label_col"]
        transform = _make_bias_transform(label_col, bspec.get("remap"))
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
            full_ds, label_col=label_col,
            train_frac = cfg["data"].get("train_frac", 0.80),
            val_frac   = cfg["data"].get("val_frac",   0.10),
            seed       = cfg["data"]["seed"],
        )
        test_subsets.append(test_ds)

    primary_label_col = specs[0]["label_col"]
    bias_loader = DataLoader(
        ConcatDataset(test_subsets), batch_size=64, shuffle=False, num_workers=4,
        collate_fn=bias_collate_fn(primary_label_col),
    )
    y_true, y_pred = collect_bias(model, bias_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    write_cm(cm, BIAS_LABELS, out_dir, "bias",
             f"{spec['display']} — political bias (test)")

    # ── Emotion (single predicted label per example, one combined matrix) ────
    print("Emotion …")
    em_cfg = EmotionalFramingConfig()
    em_cfg.magpie_data_dir = cfg["data"].get("magpie_data_dir", em_cfg.magpie_data_dir)
    em_cfg.hf_cache_dir    = cfg["data"].get("hf_cache_dir", em_cfg.hf_cache_dir)
    em_cfg.max_seq_length  = cfg["data"]["max_len"]
    em_cfg.seed            = cfg["data"]["seed"]

    dataset_dict = emotion_load_and_split(em_cfg)

    def _tokenize(batch):
        enc = tokenizer(batch["text"], max_length=em_cfg.max_seq_length,
                        padding="max_length", truncation=True)
        enc["labels"] = [list(map(float, lv)) for lv in batch["labels"]]
        return enc

    tokenized = dataset_dict.map(_tokenize, batched=True, remove_columns=["text", "source"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    em_test_torch = EmotionTorchDataset(tokenized["test"])

    y_true, y_pred = collect_emotion(model, em_test_torch, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_LABELS))))
    write_cm(cm, EMOTION_LABELS, out_dir, "emotion",
             f"{spec['display']} — emotion (test, single predicted label)")


# ── TF-IDF + LR baseline ──────────────────────────────────────────────────────
# task1 (epistemic) and task2 (bias) are already single-label, so their existing
# matrices (produced by baselines/tfidf_lr/run_baseline.py on the same seed=42
# split) are copied as-is. task3 (emotion) was fit as an 11-way One-vs-Rest
# multi-label model with per-label binary matrices; here it's collapsed to a
# single combined matrix by taking the most-confident label as the prediction
# (mirroring the argmax reduction used for the unified models), which requires
# refitting the OvR classifier on top of the saved TF-IDF vectorizer.

TFIDF_EMOTION_C = 10  # selected by CV in run_baseline.py (see cv_logs/task3_emotion.json)
TFIDF_SEED = 42


def copy_tfidf_matrices() -> None:
    dst = OUT_ROOT / TFIDF_DST_ID
    dst.mkdir(parents=True, exist_ok=True)
    if not TFIDF_SRC.is_dir():
        print(f"\nWARNING: {TFIDF_SRC} not found — run baselines/tfidf_lr/run_baseline.py first.")
        return
    n = 0
    for f in TFIDF_SRC.glob("*"):
        if f.name.startswith("task3_emotion_"):
            continue  # superseded by the single combined matrix below
        shutil.copy2(f, dst / f.name)
        n += 1
    print(f"  copied {n} files (task1 + task2) from {TFIDF_SRC} → {dst}")


def tfidf_emotion_matrix() -> None:
    """Refit the saved OvR TF-IDF+LR emotion model and collapse its per-label
    probabilities to a single most-confident predicted label per example."""
    vec_path = PROJECT_ROOT / "baselines" / "tfidf_lr" / "vectorizers" / "task3_emotion.pkl"
    with open(vec_path, "rb") as f:
        vec: TfidfVectorizer = pickle.load(f)

    em_cfg = EmotionalFramingConfig()
    em_cfg.seed = TFIDF_SEED
    dataset_dict = emotion_load_and_split(em_cfg)

    def _extract(split):
        ds = dataset_dict[split]
        return list(ds["text"]), np.array(list(ds["labels"]), dtype=np.int32)

    tr_texts, y_tr = _extract("train")
    dev_texts, y_dev = _extract("dev")
    te_texts, y_te = _extract("test")

    X_tr_all = vec.transform(tr_texts + dev_texts)   # vectorizer was fit on train+dev
    X_te     = vec.transform(te_texts)
    y_tr_all = np.vstack([y_tr, y_dev])

    clf = OneVsRestClassifier(
        LogisticRegression(C=TFIDF_EMOTION_C, max_iter=1000, class_weight="balanced",
                           solver="lbfgs", random_state=TFIDF_SEED),
        n_jobs=-1,
    )
    clf.fit(X_tr_all, y_tr_all)

    probs  = clf.predict_proba(X_te)
    y_pred = probs.argmax(axis=1)
    y_true = y_te.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_LABELS))))
    write_cm(cm, EMOTION_LABELS, OUT_ROOT / TFIDF_DST_ID, "emotion",
             "TF-IDF + LR baseline — emotion (test, single predicted label)")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for spec in RUNS:
        run_for_checkpoint(spec, device)

    print(f"\n=== TF-IDF + LR Baseline (tfidf_lr) ===")
    copy_tfidf_matrices()
    tfidf_emotion_matrix()

    print(f"\nAll confusion matrices written under {OUT_ROOT}")


if __name__ == "__main__":
    main()
