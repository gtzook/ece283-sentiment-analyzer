"""
eval.py — evaluation for the epistemic certainty model.

Evaluates a trained checkpoint on the held-out test split (document-level,
same seed/fractions as train.py) and reports:

  Sentence head
    • Per-class precision / recall / F1
    • Macro-F1  (primary metric, same as training checkpoint criterion)
    • Cohen's κ (model predictions vs. true labels)
    • 3×3 confusion matrix (saved as PNG)
    • ECE       (Expected Calibration Error, 15 bins)

  Token head
    • Binary macro-F1 / precision / recall on news test sentences

  Transfer test (BABE)
    • Skipped gracefully when epistemic label column is absent

Usage:
    python -m models.epistemic.eval \\
        --checkpoint runs/20260530_071702/best.pt

    python -m models.epistemic.eval \\
        --checkpoint runs/20260530_071702/best.pt \\
        --config models/epistemic/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.epistemic.data import LABEL_NAMES, SentDataset, TokenDataset
from models.epistemic.model import EpistemicModel
from models.epistemic.train import load_all_data


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error.

    probs:  (N, 3) softmax probabilities
    labels: (N,)   true integer labels
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf  = confidences[mask].mean()
        avg_acc   = correct[mask].mean()
        ece      += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(labels))


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    save_path: Path,
    title: str = "Confusion matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="True",
        title=title,
    )
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_sent_inference(
    model: EpistemicModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (true_labels, pred_labels, all_probs) arrays."""
    model.eval()
    all_true, all_pred, all_probs = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred  = model.predict(batch["input_ids"], batch["attention_mask"])
        all_true.extend(batch["sent_label"].cpu().numpy())
        all_pred.extend(pred["label"].cpu().numpy())
        all_probs.append(pred["sent_probs"].cpu().numpy())
    return (
        np.array(all_true),
        np.array(all_pred),
        np.concatenate(all_probs, axis=0),
    )


@torch.no_grad()
def run_token_inference(
    model: EpistemicModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (true_flat, pred_flat) ignoring -100 positions."""
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        batch    = {k: v.to(device) for k, v in batch.items()}
        out      = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_labels=batch["token_labels"],
        )
        flat_true = batch["token_labels"].view(-1).cpu().numpy()
        flat_pred = out["token_logits"].view(-1, 2).argmax(-1).cpu().numpy()
        mask      = flat_true != -100
        all_true.append(flat_true[mask])
        all_pred.append(flat_pred[mask])
    return np.concatenate(all_true), np.concatenate(all_pred)


# ── BABE transfer test ────────────────────────────────────────────────────────

def eval_babe(
    model: EpistemicModel,
    tokenizer,
    babe_path: str,
    device: torch.device,
    max_len: int,
) -> dict | None:
    """
    Evaluate on BABE sentences if an epistemic label column is present.
    Returns a metrics dict, or None if the column is absent.
    """
    import csv

    babe_path = Path(babe_path)
    if not babe_path.exists():
        print(f"  BABE file not found at {babe_path} — skipping transfer test.")
        return None

    LABEL_COL = "epistemic_label"
    texts, labels = [], []
    with open(babe_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if LABEL_COL not in (reader.fieldnames or []):
            print(f"  BABE CSV has no '{LABEL_COL}' column — "
                  "run LLM-ensemble labeling first. Skipping transfer test.")
            return None
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row[LABEL_COL]))

    from models.epistemic.data import SentExample
    examples = [SentExample(text=t, label=l, source="babe") for t, l in zip(texts, labels)]
    ds     = SentDataset(examples, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    true, pred, probs = run_sent_inference(model, loader, device)
    return {
        "babe_macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "babe_kappa":    float(cohen_kappa_score(true, pred)),
        "babe_n":        len(true),
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(checkpoint: Path, cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc     = cfg["model"]
    dc     = cfg["data"]
    max_len = dc.get("max_len", 128)

    tokenizer = AutoTokenizer.from_pretrained(mc["name"])

    # ── Re-derive test splits (deterministic; same seed/fractions as train.py) ─
    print("Loading data and deriving test split …")
    splits = load_all_data(cfg)

    sent_test_ds     = SentDataset(splits["sent_test"],     tokenizer, max_len)
    tok_news_test_ds = TokenDataset(splits["tok_news_test"], tokenizer, max_len)

    sent_loader = DataLoader(sent_test_ds,     batch_size=32, shuffle=False)
    tok_loader  = DataLoader(tok_news_test_ds, batch_size=32, shuffle=False)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint}")
    model = EpistemicModel(model_name=mc["name"]).to(device)
    # strict=False: checkpoint may include sent_loss_fn.weight (class-weight buffer
    # registered by CrossEntropyLoss during training) not needed at inference.
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=False,
    )

    # ── Sentence head evaluation ───────────────────────────────────────────────
    print("Running sentence-head inference …")
    true, pred, probs = run_sent_inference(model, sent_loader, device)

    prec, rec, f1_per, _ = precision_recall_fscore_support(
        true, pred, labels=[0, 1, 2], zero_division=0
    )
    macro_f1 = float(f1_score(true, pred, average="macro", zero_division=0))
    kappa    = float(cohen_kappa_score(true, pred))
    ece      = compute_ece(probs, true)
    cm       = confusion_matrix(true, pred, labels=[0, 1, 2])

    # ── Token head evaluation ──────────────────────────────────────────────────
    print("Running token-head inference …")
    tok_true, tok_pred = run_token_inference(model, tok_loader, device)
    tok_f1  = float(f1_score(tok_true, tok_pred, average="macro",  zero_division=0))
    tok_p   = float(f1_score(tok_true, tok_pred, average="binary", pos_label=1, zero_division=0))
    tok_r_  = float(precision_recall_fscore_support(
        tok_true, tok_pred, average="binary", pos_label=1, zero_division=0
    )[1])

    # ── BABE transfer test ─────────────────────────────────────────────────────
    print("Checking BABE transfer test …")
    babe_metrics = eval_babe(
        model, tokenizer,
        babe_path="/mldata/ece283-sentiment-analyzer/10_BABE/preprocessed.csv",
        device=device,
        max_len=max_len,
    )

    # ── Assemble results ───────────────────────────────────────────────────────
    metrics = {
        "checkpoint":      str(checkpoint),
        "n_test_sent":     int(len(true)),
        "n_test_tok":      int(len(tok_true)),
        # sentence head
        "sent_macro_f1":   macro_f1,
        "sent_kappa":      kappa,
        "sent_ece":        ece,
        "sent_per_class":  {
            LABEL_NAMES[i]: {
                "precision": float(prec[i]),
                "recall":    float(rec[i]),
                "f1":        float(f1_per[i]),
            }
            for i in range(3)
        },
        # token head
        "tok_macro_f1":    tok_f1,
        "tok_cue_f1":      tok_p,
        "tok_cue_recall":  tok_r_,
    }
    if babe_metrics:
        metrics.update(babe_metrics)

    # ── Print report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("SENTENCE HEAD — test split")
    print("=" * 58)
    print(f"  n = {metrics['n_test_sent']}")
    print(f"  Macro-F1   : {macro_f1:.4f}")
    print(f"  Cohen's κ  : {kappa:.4f}")
    print(f"  ECE (15bin): {ece:.4f}")
    print(f"\n  {'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*34}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<12} {prec[i]:>6.3f} {rec[i]:>6.3f} {f1_per[i]:>6.3f}")

    print("\n" + "=" * 58)
    print("TOKEN HEAD — news test split")
    print("=" * 58)
    print(f"  n tokens (non-pad) = {metrics['n_test_tok']}")
    print(f"  Macro-F1           : {tok_f1:.4f}")
    print(f"  Cue F1 (pos=1)     : {tok_p:.4f}")
    print(f"  Cue Recall (pos=1) : {tok_r_:.4f}")

    if babe_metrics:
        print("\n" + "=" * 58)
        print("BABE TRANSFER TEST")
        print("=" * 58)
        print(f"  n = {babe_metrics['babe_n']}")
        print(f"  Macro-F1  : {babe_metrics['babe_macro_f1']:.4f}")
        print(f"  Cohen's κ : {babe_metrics['babe_kappa']:.4f}")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    run_dir = checkpoint.parent
    metrics_path = run_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    cm_path = run_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, LABEL_NAMES, cm_path, title="Test-split confusion matrix")
    print(f"Confusion matrix saved to {cm_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the epistemic certainty model")
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to .pt checkpoint file (e.g. runs/20260530_071702/best.pt)",
    )
    parser.add_argument(
        "--config", default="models/epistemic/config.yaml", type=Path,
        help="Path to config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(args.checkpoint, cfg)


if __name__ == "__main__":
    main()
