#!/usr/bin/env python3
"""Generates explore_seed42.ipynb in the project root.

Run from the project root:
    python create_explore_notebook.py
"""
import json
import uuid
from pathlib import Path


def cell_id():
    return uuid.uuid4().hex[:8]


def md(source):
    return {"cell_type": "markdown", "id": cell_id(), "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ── Cell sources (written as plain Python strings) ────────────────────────────

SETUP = """\
import sys
import json
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
plt.style.use("ggplot")

CKPT        = ROOT / "runs/unified/seed42-5ep/best.pt"
CONFIG      = ROOT / "models/unified/config.yaml"
HISTORY     = ROOT / "runs/unified/seed42-5ep/history.json"
STEP_LOSSES = ROOT / "runs/unified/seed42-5ep/step_losses.jsonl"

for p in [CKPT, CONFIG, HISTORY, STEP_LOSSES]:
    tag = "OK     " if p.exists() else "MISSING"
    print(f"{tag} {p}")
"""

PLOT_EPOCH = """\
with open(HISTORY) as f:
    history = json.load(f)

df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A — Validation loss per task
ax = axes[0]
ax.plot(df["epoch"], df["epistemic"], marker="o", label="Epistemic", color="#4C72B0")
ax.plot(df["epoch"], df["bias"],      marker="s", label="Bias",      color="#DD8452")
ax.plot(df["epoch"], df["emotion"],   marker="^", label="Emotion",   color="#55A868")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.set_title("Validation Loss per Task")
ax.legend()

# Plot B — Validation macro-F1 + composite
ax = axes[1]
ax.plot(df["epoch"], df["epistemic_macro_f1"], marker="o", label="Epistemic F1", color="#4C72B0")
ax.plot(df["epoch"], df["bias_macro_f1"],      marker="s", label="Bias F1",      color="#DD8452")
ax.plot(df["epoch"], df["emotion_macro_f1"],   marker="^", label="Emotion F1",   color="#55A868")
ax.plot(df["epoch"], df["composite"],          marker="D", label="Composite",    color="#8172B3", linewidth=2)

best_idx = df["composite"].idxmax()
best_row = df.iloc[best_idx]
ax.plot(best_row["epoch"], best_row["composite"],
        marker="*", color="gold", markersize=18, zorder=5,
        label=f"Best composite ({best_row['composite']:.3f})")

ax.set_xlabel("Epoch")
ax.set_ylabel("Macro F1")
ax.set_title("Validation Macro-F1 + Composite")
ax.legend(fontsize=9)
ax.set_ylim(0, 1)

plt.suptitle("Training Performance — seed42-5ep", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("runs/unified/seed42-5ep/training_performance.png", bbox_inches="tight")
plt.show()
"""

PLOT_STEPS = """\
print("Loading step losses (~14 MB)...")
steps_df = pd.read_json(STEP_LOSSES, lines=True)
print(f"Loaded {len(steps_df):,} rows across {steps_df['epoch'].nunique()} epochs")

# Downsample to every 50th step (~3,835 points per task) for responsive rendering
sampled = steps_df.iloc[::50].copy()
task_colors = {"epistemic": "#4C72B0", "bias": "#DD8452", "emotion": "#55A868"}
epoch_starts = steps_df.groupby("epoch")["step"].min()

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for ax, task in zip(axes, ["epistemic", "bias", "emotion"]):
    task_data = sampled[sampled["task"] == task]
    ax.plot(task_data["step"], task_data["loss"],
            color=task_colors[task], alpha=0.8, linewidth=0.9)
    for ep, start in epoch_starts.items():
        ax.axvline(start, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("Loss")
    ax.set_title(f"{task.capitalize()} Training Loss")

# Epoch labels on the top subplot
top_ax = axes[0]
ylim = top_ax.get_ylim()
for ep, start in epoch_starts.items():
    top_ax.text(start + 400, ylim[0] + (ylim[1] - ylim[0]) * 0.88,
                f"Ep {ep}", fontsize=8, color="dimgray")

axes[-1].set_xlabel("Global Step")
plt.suptitle("Step-Level Training Loss  (every 50th step sampled)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("runs/unified/seed42-5ep/step_losses_plot.png", bbox_inches="tight")
plt.show()
"""

LOAD_MODEL = """\
from models.unified.predict import load_unified_predictor, predict_all

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print("\\nLoading checkpoint (481 MB RoBERTa-base + 3 task heads)...")
predictor = load_unified_predictor(
    checkpoint=str(CKPT),
    config=str(CONFIG),
    device=device,
)
n_params = sum(p.numel() for p in predictor.model.parameters()) / 1e6
print(f"Model loaded — {n_params:.1f}M parameters on {device}")
"""

RUN_INFER = """\
TEXTS = [
    # Epistemic variety
    "The drug may reduce symptoms in some patients.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Scientists believe the universe could be infinite.",
    # Bias variety
    "Left-wing radicals push dangerous open-border agenda.",
    "City council votes to approve new transit budget.",
    "Big government spending plan threatens economic freedom.",
    # Emotional variety
    "Outrage erupts as children suffer under failed policies.",
    "Breakthrough discovery offers hope for millions battling disease.",
    "Another year of drought leaves farmers desperate and despairing.",
    "The new partnership is expected to deliver lasting community benefits.",
    # Mixed
    "Officials fear the rumored cuts could devastate local schools.",
    "The report was released on Tuesday by the Department of Labor.",
]

CATEGORIES = [
    "Hedged scientific", "Asserted fact", "Speculative",
    "Biased (left attack)", "Neutral headline", "Biased (right attack)",
    "Anger / fear", "Joy / optimism", "Sadness / pessimism", "Trust / anticipation",
    "Mixed hedged+emotional", "Neutral factual",
]

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]

print("Running inference on 12 texts...")
results = predict_all(predictor, TEXTS)
print("Done.")
"""

SHOW_TABLE = """\
rows = []
for cat, text, r in zip(CATEGORIES, TEXTS, results):
    ep = r["epistemic"]
    bi = r["bias"]
    em = r["emotion"]
    active = [e for e in EMOTIONS if em[e] == 1]
    rows.append({
        "Category":        cat,
        "Text":            (text[:50] + "...") if len(text) > 50 else text,
        "Epistemic":       ep["label_name"],
        "Uncertainty":     f"{ep['uncertainty_score']:.3f}",
        "Bias":            bi["prediction"],
        "Bias Conf":       f"{bi['confidence']:.3f}",
        "Active Emotions": ", ".join(active) if active else "—",
    })

df_out = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 65)
pd.set_option("display.width", 220)
display(df_out)
"""

HEATMAP = """\
score_matrix = np.array([
    [r["emotion"]["scores"][e] for e in EMOTIONS]
    for r in results
])

row_labels = [f"{i+1}. {CATEGORIES[i]}" for i in range(len(TEXTS))]

fig, ax = plt.subplots(figsize=(13, 7))
im = ax.imshow(score_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(EMOTIONS)))
ax.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=10)
ax.set_yticks(range(len(TEXTS)))
ax.set_yticklabels(row_labels, fontsize=10)

for i in range(len(TEXTS)):
    for j in range(len(EMOTIONS)):
        val = score_matrix[i, j]
        text_color = "white" if val > 0.65 else "black"
        weight = "bold" if val >= 0.5 else "normal"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight=weight)

plt.colorbar(im, ax=ax, label="Sigmoid score", shrink=0.8)
ax.set_title("Emotion Probability Heatmap  (bold = predicted active \\u2265 0.5)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("runs/unified/seed42-5ep/emotion_heatmap.png", bbox_inches="tight")
plt.show()
"""

BIAS_CHART = """\
bias_p_biased = [r["bias"]["probabilities"].get("biased", 0.0) for r in results]
bias_preds    = [r["bias"]["prediction"] for r in results]
bar_colors    = ["#d62728" if p == "biased" else "#2ca02c" for p in bias_preds]
row_labels    = [f"{i+1}. {CATEGORIES[i]}" for i in range(len(TEXTS))]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(row_labels, bias_p_biased, color=bar_colors, alpha=0.85)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision boundary")
ax.set_xlim(0, 1.18)
ax.set_xlabel("P(biased)")
ax.set_title("Political Bias — Prediction Confidence", fontsize=13, fontweight="bold")

for bar, prob in zip(bars, bias_p_biased):
    ax.text(prob + 0.015, bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}", va="center", fontsize=9)

red_p   = mpatches.Patch(color="#d62728", alpha=0.85, label="Predicted: biased")
green_p = mpatches.Patch(color="#2ca02c", alpha=0.85, label="Predicted: not biased")
ax.legend(handles=[red_p, green_p], fontsize=10, loc="lower right")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("runs/unified/seed42-5ep/bias_confidence.png", bbox_inches="tight")
plt.show()
"""

EPISTEMIC_CHART = """\
unc_scores  = [r["epistemic"]["uncertainty_score"] for r in results]
ep_labels   = [r["epistemic"]["label_name"] for r in results]
label_color = {"asserted": "#4C72B0", "hedged": "#DD8452", "speculative": "#d62728"}
bar_colors  = [label_color[l] for l in ep_labels]
row_labels  = [f"{i+1}. {CATEGORIES[i]}" for i in range(len(TEXTS))]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(row_labels, unc_scores, color=bar_colors, alpha=0.85)
ax.set_xlim(0, 1.22)
ax.set_xlabel("Uncertainty Score  (0 = asserted, 0.5 = hedged, 1 = speculative)")
ax.set_title("Epistemic Uncertainty Score by Text", fontsize=13, fontweight="bold")

for bar, score, lbl in zip(bars, unc_scores, ep_labels):
    ax.text(score + 0.015, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}  ({lbl})", va="center", fontsize=9)

patches = [mpatches.Patch(color=c, alpha=0.85, label=l.capitalize())
           for l, c in label_color.items()]
ax.legend(handles=patches, fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("runs/unified/seed42-5ep/epistemic_uncertainty.png", bbox_inches="tight")
plt.show()
"""

# ── Assemble notebook ──────────────────────────────────────────────────────────

cells = [
    md(
        "# ECE283 — seed42-5ep Model Exploration\n\n"
        "Loads `runs/unified/seed42-5ep/best.pt` — RoBERTa-base fine-tuned jointly on three tasks"
        " over 5 epochs (~9.75 h, 191 760 steps).  Final composite val F1: **0.543**\n"
        "(Epistemic 0.573 · Bias 0.670 · Emotion 0.386)\n\n"
        "**Sections**\n"
        "1. Training performance: per-task loss + macro-F1 over epochs, step-level loss curves\n"
        "2. Example inferences: 12 diverse texts spanning all three tasks\n"
        "3. Visualizations: emotion probability heatmap, bias confidence chart,"
        " epistemic uncertainty chart\n\n"
        "> Run from the project root so `models/` and `src/` imports resolve."
    ),
    code(SETUP),
    md("## 1 — Training Performance"),
    code(PLOT_EPOCH),
    code(PLOT_STEPS),
    md("## 2 — Example Inferences"),
    code(LOAD_MODEL),
    code(RUN_INFER),
    code(SHOW_TABLE),
    md("## 3 — Visualizations"),
    code(HEATMAP),
    code(BIAS_CHART),
    code(EPISTEMIC_CHART),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("explore_seed42.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Created {out}  ({out.stat().st_size:,} bytes)")
