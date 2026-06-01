"""
Run from the project root:
    python3 explore_seed42.py

Outputs:
    - Printed inference table
    - runs/unified/seed42-5ep/training_performance.png
    - runs/unified/seed42-5ep/step_losses_plot.png
    - runs/unified/seed42-5ep/emotion_heatmap.png
    - runs/unified/seed42-5ep/bias_confidence.png
    - runs/unified/seed42-5ep/epistemic_uncertainty.png
"""
import sys
import json
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # headless — saves to file instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
plt.style.use("ggplot")

CKPT        = ROOT / "runs/unified/seed42-5ep/best.pt"
CONFIG      = ROOT / "models/unified/config.yaml"
HISTORY     = ROOT / "runs/unified/seed42-5ep/history.json"
STEP_LOSSES = ROOT / "runs/unified/seed42-5ep/step_losses.jsonl"
OUT_DIR     = ROOT / "runs/unified/seed42-5ep"

print("=== Path check ===")
for p in [CKPT, CONFIG, HISTORY, STEP_LOSSES]:
    tag = "OK     " if p.exists() else "MISSING"
    print(f"  {tag} {p}")

# ── 1. Training performance ────────────────────────────────────────────────────

print("\n=== Training performance ===")

with open(HISTORY) as f:
    history = json.load(f)
df = pd.DataFrame(history)
print(df[["epoch", "epistemic_macro_f1", "bias_macro_f1", "emotion_macro_f1", "composite"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(df["epoch"], df["epistemic"], marker="o", label="Epistemic", color="#4C72B0")
ax.plot(df["epoch"], df["bias"],      marker="s", label="Bias",      color="#DD8452")
ax.plot(df["epoch"], df["emotion"],   marker="^", label="Emotion",   color="#55A868")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.set_title("Validation Loss per Task")
ax.legend()

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
out = OUT_DIR / "training_performance.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 2. Step-level losses ───────────────────────────────────────────────────────

print("\n=== Step-level losses ===")
print("Loading step_losses.jsonl (~14 MB)...")
steps_df = pd.read_json(STEP_LOSSES, lines=True)
print(f"Loaded {len(steps_df):,} rows across {steps_df['epoch'].nunique()} epochs")

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

top_ax = axes[0]
ylim = top_ax.get_ylim()
for ep, start in epoch_starts.items():
    top_ax.text(start + 400, ylim[0] + (ylim[1] - ylim[0]) * 0.88,
                f"Ep {ep}", fontsize=8, color="dimgray")

axes[-1].set_xlabel("Global Step")
plt.suptitle("Step-Level Training Loss  (every 50th step sampled)", fontsize=13, fontweight="bold")
plt.tight_layout()
out = OUT_DIR / "step_losses_plot.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 3. Load model ──────────────────────────────────────────────────────────────

print("\n=== Model ===")
from models.unified.predict import load_unified_predictor, predict_all

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print("Loading checkpoint...")
predictor = load_unified_predictor(
    checkpoint=str(CKPT),
    config=str(CONFIG),
    device=device,
)
n_params = sum(p.numel() for p in predictor.model.parameters()) / 1e6
print(f"Loaded — {n_params:.1f}M parameters")

# ── 4. Inference ───────────────────────────────────────────────────────────────

TEXTS = [
    "The drug may reduce symptoms in some patients.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Scientists believe the universe could be infinite.",
    "Left-wing radicals push dangerous open-border agenda.",
    "City council votes to approve new transit budget.",
    "Big government spending plan threatens economic freedom.",
    "Outrage erupts as children suffer under failed policies.",
    "Breakthrough discovery offers hope for millions battling disease.",
    "Another year of drought leaves farmers desperate and despairing.",
    "The new partnership is expected to deliver lasting community benefits.",
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

print("\n=== Running inference on 12 texts ===")
results = predict_all(predictor, TEXTS)

rows = []
for cat, text, r in zip(CATEGORIES, TEXTS, results):
    ep = r["epistemic"]
    bi = r["bias"]
    em = r["emotion"]
    active = [e for e in EMOTIONS if em[e] == 1]
    rows.append({
        "Category":        cat,
        "Text":            (text[:48] + "...") if len(text) > 48 else text,
        "Epistemic":       ep["label_name"],
        "Uncertainty":     f"{ep['uncertainty_score']:.3f}",
        "Bias":            bi["prediction"],
        "Bias Conf":       f"{bi['confidence']:.3f}",
        "Active Emotions": ", ".join(active) if active else "-",
    })

df_out = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 240)
print("\n" + df_out.to_string(index=False))

# ── 5. Emotion heatmap ─────────────────────────────────────────────────────────

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
ax.set_title("Emotion Probability Heatmap  (bold = predicted active ≥ 0.5)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = OUT_DIR / "emotion_heatmap.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

# ── 6. Bias confidence chart ───────────────────────────────────────────────────

bias_p_biased = [r["bias"]["probabilities"].get("biased", 0.0) for r in results]
bias_preds    = [r["bias"]["prediction"] for r in results]
bar_colors    = ["#d62728" if p == "biased" else "#2ca02c" for p in bias_preds]
row_labels_n  = [f"{i+1}. {CATEGORIES[i]}" for i in range(len(TEXTS))]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(row_labels_n, bias_p_biased, color=bar_colors, alpha=0.85)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5)
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
out = OUT_DIR / "bias_confidence.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# ── 7. Epistemic uncertainty chart ────────────────────────────────────────────

unc_scores  = [r["epistemic"]["uncertainty_score"] for r in results]
ep_labels   = [r["epistemic"]["label_name"] for r in results]
label_color = {"asserted": "#4C72B0", "hedged": "#DD8452", "speculative": "#d62728"}
bar_colors  = [label_color[l] for l in ep_labels]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(row_labels_n, unc_scores, color=bar_colors, alpha=0.85)
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
out = OUT_DIR / "epistemic_uncertainty.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

print("\n=== Done ===")
