"""
Multi-task NLP model performance figure generator.
To add new checkpoint data: append new rows to `results.csv` and re-run the script.
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

matplotlib.rcParams["font.family"] = "serif"

STYLE = {
    "figure_dpi": 300,
    "font_family": "serif",
    "font_size_title": 14,
    "font_size_label": 11,
    "font_size_tick": 9,
    "font_size_legend": 9,
    "model_colors": {
        "epistemic":         "#2E86AB",
        "political":         "#E84855",
        "emotion":           "#3BB273",
        "unified_warm":      "#F7B731",
        "unified_cold":      "#9B59B6",
        "unified_largebatch": "#E67E22",
        "unified_reg":       "#1ABC9C",
        "unified_difflr":    "#C0392B",
        "unified_weights":   "#7F8C8D",
        "tfidf_lr":          "#555555",
    },
    "model_linestyles": {
        "epistemic":         "-",
        "political":         "-",
        "emotion":           "-",
        "unified_warm":      "--",
        "unified_cold":      "-.",
        "unified_largebatch": "--",
        "unified_reg":       "-.",
        "unified_difflr":    "--",
        "unified_weights":   "-.",
        "tfidf_lr":          ":",
    },
    "in_progress_hatch": "//",
    "baseline_color": "#888888",
    "background": "white",
    "grid_alpha": 0.3,
}

MODEL_DISPLAY = {
    "epistemic":          "Epistemic specialist",
    "political":          "Political specialist",
    "emotion":            "Emotion specialist",
    "unified_warm":       "Unified (warm-start)",
    "unified_cold":       "Unified (cold-start)",
    "unified_largebatch": "Unified (large batch)",
    "unified_reg":        "Unified (regularized)",
    "unified_difflr":     "Unified (diff. LR)",
    "unified_weights":    "Unified (task weights)",
    "tfidf_lr":           "TF-IDF + LR baseline",
}

TASK_DISPLAY = {
    "epistemic": "Epistemic",
    "political":  "Political Bias",
    "emotion":    "Emotion",
}

SUBTASK_MODELS  = ["epistemic", "political", "emotion"]
UNIFIED_MODELS  = [
    "unified_warm", "unified_cold",
    "unified_largebatch", "unified_reg", "unified_difflr", "unified_weights",
]
BASELINE_MODELS = ["tfidf_lr"]
ALL_MODELS      = SUBTASK_MODELS + UNIFIED_MODELS + BASELINE_MODELS
ALL_TASKS       = ["epistemic", "political", "emotion"]

# ---------------------------------------------------------------------------
# Data loading & normalization
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in ["model_id", "task", "metric", "split"]:
        df[col] = df[col].str.strip()
    return df


def normalize_to_relative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Express every value as a ratio relative to the matching sub-task specialist.

    For 'accuracy' and 'f1' (higher-is-better), ratio = model / specialist.
    For 'calibration_error' and 'bias_score' (lower-is-better), ratio =
    specialist / model, so that values > 1.0 still mean "better than baseline".

    The specialist for task T is the model whose model_id == T.
    Only the test split is used for baselines to keep evaluation canonical.
    """
    lower_is_better = {"calibration_error", "bias_score"}

    # Build baseline table: specialist performance on its own task at its
    # latest checkpoint.  Using the mean across all checkpoints would give a
    # value lower than the final model (training improves over time), causing
    # the specialist to appear > 1.0 relative to itself.
    # Prefer test split; fall back to val when no test data exists.
    baselines = {}
    for split in df["split"].unique():
        candidate_rows = df[
            df["model_id"].isin(SUBTASK_MODELS) &
            (df["split"] == split)
        ]
        for (task, metric), grp in candidate_rows.groupby(["task", "metric"]):
            specialist_rows = grp[grp["model_id"] == task]
            if specialist_rows.empty:
                continue
            # Use the latest checkpoint only, not the mean across all checkpoints.
            latest_idx = specialist_rows["checkpoint_step"].idxmax()
            baselines[(task, metric, split)] = specialist_rows.loc[latest_idx, "value"]

    def relative(row):
        key = (row["task"], row["metric"], row["split"])
        base = baselines.get(key)
        if base is None:
            # tfidf_lr and others evaluated on test split when specialists
            # only have val data — fall back to the val-split baseline.
            base = baselines.get((row["task"], row["metric"], "val"))
        if base is None or base == 0:
            return np.nan
        if row["metric"] in lower_is_better:
            return base / row["value"]
        return row["value"] / base

    df = df.copy()
    df["relative"] = df.apply(relative, axis=1)
    return df


def get_latest_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows at the highest checkpoint_step seen per model × task × metric × split."""
    idx = df.groupby(["model_id", "task", "metric", "split"])["checkpoint_step"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def is_in_progress(df: pd.DataFrame, model_id: str) -> bool:
    """True when a unified model has fewer checkpoint steps than the sub-task specialists,
    or has no rows at all (training hasn't produced a validation checkpoint yet)."""
    if model_id not in UNIFIED_MODELS:
        return False
    model_rows = df[df["model_id"] == model_id]
    if model_rows.empty:
        return True
    specialist_max = df[df["model_id"].isin(SUBTASK_MODELS)]["checkpoint_step"].max()
    model_max      = model_rows["checkpoint_step"].max()
    return model_max < specialist_max


def _get_display_df(df: pd.DataFrame, models: list, tasks: list) -> pd.DataFrame:
    """
    Return latest-checkpoint rows for each model, using val split where available
    and falling back to test split for models (e.g. tfidf_lr) that were only
    evaluated on the test set.
    """
    parts = []
    for mid in models:
        mdf = df[(df["model_id"] == mid) & (df["task"].isin(tasks))]
        val_df = mdf[mdf["split"] == "val"]
        parts.append(val_df if not val_df.empty else mdf[mdf["split"] == "test"])
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return get_latest_checkpoint(pd.concat(parts, ignore_index=True))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_base_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(STYLE["background"])
    ax.grid(alpha=STYLE["grid_alpha"])
    if title:
        ax.set_title(title, fontsize=STYLE["font_size_title"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE["font_size_label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE["font_size_label"])
    ax.tick_params(labelsize=STYLE["font_size_tick"])


def save_figure(fig, name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=STYLE["figure_dpi"], bbox_inches="tight",
                    facecolor=STYLE["background"])
    print(f"  Saved: {name}.png / {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1 — Radar chart
# ---------------------------------------------------------------------------

def plot_radar(df: pd.DataFrame, output_dir: str, tasks=None, models=None):
    tasks  = tasks  or ALL_TASKS
    models = models or ALL_MODELS

    latest = _get_display_df(df, models, tasks)
    # Average across metrics for each model × task
    avg = (
        latest[latest["model_id"].isin(models) & latest["task"].isin(tasks)]
        .groupby(["model_id", "task"])["relative"]
        .mean()
        .reset_index()
    )

    n_axes = len(tasks)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_facecolor(STYLE["background"])

    # Shade the baseline ring at relative = 1.0
    ax.fill_between(
        np.linspace(0, 2 * np.pi, 200),
        0.95, 1.05,
        color=STYLE["baseline_color"], alpha=0.15, zorder=0,
    )
    ax.plot(
        np.linspace(0, 2 * np.pi, 200),
        [1.0] * 200,
        color=STYLE["baseline_color"], lw=1.2, ls=":", zorder=1,
        label="Sub-task baseline (1.0)",
    )

    for model_id in models:
        vals = []
        for task in tasks:
            row = avg[(avg["model_id"] == model_id) & (avg["task"] == task)]
            vals.append(float(row["relative"].values[0]) if not row.empty else np.nan)
        vals += vals[:1]

        color = STYLE["model_colors"][model_id]
        ls    = STYLE["model_linestyles"][model_id]
        lw    = 2.2 if model_id in UNIFIED_MODELS else 1.5
        label = MODEL_DISPLAY[model_id]
        if is_in_progress(df, model_id):
            label += " *"

        ax.plot(angles, vals, color=color, ls=ls, lw=lw, label=label, marker="o", ms=5)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks],
                       fontsize=STYLE["font_size_label"])
    ax.tick_params(labelsize=STYLE["font_size_tick"])
    ax.set_title("Relative performance — all tasks (latest checkpoint)",
                 fontsize=STYLE["font_size_title"], pad=18)

    # Note about in-progress models
    in_progress = [m for m in UNIFIED_MODELS if m in models and is_in_progress(df, m)]
    if in_progress:
        note = "* Still training: " + ", ".join(MODEL_DISPLAY[m] for m in in_progress)
        fig.text(0.5, 0.01, note, ha="center", fontsize=STYLE["font_size_legend"],
                 color=STYLE["baseline_color"])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=STYLE["font_size_legend"])
    fig.tight_layout()
    save_figure(fig, "radar_per_task", output_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Grouped bar chart (latest checkpoint)
# ---------------------------------------------------------------------------

def plot_bar_latest(df: pd.DataFrame, output_dir: str, tasks=None, models=None):
    tasks  = tasks  or ALL_TASKS
    models = models or ALL_MODELS

    # Specialists are always 1.0 by definition — omit them to reduce clutter.
    bar_models = [m for m in models if m not in SUBTASK_MODELS]

    latest = _get_display_df(df, bar_models, tasks)
    avg = (
        latest[latest["model_id"].isin(bar_models) & latest["task"].isin(tasks)]
        .groupby(["model_id", "task"])["relative"]
        .mean()
        .reset_index()
    )

    # Only count models that have data — in-progress models with no rows yet
    # would otherwise leave phantom gaps inside each task group.
    bar_models = [m for m in bar_models
                  if not avg[avg["model_id"] == m].empty]

    n_tasks  = len(tasks)
    n_models = len(bar_models)
    # 0.8 total group width keeps inter-group separation; drawn width = slot
    # width (no * 0.9 multiplier) so bars within a group are fully adjacent.
    width = 0.8 / n_models
    x     = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_base_style(ax,
                      title="Relative performance at latest checkpoint",
                      xlabel="Task",
                      ylabel="Relative performance (1.0 = specialist baseline)")

    for i, model_id in enumerate(bar_models):
        offset = (i - n_models / 2 + 0.5) * width
        sub    = avg[avg["model_id"] == model_id]
        vals   = [sub[sub["task"] == t]["relative"].values[0]
                  if not sub[sub["task"] == t].empty else np.nan
                  for t in tasks]

        color = STYLE["model_colors"][model_id]
        label = MODEL_DISPLAY[model_id]
        if model_id in BASELINE_MODELS:
            hatch = "xx"
            edgecolor = "gray"
        elif is_in_progress(df, model_id):
            hatch = STYLE["in_progress_hatch"]
            edgecolor = "gray"
            label += " (partial)"
        else:
            hatch = ""
            edgecolor = "white"

        ax.bar(x + offset, vals, width=width,  # no *0.9 — bars touch within group
               color=color, hatch=hatch, edgecolor=edgecolor,
               alpha=0.85, label=label)

    ax.axhline(1.0, color=STYLE["baseline_color"], ls="--", lw=1.5,
               label="Specialist baseline (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks], fontsize=STYLE["font_size_label"])

    # Legend outside the axes to the right so it never overlaps bars
    ax.legend(fontsize=STYLE["font_size_legend"],
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    save_figure(fig, "bar_latest_checkpoint", output_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Training curves (one subplot per task)
# ---------------------------------------------------------------------------

def plot_training_curves(df: pd.DataFrame, output_dir: str, tasks=None, models=None):
    tasks  = tasks  or ALL_TASKS
    models = models or ALL_MODELS

    # Training curves use val split for neural models.
    # Baseline models (tfidf_lr) have no val data, so fetch their test metrics
    # separately for use as horizontal reference lines.
    neural_models   = [m for m in models if m not in BASELINE_MODELS]
    baseline_models = [m for m in models if m in BASELINE_MODELS]

    val_df = df[df["split"] == "val"]
    curves = (
        val_df[val_df["model_id"].isin(neural_models) & val_df["task"].isin(tasks)]
        .groupby(["model_id", "task", "checkpoint_step"])["relative"]
        .mean()
        .reset_index()
    )

    # One averaged relative value per baseline model × task (across metrics)
    baseline_display = _get_display_df(df, baseline_models, tasks)
    baseline_avg = (
        baseline_display[baseline_display["model_id"].isin(baseline_models)]
        .groupby(["model_id", "task"])["relative"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        _apply_base_style(ax, title=TASK_DISPLAY[task],
                          xlabel="Checkpoint step",
                          ylabel="Relative performance" if ax == axes[0] else "")

        ax.axhline(1.0, color=STYLE["baseline_color"], ls=":", lw=1.2, alpha=0.8)

        # Baseline models: draw as horizontal reference lines spanning the full x range
        for model_id in baseline_models:
            row = baseline_avg[
                (baseline_avg["model_id"] == model_id) & (baseline_avg["task"] == task)
            ]
            if row.empty:
                continue
            y = float(row["relative"].values[0])
            color = STYLE["model_colors"][model_id]
            ax.axhline(y, color=color, ls=STYLE["model_linestyles"][model_id],
                       lw=1.5, alpha=0.85, label=MODEL_DISPLAY[model_id])

        for model_id in neural_models:
            sub = curves[(curves["model_id"] == model_id) & (curves["task"] == task)]
            if sub.empty:
                continue

            color = STYLE["model_colors"][model_id]
            ls    = STYLE["model_linestyles"][model_id]
            lw    = 2.0 if model_id in UNIFIED_MODELS else 1.5

            sub = sub.sort_values("checkpoint_step")
            ax.plot(sub["checkpoint_step"], sub["relative"],
                    color=color, ls=ls, lw=lw, marker="o", ms=4,
                    label=MODEL_DISPLAY[model_id])

            # Shade "training in progress" region past the last checkpoint
            if is_in_progress(df, model_id):
                x_end    = sub["checkpoint_step"].max()
                x_target = df[df["model_id"].isin(SUBTASK_MODELS)]["checkpoint_step"].max()
                ax.axvspan(x_end, x_target * 1.05,
                           color=color, alpha=0.06,
                           label=f"{MODEL_DISPLAY[model_id]} (training)")
                ax.annotate("ongoing", xy=(x_end, sub["relative"].iloc[-1]),
                            xytext=(8, 0), textcoords="offset points",
                            fontsize=7, color=color, va="center")

    # Shared legend outside grid
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen   = {}
    h_out  = []
    l_out  = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            h_out.append(h)
            l_out.append(l)
    fig.legend(h_out, l_out, loc="lower center",
               ncol=min(len(models), 5),
               fontsize=STYLE["font_size_legend"],
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Training curves by task", fontsize=STYLE["font_size_title"], y=1.02)
    fig.tight_layout()
    save_figure(fig, "training_curves_by_task", output_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Heatmap (models × tasks)
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, output_dir: str, tasks=None, models=None):
    tasks  = tasks  or ALL_TASKS
    models = models or ALL_MODELS

    latest = _get_display_df(df, models, tasks)
    avg = (
        latest[latest["model_id"].isin(models) & latest["task"].isin(tasks)]
        .groupby(["model_id", "task"])["relative"]
        .mean()
        .reset_index()
    )

    matrix = pd.DataFrame(index=models, columns=tasks, dtype=float)
    for _, row in avg.iterrows():
        matrix.loc[row["model_id"], row["task"]] = row["relative"]

    # Diverging colormap centered at 1.0
    vmin = min(0.7, float(matrix.min().min()) - 0.05)
    vmax = max(1.3, float(matrix.max().max()) + 0.05)
    center = 1.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        matrix.astype(float),
        ax=ax,
        cmap="RdYlGn",
        center=center, vmin=vmin, vmax=vmax,
        annot=True, fmt=".3f",
        linewidths=0.5, linecolor="#dddddd",
        cbar_kws={"label": "Relative performance vs. sub-task specialist"},
    )

    # Mark in-progress cells with an asterisk overlay
    for i, model_id in enumerate(models):
        if is_in_progress(df, model_id):
            for j in range(len(tasks)):
                ax.text(j + 0.85, i + 0.18, "*",
                        fontsize=10, color="#333333", fontweight="bold")

    ax.set_yticklabels([MODEL_DISPLAY[m] for m in models],
                       fontsize=STYLE["font_size_tick"], rotation=0)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in tasks],
                       fontsize=STYLE["font_size_tick"])
    ax.set_title("Performance heatmap (models × tasks, latest checkpoint)",
                 fontsize=STYLE["font_size_title"])

    in_progress = [m for m in UNIFIED_MODELS if m in models and is_in_progress(df, m)]
    if in_progress:
        fig.text(0.5, -0.04,
                 "* Incomplete training data for: " +
                 ", ".join(MODEL_DISPLAY[m] for m in in_progress),
                 ha="center", fontsize=STYLE["font_size_legend"],
                 color=STYLE["baseline_color"])

    fig.tight_layout()
    save_figure(fig, "heatmap_model_x_task", output_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Unified warm vs. cold scatter
# ---------------------------------------------------------------------------

def plot_unified_scatter(df: pd.DataFrame, output_dir: str, tasks=None, models=None):
    tasks = tasks or ALL_TASKS

    latest = get_latest_checkpoint(df[df["split"] == "val"])

    warm = latest[latest["model_id"] == "unified_warm"][["task", "metric", "relative"]]
    cold = latest[latest["model_id"] == "unified_cold"][["task", "metric", "relative"]]
    merged = warm.merge(cold, on=["task", "metric"], suffixes=("_warm", "_cold"))
    merged = merged[merged["task"].isin(tasks)]

    if merged.empty:
        warnings.warn("No overlapping (task, metric) data for both unified models — "
                      "skipping scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    _apply_base_style(ax,
                      title="Unified warm-start vs. cold-start",
                      xlabel="Unified cold-start (relative performance)",
                      ylabel="Unified warm-start (relative performance)")

    task_colors = {t: STYLE["model_colors"][t] for t in ALL_TASKS}

    for task, grp in merged.groupby("task"):
        if task not in tasks:
            continue
        ax.scatter(grp["relative_cold"], grp["relative_warm"],
                   color=task_colors.get(task, "#555555"),
                   label=TASK_DISPLAY.get(task, task),
                   s=60, zorder=3, alpha=0.85)

    # y = x diagonal
    lims = [min(merged["relative_cold"].min(), merged["relative_warm"].min()) - 0.05,
            max(merged["relative_cold"].max(), merged["relative_warm"].max()) + 0.05]
    ax.plot(lims, lims, ls="--", color=STYLE["baseline_color"], lw=1.2,
            label="Equal performance (y = x)")

    # Sub-task specialist reference point
    ax.scatter([1.0], [1.0], marker="*", s=220, color="#FFD700",
               edgecolors="#444444", zorder=5, label="Sub-task specialist (1.0, 1.0)")

    # Marginal rug plots for density context when points overlap
    ax.plot(merged["relative_cold"],
            [ax.get_ylim()[0]] * len(merged) if ax.get_ylim()[0] != 0 else [lims[0]] * len(merged),
            "|", color="#888888", alpha=0.4, ms=10)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=STYLE["font_size_legend"])

    if is_in_progress(df, "unified_warm") or is_in_progress(df, "unified_cold"):
        fig.text(0.5, -0.03,
                 "Note: one or both unified models are still training; "
                 "values reflect latest available checkpoint.",
                 ha="center", fontsize=STYLE["font_size_legend"],
                 color=STYLE["baseline_color"])

    fig.tight_layout()
    save_figure(fig, "unified_vs_specialist_scatter", output_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate performance comparison figures for multi-task NLP models."
    )
    parser.add_argument("--data",   default="./results.csv",
                        help="Path to results CSV (default: ./results.csv)")
    parser.add_argument("--output", default="./figures/",
                        help="Output directory for figures (default: ./figures/)")
    parser.add_argument("--tasks",  nargs="*", choices=ALL_TASKS,
                        help="Subset of tasks to include (default: all)")
    parser.add_argument("--models", nargs="*", choices=ALL_MODELS,
                        help="Subset of models to include (default: all)")
    args = parser.parse_args()

    tasks  = args.tasks  or ALL_TASKS
    models = args.models or ALL_MODELS

    print(f"Loading data from: {args.data}")
    df = load_data(args.data)
    df = normalize_to_relative(df)

    # Warn about in-progress unified models
    specialist_max = df[df["model_id"].isin(SUBTASK_MODELS)]["checkpoint_step"].max()
    for m in UNIFIED_MODELS:
        if m in models and is_in_progress(df, m):
            model_rows = df[df["model_id"] == m]
            if model_rows.empty:
                warnings.warn(
                    f"[in-progress] {MODEL_DISPLAY[m]} has no validation checkpoints yet "
                    f"(specialists are at step {specialist_max}). "
                    "Model will appear as in-progress but has no data to plot."
                )
            else:
                model_max = model_rows["checkpoint_step"].max()
                warnings.warn(
                    f"[in-progress] {MODEL_DISPLAY[m]} has only reached checkpoint "
                    f"{model_max} (specialists are at {specialist_max}). "
                    "Figures will use available data."
                )

    print(f"\nGenerating figures → {args.output}")
    plot_radar(df, args.output, tasks=tasks, models=models)
    plot_bar_latest(df, args.output, tasks=tasks, models=models)
    plot_training_curves(df, args.output, tasks=tasks, models=models)
    plot_heatmap(df, args.output, tasks=tasks, models=models)
    plot_unified_scatter(df, args.output, tasks=tasks, models=models)

    print("\nDone. 5 figures saved (10 files: .png + .pdf each).")


if __name__ == "__main__":
    main()
