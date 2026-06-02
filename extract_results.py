"""
Extract model evaluation metrics from run directories and write results.csv.

Usage:
    python extract_results.py [--runs /mldata/ece283-sentiment-analyzer/runs]
                              [--baselines ./baselines]
                              [--output results.csv]

Reads from these run directories (relative to --runs):
  bias/            → model_id "political"  (political-bias specialist)
  emotion/         → model_id "emotion"
  epistemic/       → model_id "epistemic"
  unified/coldStart → model_id "unified_cold"

Also reads from:
  <baselines>/tfidf_lr/metrics_baseline.json → model_id "tfidf_lr"

To add a new run: configure it in RUN_CONFIG below, then re-run this script.
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map run path (relative to --runs) → CSV model_id
MODEL_ID_MAP = {
    "bias":                    "political",
    "emotion":                 "emotion",
    "epistemic":               "epistemic",
    "unified/coldStart":       "unified_cold",
    "unified/20260602_010419": "unified_warm",
    "unified_exp1_large":      "unified_largebatch",
    "unified_exp2_reg":        "unified_reg",
    "unified_exp3_difflr":     "unified_difflr",
    "unified_exp4_weights":    "unified_weights",
    "unified_exp6_avgencoder": "unified_avgencoder",
    "unified_exp7_staged":     "unified_staged",
}

# Internal task names in source files → canonical CSV task name
TASK_NAME_MAP = {
    "bias":      "political",
    "epistemic": "epistemic",
    "emotion":   "emotion",
}

# ---------------------------------------------------------------------------
# Per-run extractors
# ---------------------------------------------------------------------------

def _epoch_to_step_map(run_dir: str) -> dict[int, int]:
    """
    Read step_losses.jsonl and return {epoch: max_step_in_that_epoch}.
    Used to convert epoch numbers into actual training step counts.
    """
    path = os.path.join(run_dir, "step_losses.jsonl")
    if not os.path.exists(path):
        return {}
    epoch_max: dict[int, int] = defaultdict(int)
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            e = int(obj["epoch"])
            s = int(obj["step"])
            if s > epoch_max[e]:
                epoch_max[e] = s
    return dict(epoch_max)


def extract_bias(run_dir: str) -> list[dict]:
    """
    Bias specialist.
    Sources:
      history.json  — per-epoch val metrics (accuracy, f1_macro, loss)
      test_metrics.json — single test-split evaluation point

    Checkpoint step: bias has no step_losses.jsonl, so epoch number is used
    directly as checkpoint_step.  This is the only model where step ≠ gradient
    step; the figures treat bias as a reference line so axis scale doesn't matter.
    """
    rows = []
    model_id = "political"
    task     = "political"

    history_path = os.path.join(run_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        metric_map = {
            "accuracy": "accuracy",
            "f1_macro": "f1",
            "loss":     "calibration_error",
        }
        for entry in history:
            epoch = int(entry["epoch"])
            for src_key, metric_name in metric_map.items():
                if src_key in entry:
                    rows.append({
                        "model_id":        model_id,
                        "task":            task,
                        "metric":          metric_name,
                        "value":           entry[src_key],
                        "checkpoint_step": epoch,   # epoch used as ordinal step
                        "split":           "val",
                    })
    else:
        warnings.warn(f"bias: history.json not found at {history_path}")

    test_path = os.path.join(run_dir, "test_metrics.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            tm = json.load(f)
        # Use last epoch as checkpoint_step for the test evaluation
        last_step = max(int(e["epoch"]) for e in history) if rows else 0
        test_metric_map = {
            "accuracy": "accuracy",
            "f1_macro": "f1",
            "loss":     "calibration_error",
        }
        for src_key, metric_name in test_metric_map.items():
            if src_key in tm:
                rows.append({
                    "model_id":        model_id,
                    "task":            task,
                    "metric":          metric_name,
                    "value":           tm[src_key],
                    "checkpoint_step": last_step,
                    "split":           "test",
                })
    else:
        warnings.warn(f"bias: test_metrics.json not found at {test_path}")

    return rows


def extract_emotion(run_dir: str) -> list[dict]:
    """
    Emotion specialist.
    Source: trainer_state.json log_history — eval entries (those with eval_macro_f1).
    One entry per epoch; step numbers are exact gradient steps.
    """
    rows = []
    model_id = "emotion"
    task     = "emotion"

    state_path = os.path.join(run_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        warnings.warn(f"emotion: trainer_state.json not found at {state_path}")
        return rows

    with open(state_path) as f:
        state = json.load(f)

    metric_map = {
        "eval_macro_f1":       "f1",
        "eval_subset_accuracy": "accuracy",
        "eval_hamming_loss":   "calibration_error",
    }

    eval_entries = [e for e in state["log_history"] if "eval_macro_f1" in e]
    if not eval_entries:
        warnings.warn("emotion: no eval entries found in trainer_state.json")
        return rows

    for entry in eval_entries:
        step = int(entry["step"])
        for src_key, metric_name in metric_map.items():
            if src_key in entry:
                rows.append({
                    "model_id":        model_id,
                    "task":            task,
                    "metric":          metric_name,
                    "value":           entry[src_key],
                    "checkpoint_step": step,
                    "split":           "val",
                })

    return rows


def extract_epistemic(run_dir: str) -> list[dict]:
    """
    Epistemic specialist.
    Source: metrics.json — per-epoch val metrics; each entry carries its step.
    """
    rows = []
    model_id = "epistemic"
    task     = "epistemic"

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        warnings.warn(f"epistemic: metrics.json not found at {metrics_path}")
        return rows

    with open(metrics_path) as f:
        history = json.load(f)

    metric_map = {
        "val_macro_f1": "f1",
        "val_loss":     "calibration_error",
    }

    for entry in history:
        step = int(entry["step"])
        for src_key, metric_name in metric_map.items():
            if src_key in entry:
                rows.append({
                    "model_id":        model_id,
                    "task":            task,
                    "metric":          metric_name,
                    "value":           entry[src_key],
                    "checkpoint_step": step,
                    "split":           "val",
                })

    return rows


def _find_run_dir(base_path: str) -> str:
    """
    Resolve the actual directory containing run artifacts.
    If history.json or step_losses.jsonl exist directly under base_path, return it.
    Otherwise, return the lexicographically latest subdirectory (date-stamped runs
    like unified_exp*/YYYYMMDD_HHMMSS sort correctly by name).
    """
    sentinel_files = {"history.json", "step_losses.jsonl", "best.pt"}
    if any(os.path.exists(os.path.join(base_path, f)) for f in sentinel_files):
        return base_path
    subdirs = sorted(
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    )
    if subdirs:
        return os.path.join(base_path, subdirs[-1])
    return base_path


def _extract_unified_history(run_dir: str, model_id: str) -> list[dict]:
    """
    Shared extractor for any unified model that saves history.json.
    Source: history.json — per-epoch per-task val metrics ({task}_macro_f1).
    Checkpoint step is derived from step_losses.jsonl (max step per epoch).
    Falls back to epoch * 38351 (observed steps/epoch for the cold-start run)
    when step_losses is absent.
    """
    rows = []

    history_path = os.path.join(run_dir, "history.json")
    if not os.path.exists(history_path):
        warnings.warn(
            f"{model_id}: no history.json at {history_path} — "
            "run may still be in the first epoch (no validation checkpoint yet). "
            "Skipping; model will appear as in-progress in figures."
        )
        return rows

    with open(history_path) as f:
        history = json.load(f)

    epoch_step_map = _epoch_to_step_map(run_dir)
    if not epoch_step_map:
        warnings.warn(
            f"{model_id}: step_losses.jsonl not found; "
            "estimating checkpoint_step as epoch × 38351."
        )

    for entry in history:
        epoch = int(entry["epoch"])
        step = epoch_step_map.get(epoch, epoch * 38351)

        # history.json keys: {task}_macro_f1 for each task
        for internal_task, canonical_task in TASK_NAME_MAP.items():
            f1_key = f"{internal_task}_macro_f1"
            if f1_key in entry:
                rows.append({
                    "model_id":        model_id,
                    "task":            canonical_task,
                    "metric":          "f1",
                    "value":           entry[f1_key],
                    "checkpoint_step": step,
                    "split":           "val",
                })

    return rows


def extract_tfidf_lr(baselines_dir: str) -> list[dict]:
    """
    TF-IDF + Logistic Regression baseline.
    Source: baselines/tfidf_lr/metrics_baseline.json — single test-split evaluation.
    checkpoint_step = 0 (no training curve; drawn as a horizontal reference line
    in the training-curves figure).

    Task mapping:
      task1_epistemic sentence → epistemic  (macro_f1)
      task2_bias_binary        → political  (macro_f1, accuracy from confusion matrix)
      task3_emotion            → emotion    (macro_f1, hamming_loss → calibration_error)
    """
    rows = []
    model_id = "tfidf_lr"

    metrics_path = os.path.join(baselines_dir, "tfidf_lr", "metrics_baseline.json")
    if not os.path.exists(metrics_path):
        warnings.warn(f"tfidf_lr: metrics_baseline.json not found at {metrics_path}")
        return rows

    with open(metrics_path) as f:
        m = json.load(f)

    def row(task, metric, value):
        return {
            "model_id":        model_id,
            "task":            task,
            "metric":          metric,
            "value":           value,
            "checkpoint_step": 0,
            "split":           "test",
        }

    # Epistemic (sentence-level macro F1)
    ep = m.get("task1_epistemic", {}).get("sentence", {})
    if "macro_f1" in ep:
        rows.append(row("epistemic", "f1", ep["macro_f1"]))

    # Political bias (binary classification)
    bias = m.get("task2_bias_binary", {})
    if "macro_f1" in bias:
        rows.append(row("political", "f1", bias["macro_f1"]))
    cm = bias.get("confusion_matrix")
    if cm:
        correct = sum(cm[i][i] for i in range(len(cm)))
        total   = sum(v for r in cm for v in r)
        rows.append(row("political", "accuracy", correct / total))

    # Emotion (multi-label macro F1 + hamming loss)
    emo = m.get("task3_emotion", {})
    if "macro_f1" in emo:
        rows.append(row("emotion", "f1", emo["macro_f1"]))
    if "hamming_loss" in emo:
        rows.append(row("emotion", "calibration_error", emo["hamming_loss"]))

    return rows


# ---------------------------------------------------------------------------
# Dispatch table — add new run types here
# ---------------------------------------------------------------------------

def _unified(model_id):
    """Return an extractor closure for a unified model with the given model_id."""
    def _extract(run_dir):
        return _extract_unified_history(run_dir, model_id)
    return _extract


EXTRACTORS = {
    "bias":                    extract_bias,
    "emotion":                 extract_emotion,
    "epistemic":               extract_epistemic,
    "unified/coldStart":       _unified("unified_cold"),
    "unified/20260602_010419": _unified("unified_warm"),
    "unified_exp1_large":      _unified("unified_largebatch"),
    "unified_exp2_reg":        _unified("unified_reg"),
    "unified_exp3_difflr":     _unified("unified_difflr"),
    "unified_exp4_weights":    _unified("unified_weights"),
    "unified_exp6_avgencoder": _unified("unified_avgencoder"),
    "unified_exp7_staged":     _unified("unified_staged"),
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract evaluation metrics from run directories into results.csv."
    )
    parser.add_argument(
        "--runs",
        default="/mldata/ece283-sentiment-analyzer/runs",
        help="Root directory containing model run folders.",
    )
    parser.add_argument(
        "--baselines",
        default="./baselines",
        help="Root directory containing baseline results (default: ./baselines).",
    )
    parser.add_argument(
        "--output",
        default="./results.csv",
        help="Path to write results CSV (default: ./results.csv).",
    )
    args = parser.parse_args()

    all_rows = []
    for rel_path, extractor_fn in EXTRACTORS.items():
        base_dir = os.path.join(args.runs, rel_path)
        if not os.path.isdir(base_dir):
            warnings.warn(f"Run directory not found, skipping: {base_dir}")
            continue
        run_dir = _find_run_dir(base_dir)
        model_id = MODEL_ID_MAP[rel_path]
        print(f"Extracting: {rel_path}  →  model_id={model_id}  (dir: {os.path.relpath(run_dir, args.runs)})")
        rows = extractor_fn(run_dir)
        print(f"  {len(rows)} rows extracted.")
        all_rows.extend(rows)

    print(f"Extracting: baselines/tfidf_lr  →  model_id=tfidf_lr")
    rows = extract_tfidf_lr(args.baselines)
    print(f"  {len(rows)} rows extracted.")
    all_rows.extend(rows)

    if not all_rows:
        print("No data extracted. Check that --runs points to the correct directory.")
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=[
        "model_id", "task", "metric", "value", "checkpoint_step", "split"
    ])
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print(df.groupby(["model_id", "task", "metric", "split"]).size().to_string())


if __name__ == "__main__":
    main()
