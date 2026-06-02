"""
Extract model evaluation metrics from run directories and write results.csv.

Usage:
    python extract_results.py [--runs /mldata/ece283-sentiment-analyzer/runs]
                              [--output results.csv]

Reads from these run directories (relative to --runs):
  bias/            → model_id "political"  (political-bias specialist)
  emotion/         → model_id "emotion"
  epistemic/       → model_id "epistemic"
  unified/coldStart → model_id "unified_cold"

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
    "bias":              "political",
    "emotion":           "emotion",
    "epistemic":         "epistemic",
    "unified/coldStart": "unified_cold",
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


def extract_unified_cold(run_dir: str) -> list[dict]:
    """
    Unified cold-start model.
    Source: history.json — per-epoch per-task val metrics ({task}_macro_f1).
    Checkpoint step is derived from step_losses.jsonl (max step per epoch).
    Falls back to epoch * 38351 (observed steps/epoch) if step_losses is absent.
    """
    rows = []
    model_id = "unified_cold"

    history_path = os.path.join(run_dir, "history.json")
    if not os.path.exists(history_path):
        warnings.warn(f"unified/coldStart: history.json not found at {history_path}")
        return rows

    with open(history_path) as f:
        history = json.load(f)

    epoch_step_map = _epoch_to_step_map(run_dir)
    if not epoch_step_map:
        warnings.warn(
            "unified/coldStart: step_losses.jsonl not found; "
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


# ---------------------------------------------------------------------------
# Dispatch table — add new run types here
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "bias":              extract_bias,
    "emotion":           extract_emotion,
    "epistemic":         extract_epistemic,
    "unified/coldStart": extract_unified_cold,
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
        "--output",
        default="./results.csv",
        help="Path to write results CSV (default: ./results.csv).",
    )
    args = parser.parse_args()

    all_rows = []
    for rel_path, extractor_fn in EXTRACTORS.items():
        run_dir = os.path.join(args.runs, rel_path)
        if not os.path.isdir(run_dir):
            warnings.warn(f"Run directory not found, skipping: {run_dir}")
            continue
        print(f"Extracting: {rel_path}  →  model_id={MODEL_ID_MAP[rel_path]}")
        rows = extractor_fn(run_dir)
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
