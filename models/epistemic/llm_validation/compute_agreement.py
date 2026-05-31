#!/usr/bin/env python3
"""
Compute agreement between rule-based labels and LLM labels.

Reads:
  sentences.jsonl          -- gold rule labels (written by build_prompts.py)
  outputs/<model>.jsonl    -- one file per LLM, each line {"id": N, "label": "..."}

Outputs (to stdout + agreement_report.txt):
  - Cohen's κ: each LLM vs rule labels
  - Fleiss' κ: across all LLMs (inter-LLM ceiling)
  - Majority-vote LLM label vs rule label confusion matrix
  - Per-cue-type agreement breakdown
  - Decision recommendation per the project spec:
      ship if:  κ(LLM-vs-rule) > 0.7 overall
                AND conditionals don't show systematic LLM disagreement

Usage:
    python compute_agreement.py [--outputs-dir outputs/] [--report agreement_report.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

LABELS = ["asserted", "hedged", "speculative"]
LABEL2INT = {l: i for i, l in enumerate(LABELS)}

CUE_DISPLAY = {
    "none": "no_ccue (→ asserted)",
    "speculation_modal_probable_": "modal_probable (→ hedged)",
    "speculation_hypo_doxastic _": "hypo_doxastic  (→ hedged)",
    "speculation_hypo_condition _": "hypo_condition (→ speculative??)",
    "speculation_hypo_investigation _": "hypo_investigation (→ speculative)",
}


# ---------------------------------------------------------------------------
# Cohen's κ (pairwise)
# ---------------------------------------------------------------------------

def cohen_kappa(y1: list[int], y2: list[int], n_classes: int = 3) -> float:
    n = len(y1)
    if n == 0:
        return float("nan")
    # Observed agreement
    p_o = sum(a == b for a, b in zip(y1, y2)) / n
    # Expected agreement
    count1 = Counter(y1)
    count2 = Counter(y2)
    p_e = sum(
        (count1.get(c, 0) / n) * (count2.get(c, 0) / n)
        for c in range(n_classes)
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def percent_agreement(y1: list[int], y2: list[int]) -> float:
    if not y1:
        return float("nan")
    return sum(a == b for a, b in zip(y1, y2)) / len(y1)


# ---------------------------------------------------------------------------
# Fleiss' κ (multi-rater)
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings: list[list[int]], n_classes: int = 3) -> float:
    """
    ratings: list of rater label lists, all same length N.
    Each inner list contains one label (as int) per subject.
    """
    n_raters = len(ratings)
    n = len(ratings[0])
    if n_raters < 2:
        return float("nan")

    # Build n×k matrix: ratings_matrix[i][j] = number of raters who assigned
    # category j to subject i
    ratings_matrix = [[0] * n_classes for _ in range(n)]
    for rater_labels in ratings:
        for i, label in enumerate(rater_labels):
            if 0 <= label < n_classes:
                ratings_matrix[i][label] += 1

    # P_i = (1 / (n_r*(n_r-1))) * sum_j(n_ij * (n_ij - 1))
    P_i = []
    for row in ratings_matrix:
        total_agree = sum(c * (c - 1) for c in row)
        if n_raters <= 1:
            P_i.append(0.0)
        else:
            P_i.append(total_agree / (n_raters * (n_raters - 1)))

    P_bar = sum(P_i) / n

    # p_j = (1 / (n * n_r)) * sum_i(n_ij)
    p_j = []
    for j in range(n_classes):
        p_j.append(sum(ratings_matrix[i][j] for i in range(n)) / (n * n_raters))

    P_e = sum(p ** 2 for p in p_j)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def majority_vote(label_lists: list[list[int]]) -> list[int]:
    """Element-wise majority vote across rater lists."""
    n = len(label_lists[0])
    result = []
    for i in range(n):
        votes = Counter(labels[i] for labels in label_lists)
        result.append(votes.most_common(1)[0][0])
    return result


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[str]) -> str:
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            matrix[t][p] += 1

    col_w = max(len(l) for l in labels) + 2
    lines = []
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels)
    lines.append("Predicted →")
    lines.append(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_w}}  " + "  ".join(f"{v:{col_w}d}" for v in matrix[i])
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def _kappa_interp(k: float) -> str:
    if k >= 0.80:
        return "almost perfect"
    if k >= 0.60:
        return "substantial"
    if k >= 0.40:
        return "moderate"
    if k >= 0.20:
        return "fair"
    return "slight/poor"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_gold(sentences_path: Path) -> dict[int, dict]:
    gold = {}
    with sentences_path.open() as f:
        for line in f:
            rec = json.loads(line)
            gold[rec["global_id"]] = rec
    return gold


def load_llm_output(path: Path, gold: dict[int, dict]) -> tuple[str, list[int]]:
    """Return (model_name, label_list_aligned_to_gold_order)."""
    model_name = path.stem
    id_to_label: dict[int, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj and "label" in obj:
                id_to_label[int(obj["id"])] = obj["label"].lower().strip()

    gold_ids = sorted(gold.keys())
    labels = []
    missing = 0
    for gid in gold_ids:
        raw = id_to_label.get(gid)
        if raw is None:
            labels.append(-1)
            missing += 1
        elif raw in LABEL2INT:
            labels.append(LABEL2INT[raw])
        else:
            # Fuzzy match
            matched = next((l for l in LABELS if l.startswith(raw[:3])), None)
            labels.append(LABEL2INT[matched] if matched else -1)
            if not matched:
                missing += 1

    if missing:
        print(f"  WARNING [{model_name}]: {missing}/{len(gold_ids)} sentences missing or unparseable")

    return model_name, labels


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    gold: dict[int, dict],
    llm_outputs: dict[str, list[int]],
) -> str:
    gold_ids = sorted(gold.keys())
    rule_labels = [LABEL2INT.get(gold[gid]["rule_label"], -1) for gid in gold_ids]
    cue_types = [gold[gid]["primary_cue_type"] for gid in gold_ids]

    lines: list[str] = []

    def h(text: str) -> None:
        lines.append("")
        lines.append("=" * 60)
        lines.append(text)
        lines.append("=" * 60)

    # ------------------------------------------------------------------
    # 1. Per-LLM κ vs rule labels
    # ------------------------------------------------------------------
    h("1. Cohen's κ: each LLM vs rule labels")
    model_names = sorted(llm_outputs.keys())
    kappas_vs_rule: dict[str, float] = {}
    for model in model_names:
        llm = llm_outputs[model]
        # Filter to rows where both are valid
        pairs = [(r, l) for r, l in zip(rule_labels, llm) if r >= 0 and l >= 0]
        if not pairs:
            lines.append(f"  {model}: no valid pairs")
            continue
        r_vals, l_vals = zip(*pairs)
        k = cohen_kappa(list(r_vals), list(l_vals))
        pa = percent_agreement(list(r_vals), list(l_vals))
        kappas_vs_rule[model] = k
        lines.append(f"  {model:<25}  κ = {k:+.3f}  ({_kappa_interp(k)})  "
                     f"raw agreement = {pa:.1%}  n = {len(pairs)}")

    # ------------------------------------------------------------------
    # 2. Fleiss' κ across LLMs (inter-LLM ceiling)
    # ------------------------------------------------------------------
    h("2. Fleiss' κ across all LLMs (inter-LLM agreement ceiling)")
    valid_models = [m for m in model_names if llm_outputs[m]]
    if len(valid_models) >= 2:
        all_llm_labels = [llm_outputs[m] for m in valid_models]
        # Use only rows where all raters gave valid labels
        valid_mask = [all(labels[i] >= 0 for labels in all_llm_labels)
                      for i in range(len(gold_ids))]
        filtered = [[labels[i] for i in range(len(gold_ids)) if valid_mask[i]]
                    for labels in all_llm_labels]
        fk = fleiss_kappa(filtered)
        lines.append(f"  Fleiss' κ (n_raters={len(valid_models)}) = {fk:+.3f}  "
                     f"({_kappa_interp(fk)})  n = {sum(valid_mask)}")
        lines.append("")
        lines.append("  Interpretation: this is the task difficulty ceiling.")
        lines.append("  Rule labels cannot be expected to exceed this agreement.")
    else:
        lines.append("  Need ≥2 LLM output files for Fleiss' κ.")
        fk = float("nan")

    # ------------------------------------------------------------------
    # 3. Majority-vote confusion matrix
    # ------------------------------------------------------------------
    h("3. Confusion matrix: rule labels (rows) vs LLM majority vote (cols)")
    if len(valid_models) >= 1:
        all_llm_labels = [llm_outputs[m] for m in valid_models]
        # Fill -1 gaps with most frequent LLM label for that position (fallback)
        mvote = majority_vote(all_llm_labels)
        valid_pairs = [
            (r, m) for r, m in zip(rule_labels, mvote) if r >= 0 and m >= 0
        ]
        if valid_pairs:
            r_vals, m_vals = zip(*valid_pairs)
            lines.append(confusion_matrix(list(r_vals), list(m_vals), LABELS))
            lines.append("")
            k = cohen_kappa(list(r_vals), list(m_vals))
            lines.append(f"  κ(rule vs majority) = {k:+.3f}  ({_kappa_interp(k)})")
        else:
            lines.append("  No valid pairs for confusion matrix.")
    else:
        lines.append("  No LLM outputs loaded.")

    # ------------------------------------------------------------------
    # 4. Per-cue-type agreement breakdown
    # ------------------------------------------------------------------
    h("4. Per-cue-type breakdown: rule label vs LLM majority vote")
    if len(valid_models) >= 1:
        all_llm_labels = [llm_outputs[m] for m in valid_models]
        mvote = majority_vote(all_llm_labels)

        cue_groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for gid, ct, r, m in zip(gold_ids, cue_types, rule_labels, mvote):
            if r >= 0 and m >= 0:
                cue_groups[ct].append((r, m))

        lines.append(f"  {'Cue type':<45}  {'n':>4}  {'agree%':>7}  {'κ':>6}  {'LLM dist'}")
        for ct in [
            "none",
            "speculation_modal_probable_",
            "speculation_hypo_doxastic _",
            "speculation_hypo_condition _",
            "speculation_hypo_investigation _",
        ]:
            pairs = cue_groups.get(ct, [])
            if not pairs:
                continue
            r_vals, m_vals = zip(*pairs)
            pa = percent_agreement(list(r_vals), list(m_vals))
            k = cohen_kappa(list(r_vals), list(m_vals))
            llm_dist = Counter(LABELS[v] for v in m_vals)
            dist_str = " ".join(f"{LABELS[i]}:{llm_dist.get(LABELS[i], 0)}" for i in range(3))
            display = CUE_DISPLAY.get(ct, ct)
            lines.append(f"  {display:<45}  {len(pairs):>4}  {pa:>7.1%}  {k:>+6.3f}  {dist_str}")

        # Highlight the conditional case
        lines.append("")
        cond_pairs = cue_groups.get("speculation_hypo_condition _", [])
        if cond_pairs:
            r_vals, m_vals = zip(*cond_pairs)
            llm_dist = Counter(LABELS[v] for v in m_vals)
            lines.append(
                f"  !! hypo_condition detail: LLMs labeled "
                f"asserted={llm_dist.get('asserted', 0)}, "
                f"hedged={llm_dist.get('hedged', 0)}, "
                f"speculative={llm_dist.get('speculative', 0)}"
            )
            dominant = llm_dist.most_common(1)[0][0]
            rule_label = "speculative"
            if dominant != rule_label:
                lines.append(
                    f"  !! LLM majority label for conditionals = '{dominant}' "
                    f"(rule says '{rule_label}') — consider revising mapping."
                )
            else:
                lines.append(
                    f"  !! LLM majority agrees with rule ('{rule_label}') on conditionals."
                )

    # ------------------------------------------------------------------
    # 5. Decision recommendation
    # ------------------------------------------------------------------
    h("5. Decision recommendation")
    overall_kappas = list(kappas_vs_rule.values())
    if overall_kappas:
        avg_k = sum(overall_kappas) / len(overall_kappas)
        cond_pairs = cue_groups.get("speculation_hypo_condition _", []) if len(valid_models) >= 1 else []
        cond_ok = True
        if cond_pairs:
            r_vals, m_vals = zip(*cond_pairs)
            cond_k = cohen_kappa(list(r_vals), list(m_vals))
            cond_dominant = Counter(LABELS[v] for v in m_vals).most_common(1)[0][0]
            cond_ok = (cond_dominant == "speculative")

        lines.append(f"  Average LLM-vs-rule κ:  {avg_k:+.3f}")
        lines.append(f"  Conditionals consistent: {cond_ok}")
        lines.append("")
        if avg_k >= 0.70 and cond_ok:
            lines.append("  RECOMMENDATION: SHIP the current mapping.")
            lines.append("  Both overall agreement and conditional subset are acceptable.")
        elif avg_k >= 0.70 and not cond_ok:
            lines.append("  RECOMMENDATION: SHIP with conditional revision.")
            lines.append("  Overall agreement is acceptable, but conditionals show systematic")
            lines.append("  disagreement. Consider mapping hypo_condition → 'hedged' or")
            lines.append("  splitting conditionals into their own class.")
        else:
            lines.append("  RECOMMENDATION: DO NOT SHIP — revisit mapping.")
            lines.append(f"  Overall κ = {avg_k:.3f} is below the 0.70 threshold.")
            lines.append("  Review the confusion matrix for systematic error patterns.")
    else:
        lines.append("  No LLM outputs to compare yet.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default=str(Path(__file__).parent / "outputs"),
    )
    parser.add_argument(
        "--sentences",
        default=str(Path(__file__).parent / "sentences.jsonl"),
    )
    parser.add_argument(
        "--report",
        default=str(Path(__file__).parent / "agreement_report.txt"),
    )
    args = parser.parse_args()

    sentences_path = Path(args.sentences)
    outputs_dir = Path(args.outputs_dir)
    report_path = Path(args.report)

    if not sentences_path.exists():
        print(f"ERROR: {sentences_path} not found. Run build_prompts.py first.")
        sys.exit(1)

    print(f"Loading gold labels from {sentences_path}...")
    gold = load_gold(sentences_path)
    print(f"  {len(gold)} sentences loaded.")

    llm_files = sorted(outputs_dir.glob("*.jsonl"))
    if not llm_files:
        print(f"\nNo .jsonl files found in {outputs_dir}.")
        print("Drop LLM output files there (see outputs/README.txt) and re-run.")
        sys.exit(0)

    print(f"\nLoading {len(llm_files)} LLM output file(s)...")
    llm_outputs: dict[str, list[int]] = {}
    for f in llm_files:
        model_name, labels = load_llm_output(f, gold)
        llm_outputs[model_name] = labels
        n_valid = sum(1 for l in labels if l >= 0)
        print(f"  {model_name}: {n_valid}/{len(gold)} valid labels")

    report = analyze(gold, llm_outputs)
    print(report)

    report_path.write_text(report)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
