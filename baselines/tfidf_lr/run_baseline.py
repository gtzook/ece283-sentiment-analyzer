#!/usr/bin/env python3
"""
baselines/tfidf_lr/run_baseline.py — MAGPIE TF-IDF + Logistic Regression Baseline

Classical baseline using identical test splits as the unified neural evaluation
(seed=42, 80/10/10 stratified). Covers all three MAGPIE tasks:
  Task 1 — Epistemic certainty (asserted / hedged / speculative)
  Task 2 — Political bias (binary: biased/neutral; multiclass: 5-class credibility)
  Task 3 — Emotional framing (11-label multi-label)

Outputs (baselines/tfidf_lr/):
  metrics_baseline.json          — all scores with 95% bootstrap CIs
  top_features_per_class.json    — top 20 n-grams per class by log-odds
  confusion_matrices/            — PNG + CSV per task
  vectorizers/                   — fitted .pkl per task
  cv_logs/                       — per-fold CV scores
  baseline_report.md             — summary table vs majority-class floor

Usage (from project root):
    /home/gzook/venv/bin/python baselines/tfidf_lr/run_baseline.py
    /home/gzook/venv/bin/python baselines/tfidf_lr/run_baseline.py \\
        --epistemic-config models/epistemic/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Directory layout ──────────────────────────────────────────────────────────

BASE_DIR   = PROJECT_ROOT / "baselines" / "tfidf_lr"
VEC_DIR    = BASE_DIR / "vectorizers"
CV_DIR     = BASE_DIR / "cv_logs"
CM_DIR     = BASE_DIR / "confusion_matrices"

for _d in (VEC_DIR, CV_DIR, CM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Shared TF-IDF / LR config ─────────────────────────────────────────────────

VECTORIZER_PARAMS: dict = dict(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=100_000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    token_pattern=r"(?u)\b\w\w+\b",
)
C_GRID = [0.01, 0.1, 1, 10]
SEED   = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    labels: list | None = None,
) -> dict:
    rng = np.random.default_rng(SEED)
    n   = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(float(f1_score(y_true[idx], y_pred[idx],
                                     average="macro", zero_division=0,
                                     labels=labels)))
    arr = np.array(scores)
    return {
        "mean":  float(arr.mean()),
        "ci_lo": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
    }


def majority_class_f1(y: np.ndarray) -> float:
    majority = int(np.bincount(y.astype(int)).argmax())
    pred     = np.full_like(y, majority)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def cv_tune_C(
    X_train,
    y_train,
    task_name: str,
    k: int = 5,
    lr_kwargs: dict | None = None,
) -> tuple[float, list]:
    """5-fold StratifiedKFold over C_GRID; returns (best_C, fold_log)."""
    lr_kwargs = lr_kwargs or {}
    skf   = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    log_rows = []
    best_C, best_score = C_GRID[0], -1.0

    for C in C_GRID:
        fold_scores = []
        for fold_i, (tr, va) in enumerate(skf.split(X_train, y_train)):
            clf = LogisticRegression(
                C=C, max_iter=1000, class_weight="balanced",
                solver="lbfgs",
                random_state=SEED, **lr_kwargs,
            )
            clf.fit(X_train[tr], y_train[tr])
            f1 = float(f1_score(y_train[va], clf.predict(X_train[va]),
                                average="macro", zero_division=0))
            fold_scores.append(f1)
            log_rows.append({"task": task_name, "C": C, "fold": fold_i, "macro_f1": f1})
        mean_f1 = float(np.mean(fold_scores))
        log.info("  C=%-5s  cv macro-F1=%.4f", C, mean_f1)
        if mean_f1 > best_score:
            best_score, best_C = mean_f1, C

    log.info("  → best C=%s  (cv macro-F1=%.4f)", best_C, best_score)
    return best_C, log_rows


def save_cv_log(rows: list, name: str) -> None:
    path = CV_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    log.info("  cv log → %s", path)


def save_vectorizer(vec: TfidfVectorizer, name: str) -> None:
    path = VEC_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(vec, f)
    log.info("  vectorizer → %s", path)


def save_cm_png(
    cm: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="True", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
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


def top_features(
    vec: TfidfVectorizer,
    clf: LogisticRegression | OneVsRestClassifier,
    class_names: list[str],
    top_k: int = 20,
) -> dict:
    """Return {class_name: [{feature, log_odds}]} for top-k log-odds features."""
    feat_names = vec.get_feature_names_out()
    result: dict = {}

    if isinstance(clf, OneVsRestClassifier):
        # OneVsRestClassifier: one estimator per label
        for i, (cname, est) in enumerate(zip(class_names, clf.estimators_)):
            row = est.coef_[0]
            top_idx = np.argsort(row)[::-1][:top_k]
            result[cname] = [
                {"feature": feat_names[j], "log_odds": float(row[j])}
                for j in top_idx
            ]
        return result

    # Plain LogisticRegression
    coef = clf.coef_          # shape (n_seen_classes, n_features) or (1, n_features)
    seen_classes = list(clf.classes_)  # actual integer class indices seen at fit time
    n_seen = coef.shape[0]

    for i, cname in enumerate(class_names):
        # Check if this class was seen during training
        if n_seen == 1:
            # Binary case stored as single row (positive class)
            row = coef[0] * (1 if i == 1 else -1)
        elif i in seen_classes:
            row = coef[seen_classes.index(i)]
        else:
            # Class not seen during training — no discriminative features
            result[cname] = [{"feature": "(class not in training data)", "log_odds": 0.0}]
            continue
        top_idx = np.argsort(row)[::-1][:top_k]
        result[cname] = [
            {"feature": feat_names[j], "log_odds": float(row[j])}
            for j in top_idx
        ]
    return result


# ── Task 1 — Epistemic Certainty ──────────────────────────────────────────────

EPISTEMIC_LABELS = ["asserted", "hedged", "speculative"]


def _tokenize_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Return (char_start, char_end, word) for each whitespace-bounded token."""
    return [(m.start(), m.end(), m.group())
            for m in re.finditer(r'\S+', text)]


def _is_cue(char_start: int, char_end: int, cue_spans: list) -> bool:
    return any(cs < char_end and ce > char_start for cs, ce in cue_spans)


def _span_f1(pred_labels: list[int], true_labels: list[int]) -> float:
    """Exact-span F1: groups adjacent 1s into spans and compares."""
    def to_spans(labels):
        spans, in_span, start = set(), False, None
        for i, l in enumerate(labels):
            if l == 1 and not in_span:
                start, in_span = i, True
            elif l == 0 and in_span:
                spans.add((start, i)); in_span = False
        if in_span:
            spans.add((start, len(labels)))
        return spans

    pred_spans = to_spans(pred_labels)
    true_spans = to_spans(true_labels)
    if not pred_spans and not true_spans:
        return 1.0
    if not pred_spans or not true_spans:
        return 0.0
    tp = len(pred_spans & true_spans)
    prec = tp / len(pred_spans)
    rec  = tp / len(true_spans)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _build_window_examples(
    token_examples: list,
    window: int = 2,
) -> tuple[list[str], list[int]]:
    """Create (context_text, label) pairs for each token in every example."""
    texts, labels = [], []
    for ex in token_examples:
        words_with_offsets = _tokenize_with_offsets(ex.text)
        words = [w for _, _, w in words_with_offsets]
        for i, (cs, ce, _) in enumerate(words_with_offsets):
            ctx = " ".join(words[max(0, i - window): i + window + 1])
            label = int(_is_cue(cs, ce, ex.cue_spans))
            texts.append(ctx)
            labels.append(label)
    return texts, labels


def run_task1(ep_cfg: dict) -> dict:
    """Train and evaluate Task 1 — Epistemic Certainty."""
    log.info("\n═══ Task 1: Epistemic Certainty ═══")

    from models.epistemic.train import load_all_data, doc_level_split
    from models.epistemic.data import load_bioscope

    data = load_all_data(ep_cfg)
    sent_train = data["sent_train"]
    sent_test  = data["sent_test"]

    # ── Sentence-level: fit TF-IDF + LR ──────────────────────────────────────
    log.info("Fitting sentence-level TF-IDF + LR …")
    X_texts_tr = [ex.text  for ex in sent_train]
    y_tr        = np.array([ex.label for ex in sent_train])
    X_texts_te  = [ex.text  for ex in sent_test]
    y_te        = np.array([ex.label for ex in sent_test])

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_tr = vec.fit_transform(X_texts_tr)
    X_te = vec.transform(X_texts_te)

    log.info("CV tuning C (sentence head) …")
    best_C, cv_rows = cv_tune_C(X_tr, y_tr, "task1_epistemic_sent")
    save_cv_log(cv_rows, "task1_epistemic_sent")

    clf = LogisticRegression(
        C=best_C, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    y_pred_sent = clf.predict(X_te)

    save_vectorizer(vec, "task1_epistemic_sent")

    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_te, y_pred_sent, labels=[0, 1, 2], zero_division=0
    )
    macro_f1 = float(f1_score(y_te, y_pred_sent, average="macro", zero_division=0))
    ci       = bootstrap_macro_f1(y_te, y_pred_sent)
    cm_sent  = confusion_matrix(y_te, y_pred_sent, labels=[0, 1, 2])
    cm_path  = CM_DIR / "task1_epistemic.png"
    save_cm_png(cm_sent, EPISTEMIC_LABELS, cm_path, "Epistemic Certainty — TF-IDF+LR")
    save_cm_csv(cm_sent, EPISTEMIC_LABELS, cm_path.with_suffix(".csv"))

    log.info("  sent macro-F1=%.4f  (best C=%s)", macro_f1, best_C)

    per_class = {
        EPISTEMIC_LABELS[i]: {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1c[i]),
        }
        for i in range(3)
    }
    majority_f1 = majority_class_f1(y_te)

    sent_results = {
        "macro_f1":       macro_f1,
        "bootstrap_ci":   ci,
        "majority_class_f1": majority_f1,
        "best_C":         best_C,
        "n_test":         int(len(y_te)),
        "per_class":      per_class,
        "confusion_matrix": cm_sent.tolist(),
    }

    # ── Token-level: BioScope sliding-window ──────────────────────────────────
    log.info("Building BioScope token-level examples …")
    bio_cfg = ep_cfg["data"]
    bioscope_paths = bio_cfg.get("bioscope_xmls", [])

    span_f1_result: dict = {}
    if bioscope_paths:
        all_bio_tok: list = []
        for path in bioscope_paths:
            all_bio_tok.extend(load_bioscope(path))

        # Split by doc_id (same strategy as news tokens)
        from models.epistemic.train import doc_level_split as _dls
        bio_train, _, bio_test = _dls(
            all_bio_tok,
            train_frac=ep_cfg["data"].get("train_frac", 0.80),
            val_frac=ep_cfg["data"].get("val_frac", 0.10),
            seed=ep_cfg["training"]["seed"],
        )

        win_texts_tr, win_labels_tr = _build_window_examples(bio_train)
        win_texts_te, win_labels_te = _build_window_examples(bio_test)

        log.info("  bio train windows=%d  test windows=%d",
                 len(win_texts_tr), len(win_texts_te))

        vec_tok = TfidfVectorizer(**VECTORIZER_PARAMS)
        X_tok_tr = vec_tok.fit_transform(win_texts_tr)
        X_tok_te = vec_tok.transform(win_texts_te)
        y_tok_tr  = np.array(win_labels_tr)
        y_tok_te  = np.array(win_labels_te)

        log.info("  CV tuning C (token head) …")
        best_C_tok, cv_rows_tok = cv_tune_C(
            X_tok_tr, y_tok_tr, "task1_epistemic_tok"
        )
        save_cv_log(cv_rows_tok, "task1_epistemic_tok")

        clf_tok = LogisticRegression(
            C=best_C_tok, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=SEED,
        )
        clf_tok.fit(X_tok_tr, y_tok_tr)
        y_tok_pred = clf_tok.predict(X_tok_te)

        save_vectorizer(vec_tok, "task1_epistemic_tok")

        tok_macro_f1 = float(f1_score(y_tok_te, y_tok_pred,
                                      average="macro", zero_division=0))
        tok_cue_f1   = float(f1_score(y_tok_te, y_tok_pred,
                                      average="binary", pos_label=1, zero_division=0))

        # Span F1: aggregate over each BioScope test example
        ptr = 0
        span_f1_scores = []
        for ex in bio_test:
            words_with_offsets = _tokenize_with_offsets(ex.text)
            n_tok = len(words_with_offsets)
            if n_tok == 0:
                continue
            true_tok = [
                int(_is_cue(cs, ce, ex.cue_spans))
                for cs, ce, _ in words_with_offsets
            ]
            pred_tok = y_tok_pred[ptr: ptr + n_tok].tolist()
            span_f1_scores.append(_span_f1(pred_tok, true_tok))
            ptr += n_tok

        mean_span_f1 = float(np.mean(span_f1_scores)) if span_f1_scores else 0.0
        log.info("  tok macro-F1=%.4f  cue-F1=%.4f  span-F1=%.4f",
                 tok_macro_f1, tok_cue_f1, mean_span_f1)

        cm_tok = confusion_matrix(y_tok_te, y_tok_pred, labels=[0, 1])
        save_cm_png(cm_tok, ["non-cue", "cue"],
                    CM_DIR / "task1_epistemic_token.png",
                    "Epistemic Token Head — TF-IDF+LR")
        save_cm_csv(cm_tok, ["non-cue", "cue"],
                    CM_DIR / "task1_epistemic_token.csv")

        span_f1_result = {
            "token_macro_f1": tok_macro_f1,
            "cue_f1":         tok_cue_f1,
            "span_f1_bioscope": mean_span_f1,
            "best_C_token":   best_C_tok,
            "n_test_windows": int(len(y_tok_te)),
            "n_test_examples": len(bio_test),
        }
    else:
        log.warning("  No bioscope_xmls configured — skipping token-level eval")

    top_f = top_features(vec, clf, EPISTEMIC_LABELS)

    return {
        "sentence": sent_results,
        "token":    span_f1_result,
        "top_features": top_f,
    }


# ── Task 2 — Political Bias ───────────────────────────────────────────────────

BIAS_BINARY_LABELS     = ["neutral", "biased"]
BIAS_MULTICLASS_LABELS = [
    "highly_credible", "mostly_credible", "mixed",
    "mostly_unreliable", "highly_unreliable",
]


def _load_babe(cache_dir: str) -> pd.DataFrame:
    path = Path(cache_dir) / "10_BABE" / "preprocessed.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"]   = df["text"].astype(str)
    df["binary"] = df["label"].astype(int)   # 0=neutral, 1=biased
    return df[["text", "binary"]]


def _load_basil(cache_dir: str) -> pd.DataFrame:
    path = Path(cache_dir) / "9_BASIL" / "preprocessed.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    # label: 0=lexical bias, 1=informational bias, 2=non-biased
    df["binary"] = df["label"].apply(lambda x: 0 if int(x) == 2 else 1)
    return df[["text", "binary"]]


def _load_liar(cache_dir: str) -> pd.DataFrame:
    path = Path(cache_dir) / "72_LIAR" / "preprocessed.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    # Continuous truthfulness: 0.0=true (highly credible) → 1.0=pants-fire (unreliable)
    # Map to 5 equal-width bins
    score = df["label"].astype(float)
    df["multiclass"] = pd.cut(
        score,
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    return df[["text", "multiclass"]]


def _load_fakenewsnet(cache_dir: str) -> pd.DataFrame:
    path = Path(cache_dir) / "25_FakeNewsNet" / "preprocessed.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    # 0=real (highly credible→0), 1=fake (highly unreliable→4)
    df["multiclass"] = df["label"].apply(lambda x: 0 if int(x) == 0 else 4)
    return df[["text", "multiclass"]]


def _stratified_split_df(
    df: pd.DataFrame,
    label_col: str,
    train_frac: float = 0.80,
    val_frac: float   = 0.10,
    seed: int         = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 split of a DataFrame."""
    from sklearn.model_selection import train_test_split
    labels = df[label_col].values
    # First split off test
    idx_trainval, idx_test = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=1.0 - train_frac - val_frac,
            random_state=seed,
        ).split(df, labels)
    )
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test     = df.iloc[idx_test].reset_index(drop=True)

    labels_tv = df_trainval[label_col].values
    tv_total  = len(df_trainval)
    val_frac_of_tv = val_frac / (train_frac + val_frac)
    idx_tr, idx_va = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_frac_of_tv,
            random_state=seed,
        ).split(df_trainval, labels_tv)
    )
    return (
        df_trainval.iloc[idx_tr].reset_index(drop=True),
        df_trainval.iloc[idx_va].reset_index(drop=True),
        df_test,
    )


def run_task2_binary(cache_dir: str) -> dict:
    """Binary bias: neutral vs biased (BABE + BASIL)."""
    log.info("\n─── Task 2 Binary: neutral vs biased ───")

    babe  = _load_babe(cache_dir)
    basil = _load_basil(cache_dir)

    # Split BABE to get same test set as the unified model
    babe_tr, babe_va, babe_te = _stratified_split_df(babe, "binary")
    # Use all BASIL for training (no neutral model eval on BASIL)
    basil_tr, basil_va, _ = _stratified_split_df(basil, "binary")

    train_df = pd.concat([babe_tr, babe_va, basil_tr, basil_va],
                         ignore_index=True)
    test_df  = babe_te   # evaluate on BABE test (matches unified eval)

    X_texts_tr = train_df["text"].tolist()
    y_tr        = train_df["binary"].values.astype(int)
    X_texts_te  = test_df["text"].tolist()
    y_te        = test_df["binary"].values.astype(int)

    log.info("  train=%d (BABE+BASIL), test=%d (BABE only)",
             len(X_texts_tr), len(X_texts_te))

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_tr = vec.fit_transform(X_texts_tr)
    X_te = vec.transform(X_texts_te)

    log.info("  CV tuning C …")
    best_C, cv_rows = cv_tune_C(X_tr, y_tr, "task2_binary")
    save_cv_log(cv_rows, "task2_binary")

    clf = LogisticRegression(
        C=best_C, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]  # P(biased)

    save_vectorizer(vec, "task2_binary")

    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    ci       = bootstrap_macro_f1(y_te, y_pred)
    auc      = float(roc_auc_score(y_te, y_prob))
    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    save_cm_png(cm, BIAS_BINARY_LABELS, CM_DIR / "task2_binary.png",
                "Bias Binary — TF-IDF+LR")
    save_cm_csv(cm, BIAS_BINARY_LABELS, CM_DIR / "task2_binary.csv")

    log.info("  macro-F1=%.4f  AUC-ROC=%.4f  (best C=%s)",
             macro_f1, auc, best_C)

    return {
        "macro_f1":  macro_f1,
        "auc_roc":   auc,
        "bootstrap_ci": ci,
        "best_C":    best_C,
        "majority_class_f1": majority_class_f1(y_te),
        "n_test":    int(len(y_te)),
        "per_class": {
            BIAS_BINARY_LABELS[i]: {
                "precision": float(prec[i]),
                "recall":    float(rec[i]),
                "f1":        float(f1c[i]),
            }
            for i in range(2)
        },
        "confusion_matrix": cm.tolist(),
        "top_features": top_features(vec, clf, BIAS_BINARY_LABELS),
    }


def run_task2_multiclass(cache_dir: str) -> dict:
    """5-class credibility: LIAR + FakeNewsNet."""
    log.info("\n─── Task 2 Multiclass: credibility (LIAR + FakeNewsNet) ───")
    log.info("  NOTE: 5 classes represent credibility/reliability spectrum")
    log.info("  (LIAR truthfulness bins + FakeNewsNet real/fake polarity)")

    liar = _load_liar(cache_dir)
    fnn  = _load_fakenewsnet(cache_dir)

    liar_tr, liar_va, liar_te = _stratified_split_df(liar, "multiclass")
    fnn_tr,  fnn_va,  fnn_te  = _stratified_split_df(fnn,  "multiclass")

    train_df = pd.concat([liar_tr, liar_va, fnn_tr, fnn_va], ignore_index=True)
    test_df  = pd.concat([liar_te, fnn_te],                  ignore_index=True)

    # Only classes actually present in LIAR+FNN test are [0, 1, 2, 3, 4] for LIAR
    # and [0, 4] for FNN; combined test has all 5 bins except FNN skips 1-3
    present = sorted(test_df["multiclass"].unique().tolist())

    X_texts_tr = train_df["text"].tolist()
    y_tr        = train_df["multiclass"].values.astype(int)
    X_texts_te  = test_df["text"].tolist()
    y_te        = test_df["multiclass"].values.astype(int)

    log.info("  train=%d, test=%d, classes present=%s",
             len(X_texts_tr), len(X_texts_te), present)

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_tr = vec.fit_transform(X_texts_tr)
    X_te = vec.transform(X_texts_te)

    log.info("  CV tuning C …")
    best_C, cv_rows = cv_tune_C(X_tr, y_tr, "task2_multiclass")
    save_cv_log(cv_rows, "task2_multiclass")

    clf = LogisticRegression(
        C=best_C, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    save_vectorizer(vec, "task2_multiclass")

    macro_f1 = float(f1_score(y_te, y_pred, average="macro",
                               labels=list(range(5)), zero_division=0))
    ci       = bootstrap_macro_f1(y_te, y_pred, labels=list(range(5)))
    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=list(range(5)), zero_division=0
    )
    cm = confusion_matrix(y_te, y_pred, labels=list(range(5)))
    save_cm_png(cm, BIAS_MULTICLASS_LABELS, CM_DIR / "task2_multiclass.png",
                "Bias Multiclass (credibility) — TF-IDF+LR")
    save_cm_csv(cm, BIAS_MULTICLASS_LABELS, CM_DIR / "task2_multiclass.csv")

    log.info("  macro-F1=%.4f  (best C=%s)", macro_f1, best_C)

    return {
        "macro_f1":  macro_f1,
        "bootstrap_ci": ci,
        "best_C":    best_C,
        "majority_class_f1": majority_class_f1(y_te),
        "n_test":    int(len(y_te)),
        "class_labels": BIAS_MULTICLASS_LABELS,
        "label_note": (
            "Classes are a credibility spectrum derived from LIAR truthfulness "
            "scores (0.0=true→class 0, 1.0=pants-fire→class 4) and "
            "FakeNewsNet (real→0, fake→4). MAGPIE preprocessed LIAR lacks "
            "speaker party affiliation, so strict political orientation "
            "(left/right) is not supported."
        ),
        "per_class": {
            BIAS_MULTICLASS_LABELS[i]: {
                "precision": float(prec[i]),
                "recall":    float(rec[i]),
                "f1":        float(f1c[i]),
            }
            for i in range(5)
        },
        "confusion_matrix": cm.tolist(),
        "top_features": top_features(vec, clf, BIAS_MULTICLASS_LABELS),
    }


# ── Task 3 — Emotional Framing ────────────────────────────────────────────────

EMOTION_LABELS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]


def _multilabel_cv_tune_C(
    X_train,
    y_train: np.ndarray,
    task_name: str,
    k: int = 3,
) -> tuple[float, list]:
    """
    3-fold iterative-stratification CV for the multi-label emotion task.
    Tunes a single global C (per-label tuning = 44 CV runs — too slow).
    """
    from skmultilearn.model_selection import iterative_train_test_split

    n = X_train.shape[0]
    indices   = np.arange(n, dtype=float).reshape(-1, 1)
    lab_float = y_train.astype(float)

    # Build k fold assignments via successive 1/(k-i) holdouts
    fold_ids = np.full(n, -1, dtype=int)
    remaining_idx = np.arange(n)
    remaining_lab = lab_float.copy()

    for fold_i in range(k - 1):
        frac = 1.0 / (k - fold_i)
        tr_pos, _, va_pos, _ = iterative_train_test_split(
            remaining_idx.reshape(-1, 1).astype(float),
            remaining_lab,
            test_size=frac,
        )
        va_global = va_pos.astype(int).flatten()
        fold_ids[va_global] = fold_i
        tr_global = tr_pos.astype(int).flatten()
        remaining_idx = tr_global
        remaining_lab = lab_float[tr_global]

    # Assign the leftover to the last fold
    fold_ids[fold_ids == -1] = k - 1

    log_rows: list = []
    best_C, best_score = C_GRID[0], -1.0

    for C in C_GRID:
        fold_scores = []
        for fold_i in range(k):
            va_idx = np.where(fold_ids == fold_i)[0]
            tr_idx = np.where(fold_ids != fold_i)[0]
            ovr = OneVsRestClassifier(
                LogisticRegression(
                    C=C, max_iter=1000, class_weight="balanced",
                    solver="lbfgs", random_state=SEED,
                ),
                n_jobs=-1,
            )
            ovr.fit(X_train[tr_idx], y_train[tr_idx])
            y_va_pred = ovr.predict(X_train[va_idx])
            f1 = float(f1_score(y_train[va_idx], y_va_pred,
                                average="micro", zero_division=0))
            fold_scores.append(f1)
            log_rows.append({
                "task": task_name, "C": C,
                "fold": fold_i, "micro_f1": f1,
            })
        mean_f1 = float(np.mean(fold_scores))
        log.info("  C=%-5s  cv micro-F1=%.4f", C, mean_f1)
        if mean_f1 > best_score:
            best_score, best_C = mean_f1, C

    log.info("  → best C=%s  (cv micro-F1=%.4f)", best_C, best_score)
    return best_C, log_rows


def run_task3(magpie_data_dir: str, hf_cache_dir: str) -> dict:
    """11-label multi-label emotional framing."""
    log.info("\n═══ Task 3: Emotional Framing ═══")

    from models.emotion.config import EmotionalFramingConfig
    from models.emotion.data   import load_and_split

    em_cfg = EmotionalFramingConfig()
    em_cfg.magpie_data_dir = magpie_data_dir
    em_cfg.hf_cache_dir    = hf_cache_dir
    em_cfg.seed            = SEED

    log.info("Loading and splitting emotion data …")
    dataset_dict = load_and_split(em_cfg)

    def _extract(split_name: str):
        ds = dataset_dict[split_name]
        texts  = list(ds["text"])
        labels = np.array(list(ds["labels"]), dtype=np.int32)
        return texts, labels

    tr_texts, y_tr = _extract("train")
    # combine train+dev for final fit (dev was used for threshold tuning)
    dev_texts, y_dev = _extract("dev")
    te_texts,  y_te  = _extract("test")

    train_texts_all = tr_texts + dev_texts
    y_tr_all        = np.vstack([y_tr, y_dev])

    log.info("  train+dev=%d  test=%d  labels=%d",
             len(train_texts_all), len(te_texts), y_tr.shape[1])

    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_tr_all = vec.fit_transform(train_texts_all)
    X_tr_cv  = vec.transform(tr_texts)          # CV uses train only
    X_te     = vec.transform(te_texts)

    log.info("  CV tuning C (multi-label) …")
    best_C, cv_rows = _multilabel_cv_tune_C(X_tr_cv, y_tr, "task3_emotion")
    save_cv_log(cv_rows, "task3_emotion")

    # Final model trained on train+dev
    clf = OneVsRestClassifier(
        LogisticRegression(
            C=best_C, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=SEED,
        ),
        n_jobs=-1,
    )
    clf.fit(X_tr_all, y_tr_all)

    save_vectorizer(vec, "task3_emotion")

    # Decision threshold sweep on dev set
    X_dev = vec.transform(dev_texts)
    probs_dev = clf.predict_proba(X_dev)

    best_thr, best_thr_f1 = 0.5, -1.0
    for thr in [0.3, 0.4, 0.5]:
        y_dev_pred = (probs_dev >= thr).astype(int)
        f1_thr = float(f1_score(y_dev, y_dev_pred, average="micro", zero_division=0))
        log.info("  dev threshold=%.1f  micro-F1=%.4f", thr, f1_thr)
        if f1_thr > best_thr_f1:
            best_thr_f1, best_thr = f1_thr, thr

    log.info("  → best threshold=%.1f  (dev micro-F1=%.4f)", best_thr, best_thr_f1)

    # Test evaluation
    probs_te = clf.predict_proba(X_te)
    y_pred   = (probs_te >= best_thr).astype(int)

    micro_f1 = float(f1_score(y_te, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    hl       = float(hamming_loss(y_te, y_pred))

    # Bootstrap CI on macro-F1
    rng   = np.random.default_rng(SEED)
    n_te  = len(y_te)
    boot  = np.array([
        float(f1_score(y_te[idx], y_pred[idx], average="macro", zero_division=0))
        for idx in (rng.integers(0, n_te, n_te) for _ in range(1000))
    ])
    ci = {"mean": float(boot.mean()),
          "ci_lo": float(np.percentile(boot, 2.5)),
          "ci_hi": float(np.percentile(boot, 97.5))}

    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=list(range(11)), zero_division=0
    )
    per_emotion = {
        lbl: {
            "f1":         float(f1_arr[i]),
            "precision":  float(prec_arr[i]),
            "recall":     float(rec_arr[i]),
            "prevalence": float(y_te[:, i].mean()),
        }
        for i, lbl in enumerate(EMOTION_LABELS)
    }

    log.info("  micro-F1=%.4f  macro-F1=%.4f  hamming=%.4f",
             micro_f1, macro_f1, hl)

    # Confusion matrices per emotion (binary)
    for i, emo in enumerate(EMOTION_LABELS):
        cm_e = confusion_matrix(y_te[:, i], y_pred[:, i], labels=[0, 1])
        save_cm_png(cm_e, [f"~{emo}", emo],
                    CM_DIR / f"task3_emotion_{emo}.png",
                    f"Emotion {emo} — TF-IDF+LR")
        save_cm_csv(cm_e, [f"~{emo}", emo],
                    CM_DIR / f"task3_emotion_{emo}.csv")

    top_f = top_features(vec, clf, EMOTION_LABELS)

    return {
        "micro_f1":    micro_f1,
        "macro_f1":    macro_f1,
        "hamming_loss": hl,
        "bootstrap_ci": ci,
        "best_threshold": best_thr,
        "best_C":      best_C,
        "n_test":      int(len(y_te)),
        "per_emotion": per_emotion,
        "top_features": top_f,
    }


# ── Output writers ────────────────────────────────────────────────────────────

def write_metrics_json(
    task1: dict, task2_bin: dict, task2_mc: dict, task3: dict
) -> None:
    payload = {
        "task1_epistemic": {
            "sentence": task1["sentence"],
            "token":    task1["token"],
        },
        "task2_bias_binary":     {k: v for k, v in task2_bin.items()
                                  if k != "top_features"},
        "task2_bias_multiclass": {k: v for k, v in task2_mc.items()
                                  if k != "top_features"},
        "task3_emotion":         {k: v for k, v in task3.items()
                                  if k != "top_features"},
    }
    path = BASE_DIR / "metrics_baseline.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("metrics → %s", path)


def write_top_features_json(
    task1: dict, task2_bin: dict, task2_mc: dict, task3: dict
) -> None:
    payload = {
        "task1_epistemic":       task1["top_features"],
        "task2_bias_binary":     task2_bin["top_features"],
        "task2_bias_multiclass": task2_mc["top_features"],
        "task3_emotion":         task3["top_features"],
    }
    path = BASE_DIR / "top_features_per_class.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("top features → %s", path)


def write_baseline_report(
    task1: dict, task2_bin: dict, task2_mc: dict, task3: dict
) -> None:
    def _ci(d: dict) -> str:
        c = d.get("bootstrap_ci", {})
        mf1 = d.get("macro_f1", d.get("micro_f1", 0.0))
        return f"{mf1:.4f} [{c.get('ci_lo', 0):.3f}, {c.get('ci_hi', 0):.3f}]"

    def _pct(x): return f"{100 * x:.1f}%"

    lines = [
        "# MAGPIE — TF-IDF + Logistic Regression Baseline Report\n\n",
        "> Classical floor model. Any neural approach that cannot clearly beat "
        "these scores has not justified its added complexity.\n\n",
    ]

    # ── Summary table ─────────────────────────────────────────────────────────
    lines += [
        "## Summary: Baseline vs. Majority-Class Floor\n\n",
        "| Task | Metric | TF-IDF+LR | Majority-Class Floor | Best C | n_test |\n",
        "|------|--------|-----------|---------------------|--------|--------|\n",
    ]

    s1  = task1["sentence"]
    t1t = task1["token"]
    lines.append(
        f"| Task 1 — Epistemic (sent) | Macro-F1 (95% CI) "
        f"| {_ci(s1)} | {_pct(s1['majority_class_f1'])} "
        f"| {s1['best_C']} | {s1['n_test']} |\n"
    )
    if t1t:
        lines.append(
            f"| Task 1 — Epistemic (token) | Span-F1 "
            f"| {t1t.get('span_f1_bioscope', 0):.4f} | — "
            f"| {t1t.get('best_C_token', '?')} "
            f"| {t1t.get('n_test_examples', '?')} examples |\n"
        )
    lines.append(
        f"| Task 2 — Bias Binary | Macro-F1 (95% CI) "
        f"| {_ci(task2_bin)} | {_pct(task2_bin['majority_class_f1'])} "
        f"| {task2_bin['best_C']} | {task2_bin['n_test']} |\n"
    )
    lines.append(
        f"| Task 2 — Bias Binary | AUC-ROC "
        f"| {task2_bin['auc_roc']:.4f} | 0.5000 | — | — |\n"
    )
    lines.append(
        f"| Task 2 — Bias Multiclass | Macro-F1 (95% CI) "
        f"| {_ci(task2_mc)} | {_pct(task2_mc['majority_class_f1'])} "
        f"| {task2_mc['best_C']} | {task2_mc['n_test']} |\n"
    )
    lines.append(
        f"| Task 3 — Emotion | Micro-F1 "
        f"| {task3['micro_f1']:.4f} | — "
        f"| {task3['best_C']} | {task3['n_test']} |\n"
    )
    lines.append(
        f"| Task 3 — Emotion | Macro-F1 (95% CI) "
        f"| {_ci(task3)} | — | — | — |\n"
    )
    lines.append(
        f"| Task 3 — Emotion | Hamming Loss "
        f"| {task3['hamming_loss']:.4f} | — | — | — |\n"
    )
    lines.append("\n")

    # ── Task 1 detail ─────────────────────────────────────────────────────────
    lines += ["## Task 1 — Epistemic Certainty\n\n",
              "**Data**: Wikipedia Uncertainty corpus + FactBank (Szeged XML)  \n",
              "**Classes**: asserted (0), hedged (1), speculative (2)  \n",
              "**Split**: document-level 80/10/10, seed=42\n\n",
              "### Per-class scores (sentence head)\n\n",
              "| Class | Precision | Recall | F1 |\n",
              "|-------|-----------|--------|----|\n"]
    for cls, m in s1["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")

    if t1t:
        lines += [
            "\n### Token head — BioScope sliding-window (±2 tokens)\n\n",
            f"- Token macro-F1: **{t1t.get('token_macro_f1', 0):.4f}**\n",
            f"- Cue-F1 (binary): **{t1t.get('cue_f1', 0):.4f}**\n",
            f"- Span-F1 (BioScope): **{t1t.get('span_f1_bioscope', 0):.4f}**\n",
            f"- Best C (token model): {t1t.get('best_C_token', '?')}\n",
            f"- Test windows: {t1t.get('n_test_windows', '?')} | "
            f"Test examples: {t1t.get('n_test_examples', '?')}\n\n",
        ]

    lines += ["\n### Top discriminative n-grams (sentence head)\n\n"]
    for cls, feats in task1["top_features"].items():
        top5 = ", ".join(f"`{f['feature']}`" for f in feats[:5])
        lines.append(f"- **{cls}**: {top5}\n")
    lines.append("\n")

    # ── Task 2 detail ─────────────────────────────────────────────────────────
    lines += ["## Task 2 — Political Bias\n\n",
              "### Binary (biased / neutral)\n\n",
              "**Data**: BABE (train+val, ~2.9K) + BASIL (train+val, ~6.3K)  \n",
              "**Test**: BABE test split only (matches unified model eval)\n\n",
              "| Class | Precision | Recall | F1 |\n",
              "|-------|-----------|--------|----|\n"]
    for cls, m in task2_bin["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")

    lines += ["\n**Top n-grams:**\n"]
    for cls, feats in task2_bin["top_features"].items():
        top5 = ", ".join(f"`{f['feature']}`" for f in feats[:5])
        lines.append(f"- **{cls}**: {top5}\n")

    lines += ["\n### Multiclass (5-class credibility spectrum)\n\n",
              f"> {task2_mc['label_note']}\n\n",
              "**Classes** (0→most credible, 4→least credible):\n",
              f"{', '.join(BIAS_MULTICLASS_LABELS)}\n\n",
              "| Class | Precision | Recall | F1 |\n",
              "|-------|-----------|--------|----|\n"]
    for cls, m in task2_mc["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")

    lines += ["\n**Top n-grams:**\n"]
    for cls, feats in task2_mc["top_features"].items():
        top5 = ", ".join(f"`{f['feature']}`" for f in feats[:5])
        lines.append(f"- **{cls}**: {top5}\n")
    lines.append("\n")

    # ── Task 3 detail ─────────────────────────────────────────────────────────
    lines += ["## Task 3 — Emotional Framing\n\n",
              "**Data**: GoEmotions + dair-ai/emotion + TweetEval + MAGPIE 84_emotion_tweets  \n",
              f"**Threshold**: {task3['best_threshold']:.1f} (tuned on dev set)  \n",
              f"**Best C**: {task3['best_C']}\n\n",
              "### Per-emotion F1\n\n",
              "| Emotion | Prevalence | Precision | Recall | F1 |\n",
              "|---------|-----------|-----------|--------|----|\n"]
    for emo, m in task3["per_emotion"].items():
        lines.append(
            f"| {emo} | {m['prevalence']:.3f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n"
        )

    lines += ["\n**Top n-grams per emotion:**\n"]
    for emo, feats in task3["top_features"].items():
        top5 = ", ".join(f"`{f['feature']}`" for f in feats[:5])
        lines.append(f"- **{emo}**: {top5}\n")
    lines.append("\n")

    # ── Artifact note ─────────────────────────────────────────────────────────
    lines += [
        "## Artifact / Leakage Notes\n\n",
        "Inspecting top discriminative features is mandatory for detecting dataset "
        "artifacts (e.g. source-name leakage in LIAR, near-duplicate sentences "
        "in BASIL). If source names (e.g. newspaper names) appear in top features, "
        "the model is learning domain identity rather than bias signal.\n",
    ]

    path = BASE_DIR / "baseline_report.md"
    with open(path, "w") as f:
        f.writelines(lines)
    log.info("report → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAGPIE TF-IDF + LR Baseline — all three tasks"
    )
    parser.add_argument(
        "--epistemic-config",
        default=str(PROJECT_ROOT / "models" / "epistemic" / "config.yaml"),
        help="Path to models/epistemic/config.yaml",
    )
    parser.add_argument(
        "--magpie-data-dir",
        default="/mldata/ece283-sentiment-analyzer",
        help="Root of downloaded MAGPIE CSVs",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="/mldata/ece283-sentiment-analyzer/hf_cache",
        help="HuggingFace dataset cache dir",
    )
    parser.add_argument(
        "--skip-task1", action="store_true",
        help="Skip Task 1 (epistemic)"
    )
    parser.add_argument(
        "--skip-task2", action="store_true",
        help="Skip Task 2 (bias)"
    )
    parser.add_argument(
        "--skip-task3", action="store_true",
        help="Skip Task 3 (emotion)"
    )
    args = parser.parse_args()

    log.info("Project root: %s", PROJECT_ROOT)
    log.info("Output dir:   %s", BASE_DIR)

    with open(args.epistemic_config) as f:
        ep_cfg = yaml.safe_load(f)
    # Patch training key for seed (train.py expects cfg["training"]["seed"])
    if "training" not in ep_cfg:
        ep_cfg["training"] = {}
    ep_cfg["training"].setdefault("seed", SEED)

    task1 = task2_bin = task2_mc = task3 = None

    # ── Task 1 ────────────────────────────────────────────────────────────────
    if not args.skip_task1:
        task1 = run_task1(ep_cfg)
        log.info("[Task 1 done] sent macro-F1=%.4f  span-F1=%.4f",
                 task1["sentence"]["macro_f1"],
                 task1["token"].get("span_f1_bioscope", float("nan")))
    else:
        log.info("Task 1 skipped")

    # ── Task 2 ────────────────────────────────────────────────────────────────
    if not args.skip_task2:
        task2_bin = run_task2_binary(args.magpie_data_dir)
        task2_mc  = run_task2_multiclass(args.magpie_data_dir)
        log.info("[Task 2 done] binary macro-F1=%.4f  multiclass macro-F1=%.4f",
                 task2_bin["macro_f1"], task2_mc["macro_f1"])
    else:
        log.info("Task 2 skipped")

    # ── Task 3 ────────────────────────────────────────────────────────────────
    if not args.skip_task3:
        task3 = run_task3(args.magpie_data_dir, args.hf_cache_dir)
        log.info("[Task 3 done] micro-F1=%.4f  macro-F1=%.4f  hamming=%.4f",
                 task3["micro_f1"], task3["macro_f1"], task3["hamming_loss"])
    else:
        log.info("Task 3 skipped")

    # ── Write outputs (only if all three tasks ran) ───────────────────────────
    if task1 and task2_bin and task2_mc and task3:
        write_metrics_json(task1, task2_bin, task2_mc, task3)
        write_top_features_json(task1, task2_bin, task2_mc, task3)
        write_baseline_report(task1, task2_bin, task2_mc, task3)
        log.info("\nAll outputs written to %s/", BASE_DIR)
    else:
        log.info("Partial run — skipping combined output files")
        if task1:
            log.info("  task1 → available in memory")
        if task2_bin and task2_mc:
            log.info("  task2 → available in memory")
        if task3:
            log.info("  task3 → available in memory")

    print("\n" + "=" * 64)
    print("TF-IDF + LR BASELINE — FINAL RESULTS")
    print("=" * 64)
    if task1:
        print(f"  Task 1 — Epistemic")
        print(f"    sent macro-F1 : {task1['sentence']['macro_f1']:.4f}  "
              f"(vs majority {task1['sentence']['majority_class_f1']:.4f})")
        if task1["token"]:
            print(f"    span-F1       : {task1['token'].get('span_f1_bioscope', 0):.4f}")
            print(f"    cue-F1        : {task1['token'].get('cue_f1', 0):.4f}")
    if task2_bin:
        print(f"  Task 2 — Bias Binary")
        print(f"    macro-F1 : {task2_bin['macro_f1']:.4f}  "
              f"AUC-ROC: {task2_bin['auc_roc']:.4f}")
    if task2_mc:
        print(f"  Task 2 — Bias Multiclass")
        print(f"    macro-F1 : {task2_mc['macro_f1']:.4f}")
    if task3:
        print(f"  Task 3 — Emotion")
        print(f"    micro-F1 : {task3['micro_f1']:.4f}  "
              f"macro-F1: {task3['macro_f1']:.4f}  "
              f"hamming: {task3['hamming_loss']:.4f}")
    print(f"\n  Outputs: {BASE_DIR}/")


if __name__ == "__main__":
    main()
