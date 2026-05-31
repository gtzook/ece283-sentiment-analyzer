"""
Train / val / test split utilities for MAGPIEDataset.

Stratified split is used for classification tasks; random split for regression.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset, Subset

from .registry import TaskType


def stratified_split(
    dataset: Dataset,
    label_col: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Return (train, val, test) Subsets with class-balanced splits.

    Falls back to random split when the dataset has no discrete label column
    (e.g. regression-only datasets).
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)

    labels: Optional[np.ndarray] = None
    try:
        sample = dataset[0]
        if "labels" in sample and label_col in sample["labels"]:
            raw = sample["labels"][label_col]
            if hasattr(raw, "item") and isinstance(raw.item(), int):
                labels = np.array([
                    dataset[i]["labels"][label_col].item() for i in range(n)
                ])
    except Exception:
        pass

    if labels is not None:
        return _stratified(labels, dataset, train_frac, val_frac, rng)
    return _random(n, dataset, train_frac, val_frac, rng)


def _stratified(
    labels: np.ndarray,
    dataset: Dataset,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[Subset, Subset, Subset]:
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train: n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def _random(
    n: int,
    dataset: Dataset,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[Subset, Subset, Subset]:
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        Subset(dataset, idx[:n_train].tolist()),
        Subset(dataset, idx[n_train: n_train + n_val].tolist()),
        Subset(dataset, idx[n_train + n_val:].tolist()),
    )
