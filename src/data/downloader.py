"""
Download MAGPIE preprocessed CSVs from GitHub and cache them locally.

Each dataset lives at:
  https://raw.githubusercontent.com/Media-Bias-Group/magpie-multi-task/main/datasets/{id}/preprocessed.csv
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

from .registry import REGISTRY, DatasetMeta

_GITHUB_BASE = (
    "https://raw.githubusercontent.com/Media-Bias-Group/"
    "magpie-multi-task/main/datasets/{dataset_id}/preprocessed.csv"
)

_DEFAULT_CACHE = Path.home() / ".cache" / "magpie"


def get_cache_dir(cache_dir: Optional[str | Path] = None) -> Path:
    path = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    path.mkdir(parents=True, exist_ok=True)
    return path


def csv_path(dataset_id: str, cache_dir: Optional[str | Path] = None) -> Path:
    return get_cache_dir(cache_dir) / dataset_id / "preprocessed.csv"


def download(
    dataset_id: str,
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """Download the preprocessed CSV for one dataset if not already cached.

    Returns the local path to the CSV.
    """
    if dataset_id not in REGISTRY:
        raise ValueError(f"Unknown dataset id '{dataset_id}'. Check registry.py.")

    dest = csv_path(dataset_id, cache_dir)
    if dest.exists() and not force:
        return dest

    url = _GITHUB_BASE.format(dataset_id=dataset_id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_id} → {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download {dataset_id} (HTTP {exc.code}). "
            "The dataset may require separate licensing — check its README at "
            f"https://github.com/Media-Bias-Group/magpie-multi-task/tree/main/datasets/{dataset_id}"
        ) from exc

    return dest


def download_all(
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
    news_only: bool = False,
    skip_sequence_label: bool = True,
) -> dict[str, Path]:
    """Download every dataset in the registry.

    Args:
        cache_dir: Override the default cache location.
        force: Re-download even if already cached.
        news_only: Restrict to datasets flagged news_relevant=True.
        skip_sequence_label: Skip datasets that are purely sequence-labeling tasks
            (their label format is different and they need special handling).

    Returns:
        Mapping of dataset_id -> local CSV path.
    """
    from .registry import TaskType

    results: dict[str, Path] = {}
    for dataset_id, meta in REGISTRY.items():
        if news_only and not meta.news_relevant:
            continue
        if skip_sequence_label and all(
            lc.task_type == TaskType.SEQUENCE_LABEL for lc in meta.label_columns
        ):
            continue
        try:
            results[dataset_id] = download(dataset_id, cache_dir=cache_dir, force=force)
        except RuntimeError as exc:
            print(f"  WARNING: {exc}")

    return results


def list_cached(cache_dir: Optional[str | Path] = None) -> list[str]:
    """Return ids of datasets already present in the cache."""
    root = get_cache_dir(cache_dir)
    return [
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "preprocessed.csv").exists()
    ]
