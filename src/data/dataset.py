"""
PyTorch Dataset for MAGPIE datasets.

Usage
-----
from src.data import load_dataset

ds = load_dataset("10_BABE", tokenizer=tokenizer)
# ds[0] -> {"input_ids": ..., "attention_mask": ..., "labels": {"label": tensor(1)}}

For multi-task training, combine datasets via CombinedDataset or DataLoader directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .downloader import csv_path, download
from .registry import REGISTRY, DatasetMeta, LabelColumn, TaskType


class MAGPIEDataset(Dataset):
    """Wraps one MAGPIE preprocessed CSV as a PyTorch Dataset.

    Each item is a dict::

        {
            "text":  str,
            "labels": {col_name: int | float | torch.Tensor},
        }

    If a tokenizer is provided the dict also contains ``input_ids`` and
    ``attention_mask`` (and ``token_type_ids`` if the tokenizer produces them).
    """

    def __init__(
        self,
        dataset_id: str,
        cache_dir: Optional[str | Path] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
        download_if_missing: bool = True,
        label_col_filter: Optional[list[str]] = None,
        transform: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        """
        Args:
            dataset_id: Must match a key in the registry (e.g. "10_BABE").
            cache_dir: Where CSVs are cached; defaults to ~/.cache/magpie.
            tokenizer: Any HuggingFace tokenizer. If None, raw text is returned.
            max_length: Tokenizer max sequence length.
            download_if_missing: Auto-download the CSV when not found in cache.
            label_col_filter: Only include these label columns (default: all).
            transform: Optional callable applied to each item after label encoding.
        """
        self.meta: DatasetMeta = REGISTRY[dataset_id]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        path = csv_path(dataset_id, cache_dir)
        if not path.exists():
            if download_if_missing:
                path = download(dataset_id, cache_dir)
            else:
                raise FileNotFoundError(
                    f"CSV for '{dataset_id}' not found at {path}. "
                    "Pass download_if_missing=True or call downloader.download() first."
                )

        self._df = pd.read_csv(path, low_memory=False)
        self._label_cols: list[LabelColumn] = [
            lc for lc in self.meta.label_columns
            if (label_col_filter is None or lc.col in label_col_filter)
            and lc.col in self._df.columns
        ]
        # Drop rows missing text or any required label
        required_cols = [self.meta.text_col] + [lc.col for lc in self._label_cols]
        self._df = self._df.dropna(subset=required_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df.iloc[idx]
        text = str(row[self.meta.text_col])

        item: dict[str, Any] = {"text": text}

        # Encode labels
        item["labels"] = {lc.col: self._encode_label(lc, row[lc.col]) for lc in self._label_cols}

        # Tokenize
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for key, val in enc.items():
                item[key] = val.squeeze(0)

        if self.transform:
            item = self.transform(item)

        return item

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_label(self, lc: LabelColumn, raw: Any) -> int | float | torch.Tensor:
        if lc.task_type == TaskType.REGRESSION:
            return torch.tensor(float(raw), dtype=torch.float32)

        if lc.task_type == TaskType.SEQUENCE_LABEL:
            # Sequence labels are returned as strings; downstream collation handles them
            return raw

        # Classification: apply label_map if defined, else cast to int
        if lc.label_map:
            canonical = lc.label_map[str(int(float(raw)))]
        else:
            canonical = int(raw)
        return torch.tensor(canonical, dtype=torch.long)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def dataset_id(self) -> str:
        return self.meta.id

    @property
    def label_columns(self) -> list[LabelColumn]:
        return self._label_cols

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first n rows of the raw dataframe for inspection."""
        return self._df.head(n)


class CombinedDataset(Dataset):
    """Concatenates multiple MAGPIEDatasets for joint multi-task sampling.

    Each item includes a ``dataset_id`` field so the model can route to
    the correct task head.

    Example::

        combined = CombinedDataset([
            MAGPIEDataset("10_BABE", tokenizer=tok),
            MAGPIEDataset("99_SST2", tokenizer=tok),
        ])
        loader = DataLoader(combined, batch_size=32, shuffle=True)
    """

    def __init__(self, datasets: list[MAGPIEDataset]) -> None:
        self._datasets = datasets
        self._cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self) -> list[int]:
        totals = []
        running = 0
        for ds in self._datasets:
            running += len(ds)
            totals.append(running)
        return totals

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx, sample_idx = self._find(idx)
        item = self._datasets[ds_idx][sample_idx]
        item["dataset_id"] = self._datasets[ds_idx].dataset_id
        return item

    def _find(self, idx: int) -> tuple[int, int]:
        for ds_idx, cum in enumerate(self._cumulative_sizes):
            if idx < cum:
                offset = self._cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0
                return ds_idx, idx - offset
        raise IndexError(idx)
