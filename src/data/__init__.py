"""Public API for the data loading layer."""

from .dataset import CombinedDataset, MAGPIEDataset
from .downloader import download, download_all, list_cached
from .registry import REGISTRY, DatasetMeta, LabelColumn, TaskType
from .unifier import SentimentScores, has_projector, project_label, projectable_datasets


def load_dataset(
    dataset_id: str,
    tokenizer=None,
    max_length: int = 128,
    cache_dir=None,
    label_col_filter=None,
) -> MAGPIEDataset:
    """Convenience wrapper: download if needed, then return a MAGPIEDataset."""
    return MAGPIEDataset(
        dataset_id,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        download_if_missing=True,
        label_col_filter=label_col_filter,
    )


__all__ = [
    "load_dataset",
    "MAGPIEDataset",
    "CombinedDataset",
    "download",
    "download_all",
    "list_cached",
    "REGISTRY",
    "DatasetMeta",
    "LabelColumn",
    "TaskType",
    "SentimentScores",
    "project_label",
    "has_projector",
    "projectable_datasets",
]
