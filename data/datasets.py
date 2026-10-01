"""
Dataset registry.

The pipeline resolves a dataset by name through `load_dataset()`, so adding a
new benchmark means writing one loader module and registering it here - the
retrievers, metrics and report never change.

Datasets are keyed by `Dataset.name` (declared in each loader), not by the
registry key, so the name reported in exports always matches the data used.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from data import prototype_dataset, synthetic_v2_dataset
from data.schema import Dataset, Opportunity, Query

# Loader signature: (allowed_match_types) -> Dataset. Loaders that do not
# support label stratification simply ignore the argument.
_LOADERS: Dict[str, Callable[[Sequence[str] | None], Dataset]] = {
    "prototype": lambda match_types=None: prototype_dataset.load(),
    "synthetic_v2": lambda match_types=None: synthetic_v2_dataset.load(match_types),
}

DEFAULT_DATASET = "prototype"


def available_datasets() -> List[str]:
    """Names accepted by `--dataset`."""
    return list(_LOADERS)


def load_dataset(
    name: str,
    match_types: Sequence[str] | None = None,
) -> Dataset:
    """
    Load a dataset by name.

    Parameters
    ----------
    name
        One of `available_datasets()`.
    match_types
        Optional label-stratum filter (e.g. `("semantic_analogy",)`). Only the
        `synthetic_v2` loader honours this; it is ignored elsewhere.
    """
    if name not in _LOADERS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {', '.join(available_datasets())}"
        )
    return _LOADERS[name](match_types)


__all__ = [
    "DEFAULT_DATASET",
    "Dataset",
    "Opportunity",
    "Query",
    "available_datasets",
    "load_dataset",
]
