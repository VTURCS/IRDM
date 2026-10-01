"""
Data package for the IRDM retrieval-evaluation pipeline.

Layout
------
Raw / authored data
    opportunities.py      5 listings for the v2 benchmark (data as code)
    student_profiles.csv  20 labelled-by-design student profiles (data as CSV)

Code that turns data into the canonical schema
    schema.py             Opportunity / Query / Dataset dataclasses + validation
    prototype_dataset.py  v1  - 8 listings, 4 short vague queries, hand-graded
    synthetic_v2_dataset.py v2 - 5 listings, 20 profiles, match_type-derived labels
    datasets.py           registry: name -> loader

Usage
-----
    from data import load_dataset
    dataset = load_dataset("prototype")            # preserves prior results
    dataset = load_dataset("synthetic_v2")         # the 5x4 benchmark
    dataset = load_dataset("synthetic_v2", match_types=("semantic_analogy",))
"""

from data.datasets import DEFAULT_DATASET, available_datasets, load_dataset
from data.schema import Dataset, Opportunity, Query

__all__ = [
    "DEFAULT_DATASET",
    "Dataset",
    "Opportunity",
    "Query",
    "available_datasets",
    "load_dataset",
]
