"""
Data package for the IRDM retrieval-evaluation pipeline.

Layout
------
Raw / authored data
    opportunities.py      5 listings for the v2 benchmark (data as code)
    student_profiles.csv  20 labelled-by-design student profiles (data as CSV)
    csv/main_demo/        12 listings, 5 profiles - the `main` branch demo, verbatim
    csv/heldout/          30 listings, 24 profiles - held-out comparison set;
                          single-rater illustrative labels, fixed before any
                          system was run on it

Code that turns data into the canonical schema
    schema.py             Opportunity / Query / Dataset dataclasses + validation
    prototype_dataset.py  v1  - 8 listings, 4 short vague queries, hand-graded
    synthetic_v2_dataset.py v2 - 5 listings, 20 profiles, match_type-derived labels
    csv_dataset.py        any folder of opportunities/profiles/relevance CSVs
    datasets.py           registry: name -> loader

Usage
-----
    from data import load_dataset, load_csv_dataset
    dataset = load_dataset("prototype")            # preserves prior results
    dataset = load_dataset("synthetic_v2")         # the 5x4 benchmark
    dataset = load_dataset("synthetic_v2", match_types=("semantic_analogy",))
    dataset = load_csv_dataset("path/to/folder")   # e.g. real VT listings
"""

from data.csv_dataset import load_csv_dataset
from data.datasets import DEFAULT_DATASET, available_datasets, load_dataset
from data.schema import Dataset, Opportunity, Query

__all__ = [
    "DEFAULT_DATASET",
    "Dataset",
    "Opportunity",
    "Query",
    "available_datasets",
    "load_csv_dataset",
    "load_dataset",
]
