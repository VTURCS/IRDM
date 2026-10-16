"""
CSV dataset loader - the plain-file input format from the `main` branch.

A dataset is one folder holding three UTF-8 CSV files. This is the format to use
for real VT listings: no Python is needed to add or edit data, and the folder
can be run directly with `--data-dir`.

    opportunities.csv   id, title, description
                        optional: skills (separated by ';'), eligibility, department
    profiles.csv        id, query
                        optional: major, year, match_type
    relevance.csv       profile_id, opportunity_id, grade   (0, 1 or 2)

Searchable text follows the same rules as every other dataset: title,
description and skills are searched; eligibility and department are metadata
only (see `Config`). Any other columns are ignored.

relevance.csv must grade EVERY (profile, opportunity) pair, zeros included.
Recall@k divides by the number of relevant listings, so a missing label cannot
be treated as 0 - `Dataset` rejects incomplete label sets.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from data.schema import Dataset, Opportunity, Query

OPPORTUNITY_COLUMNS = ("id", "title", "description")
PROFILE_COLUMNS = ("id", "query")
RELEVANCE_COLUMNS = ("profile_id", "opportunity_id", "grade")
SKILL_SEPARATOR = ";"
VALID_GRADES = {"0": 0, "1": 1, "2": 2}


def _read_csv(path: Path, required: Sequence[str]) -> List[Dict[str, str]]:
    """Read one CSV, requiring the given columns to exist and be non-empty."""
    if not path.is_file():
        raise ValueError(f"{path}: file not found")
    # utf-8-sig so files saved from Excel (which prepends a BOM) still parse.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: contains no data rows")
    for line, row in enumerate(rows, 2):  # line 1 is the header
        for column in required:
            value = (row.get(column) or "").strip()
            if not value:
                raise ValueError(f"{path}:{line}: '{column}' must not be empty")
            row[column] = value
    return rows


def _optional(row: Dict[str, str], column: str) -> str:
    return (row.get(column) or "").strip()


def load_csv_dataset(
    directory: str | Path,
    name: str | None = None,
    match_types: Sequence[str] | None = None,
) -> Dataset:
    """
    Load a CSV dataset folder into the canonical `Dataset`.

    Parameters
    ----------
    directory
        Folder holding opportunities.csv, profiles.csv and relevance.csv.
    name
        Dataset name used in reports and as the output subdirectory. Defaults
        to the folder name.
    match_types
        Optional whitelist of `match_type` values; requires that column in
        profiles.csv. Labels of filtered-out profiles are skipped, and the
        corpus is never filtered, so recall denominators stay correct.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory}: not a directory")

    opportunity_rows = _read_csv(directory / "opportunities.csv", OPPORTUNITY_COLUMNS)
    profile_rows = _read_csv(directory / "profiles.csv", PROFILE_COLUMNS)
    relevance_rows = _read_csv(directory / "relevance.csv", RELEVANCE_COLUMNS)

    opportunities = [
        Opportunity(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            skills=[s.strip() for s in _optional(row, "skills").split(SKILL_SEPARATOR) if s.strip()],
            eligibility=_optional(row, "eligibility"),
            department=_optional(row, "department"),
        )
        for row in opportunity_rows
    ]
    all_profile_ids = {row["id"] for row in profile_rows}

    if match_types is not None:
        if "match_type" not in profile_rows[0]:
            raise ValueError(f"{directory / 'profiles.csv'}: no match_type column to filter on")
        available = {_optional(row, "match_type") for row in profile_rows}
        unknown = set(match_types) - available
        if unknown:
            raise ValueError(
                f"Unknown match_type filter {sorted(unknown)}; "
                f"this dataset has {sorted(t for t in available if t)}"
            )
        profile_rows = [row for row in profile_rows if _optional(row, "match_type") in set(match_types)]

    queries = [
        Query(
            profile_id=row["id"],
            text=row["query"],
            major=_optional(row, "major"),
            year=_optional(row, "year"),
            match_type=_optional(row, "match_type"),
        )
        for row in profile_rows
    ]
    kept_profile_ids = {q.profile_id for q in queries}
    opportunity_ids = {o.id for o in opportunities}

    relevance_map: Dict[Tuple[str, str], int] = {}
    relevance_path = directory / "relevance.csv"
    for line, row in enumerate(relevance_rows, 2):
        pair = (row["profile_id"], row["opportunity_id"])
        if pair[0] not in all_profile_ids or pair[1] not in opportunity_ids:
            raise ValueError(f"{relevance_path}:{line}: unknown profile or opportunity id {pair}")
        if row["grade"] not in VALID_GRADES:
            raise ValueError(f"{relevance_path}:{line}: grade must be 0, 1 or 2 (got '{row['grade']}')")
        if pair[0] not in kept_profile_ids:
            continue
        if pair in relevance_map:
            raise ValueError(f"{relevance_path}:{line}: duplicate label for {pair}")
        relevance_map[pair] = VALID_GRADES[row["grade"]]

    name = name or directory.name
    return Dataset(
        name=name,
        description=(
            f"CSV dataset '{name}': {len(opportunities)} listings, "
            f"{len(queries)} profiles, labels from relevance.csv."
        ),
        opportunities=opportunities,
        queries=queries,
        relevance_map=relevance_map,
        suggested_top_k=min(5, len(opportunities)),
    )
