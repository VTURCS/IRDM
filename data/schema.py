"""
Canonical internal schema shared by every dataset loader.

The retrieval logic in `research_matching_pipeline.py` must not care which
dataset it is running against, so every loader normalises its source (embedded
literals, a Python module, a CSV) into these three types. Field-name differences
between datasets (`id` vs `opp_id`, `required_skills` vs `key_terms`) are
resolved inside the loaders, never in the pipeline.

Adding a new dataset therefore means writing one `load()` that returns a
`Dataset` - no changes to the retrievers, metrics or report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Opportunity:
    """One listing, normalised to the fields the retrievers actually use."""

    id: str
    title: str
    description: str
    skills: List[str] = field(default_factory=list)
    # Metadata only. Never used for filtering unless `metadata_filtering` is on,
    # and never part of the searchable text unless `include_eligibility_in_text`
    # is on. Both default to False so the three methods stay comparable.
    eligibility: str = ""
    department: str = ""


@dataclass(frozen=True)
class Query:
    """One student profile, reduced to the text the retrievers search with."""

    profile_id: str
    text: str
    major: str = ""
    year: str = ""
    # Only meaningful for the synthetic_v2 dataset; empty for the prototype.
    # Carried through so labelled analysis can slice results by match_type.
    match_type: str = ""
    # The listing this profile was authored against. Used for the negative
    # control: a `distractor` profile has NO relevant listings, so instead of
    # scoring it we ask how strongly each method was fooled into ranking this
    # listing highly. Empty when the dataset has no such notion.
    target_opp_id: str = ""


@dataclass(frozen=True)
class Dataset:
    """
    A complete evaluation input: corpus + queries + graded relevance labels.

    `relevance_map` maps `(profile_id, opportunity_id) -> grade` where
        grade 0 = unsuitable, 1 = partial match, 2 = strong match.

    IMPORTANT: the map must cover EVERY (query, opportunity) pair, not just the
    retrieved ones. Recall@k divides by the number of relevant listings in the
    ground truth, so a partial map silently inflates recall.
    """

    name: str
    description: str
    opportunities: List[Opportunity]
    queries: List[Query]
    relevance_map: Dict[Tuple[str, str], int]
    # Sensible default cut-off for this dataset. Small corpora want a smaller k:
    # a Top-5 list from an 8-item corpus is already 62% of everything available.
    suggested_top_k: int = 5

    def __post_init__(self) -> None:
        opp_ids = {o.id for o in self.opportunities}
        if len(opp_ids) != len(self.opportunities):
            raise ValueError(f"Dataset '{self.name}' contains duplicate opportunity ids.")

        query_ids = {q.profile_id for q in self.queries}
        if len(query_ids) != len(self.queries):
            raise ValueError(f"Dataset '{self.name}' contains duplicate query ids.")

        # Every query must be fully labelled against the whole corpus, otherwise
        # recall is not trustworthy.
        expected = {(q.profile_id, opp_id) for q in self.queries for opp_id in opp_ids}
        missing = expected - set(self.relevance_map)
        if missing:
            raise ValueError(
                f"Dataset '{self.name}' has {len(missing)} unlabelled (profile, opportunity) "
                f"pairs; relevance labels must cover the full corpus. "
                f"First few: {sorted(missing)[:5]}"
            )

        bad_grades = {g for g in self.relevance_map.values() if g not in (0, 1, 2)}
        if bad_grades:
            raise ValueError(f"Dataset '{self.name}' has out-of-range grades: {sorted(bad_grades)}")

        # ---------------------------------------------------------------------
        # Frozen index build (computed once, in __post_init__)
        # ---------------------------------------------------------------------
        # The hot path needs, for each query, the grades of every listing. The
        # original implementation rescanned the whole `relevance_map` per
        # (profile, method) pair, which is O(queries^2 * listings) and dominated
        # total runtime at scale (~2.4B ops for a 2000x200 corpus). Materialising
        # a dense (queries x listings) matrix once turns every lookup into an
        # O(1) array index.
        #
        # object.__setattr__ is required because the dataclass is frozen.
        object.__setattr__(self, "_opportunity_order", tuple(o.id for o in self.opportunities))
        object.__setattr__(self, "_query_order", tuple(q.profile_id for q in self.queries))
        object.__setattr__(
            self, "_opp_index", {opp_id: i for i, opp_id in enumerate(self._opportunity_order)}
        )
        object.__setattr__(
            self, "_query_index", {qid: i for i, qid in enumerate(self._query_order)}
        )

        grade_matrix = np.zeros((len(self._query_order), len(self._opportunity_order)), dtype=np.int8)
        for q_row, query_id in enumerate(self._query_order):
            for opp_col, opp_id in enumerate(self._opportunity_order):
                grade_matrix[q_row, opp_col] = self.relevance_map[(query_id, opp_id)]
        object.__setattr__(self, "_grade_matrix", grade_matrix)

        # Descending grade columns => the ideal DCG ordering for every query is
        # the same sort, so it can be computed once for the whole dataset.
        object.__setattr__(self, "_ideal_grades", np.sort(grade_matrix, axis=1)[:, ::-1])

    @property
    def opportunity_ids(self) -> List[str]:
        return list(self._opportunity_order)

    @property
    def query_ids(self) -> List[str]:
        return list(self._query_order)

    @property
    def grade_matrix(self) -> np.ndarray:
        """
        Dense (n_queries x n_listings) int8 grade matrix.

        Row/column order matches `query_ids` / `opportunity_ids`. Returned
        read-only: it is shared, cached state, and mutating it would corrupt
        every later metric computation in the process.
        """
        matrix: np.ndarray = self._grade_matrix
        matrix.flags.writeable = False
        return matrix

    @property
    def ideal_grades(self) -> np.ndarray:
        """Per-query grades sorted descending - the ideal DCG ordering."""
        ideal: np.ndarray = self._ideal_grades
        return ideal

    def query_row(self, profile_id: str) -> int:
        """Row index of `profile_id` in `grade_matrix`."""
        try:
            return self._query_index[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown profile_id '{profile_id}' in dataset '{self.name}'") from exc

    def opportunity_col(self, opportunity_id: str) -> int:
        """Column index of `opportunity_id` in `grade_matrix`."""
        try:
            return self._opp_index[opportunity_id]
        except KeyError as exc:
            raise KeyError(f"Unknown opportunity_id '{opportunity_id}' in dataset '{self.name}'") from exc

    def relevance_for(self, profile_id: str) -> Dict[str, int]:
        """
        Grade map scoped to a single query, keyed by opportunity id.

        Kept for readability and for callers that genuinely want a dict. Prefer
        `grade_matrix` / `query_row` in loops - this builds a fresh dict per call.
        """
        row = self.grade_matrix[self.query_row(profile_id)]
        return {opp_id: int(row[col]) for col, opp_id in enumerate(self._opportunity_order)}
