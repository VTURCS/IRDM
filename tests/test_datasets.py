"""
Dataset and schema validation tests.

`Dataset` is the guard rail that stops a broken label set from producing
confident-but-meaningless metrics. Because Recall@k divides by the number of
relevant listings in the ground truth, a label map that covers only the
retrieved subset silently inflates recall - so completeness is enforced rather
than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from data.schema import Dataset, Opportunity, Query

from conftest import REPO_ROOT, make_random_dataset


def _tiny_dataset() -> Dataset:
    opportunities = [
        Opportunity(id="A", title="Listing A", description="d", skills=["x"]),
        Opportunity(id="B", title="Listing B", description="d", skills=["y"]),
    ]
    queries = [Query(profile_id="Q1", text="query one")]
    return Dataset(
        name="tiny",
        description="two listings, one query",
        opportunities=opportunities,
        queries=queries,
        relevance_map={("Q1", "A"): 2, ("Q1", "B"): 0},
    )


def test_valid_dataset_constructs() -> None:
    dataset = _tiny_dataset()
    assert dataset.opportunity_ids == ["A", "B"]
    assert dataset.query_ids == ["Q1"]


def test_incomplete_labels_are_rejected() -> None:
    """
    A missing (query, listing) label must raise, not be silently treated as 0.

    Treating unlabelled pairs as irrelevant would quietly distort recall.
    """
    with pytest.raises(ValueError, match="unlabelled"):
        Dataset(
            name="bad",
            description="missing label",
            opportunities=[Opportunity(id="A", title="t", description="d")],
            queries=[Query(profile_id="Q1", text="q")],
            relevance_map={},  # nothing labelled
        )


def test_duplicate_opportunity_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate opportunity ids"):
        Dataset(
            name="dupes",
            description="dup ids",
            opportunities=[
                Opportunity(id="A", title="t1", description="d"),
                Opportunity(id="A", title="t2", description="d"),
            ],
            queries=[Query(profile_id="Q1", text="q")],
            relevance_map={("Q1", "A"): 1},
        )


def test_duplicate_query_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate query ids"):
        Dataset(
            name="dupes",
            description="dup queries",
            opportunities=[Opportunity(id="A", title="t", description="d")],
            queries=[Query(profile_id="Q1", text="a"), Query(profile_id="Q1", text="b")],
            relevance_map={("Q1", "A"): 1},
        )


def test_out_of_range_grades_are_rejected() -> None:
    """Grades are a 0/1/2 rubric; anything else is a labelling error."""
    with pytest.raises(ValueError, match="out-of-range"):
        Dataset(
            name="badgrades",
            description="grade 5",
            opportunities=[Opportunity(id="A", title="t", description="d")],
            queries=[Query(profile_id="Q1", text="q")],
            relevance_map={("Q1", "A"): 5},
        )


def test_grade_matrix_property_is_read_only() -> None:
    """
    The cached grade matrix is shared state; mutation would corrupt every later
    metric computation in the process.
    """
    dataset = _tiny_dataset()
    matrix = dataset.grade_matrix
    assert not matrix.flags.writeable
    with pytest.raises(ValueError):
        matrix[0, 0] = 9


def test_grade_matrix_agrees_with_relevance_map() -> None:
    """The vectorised path must agree with the dict-based path."""
    dataset = make_random_dataset(n_queries=5, n_candidates=7, seed=2)
    matrix = dataset.grade_matrix

    for row, profile_id in enumerate(dataset.query_ids):
        relevance = dataset.relevance_for(profile_id)
        for col, opportunity_id in enumerate(dataset.opportunity_ids):
            assert matrix[row, col] == relevance[opportunity_id]


def test_ideal_grades_are_sorted_descending() -> None:
    """Ideal ordering is precomputed; it must be the descending grade sort."""
    dataset = make_random_dataset(n_queries=6, n_candidates=8, seed=4)
    ideal = dataset.ideal_grades
    assert (np.diff(ideal, axis=1) <= 0).all()


def test_query_row_lookup_raises_for_unknown_profile() -> None:
    with pytest.raises(KeyError, match="Unknown profile_id"):
        _tiny_dataset().query_row("nope")


def test_opportunity_col_lookup_raises_for_unknown_listing() -> None:
    with pytest.raises(KeyError, match="Unknown opportunity_id"):
        _tiny_dataset().opportunity_col("nope")


# ---------------------------------------------------------------------------
# Real datasets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["prototype", "synthetic_v2", "main_demo", "heldout"])
def test_registered_datasets_load_fully_labelled(name: str) -> None:
    """Every shipped dataset must satisfy the label-completeness contract."""
    import research_matching_pipeline as rmp

    dataset = rmp.load_dataset(name)
    n_pairs = len(dataset.queries) * len(dataset.opportunities)
    assert len(dataset.relevance_map) == n_pairs
    assert dataset.grade_matrix.shape == (len(dataset.queries), len(dataset.opportunities))


def test_unknown_dataset_raises() -> None:
    import research_matching_pipeline as rmp

    with pytest.raises(KeyError, match="Unknown dataset"):
        rmp.load_dataset("does_not_exist")


def test_student_profiles_csv_matches_declared_schema() -> None:
    """
    The v2 CSV is authored by hand, so gate it on the schema the loader expects.
    A stray comma or renamed column would otherwise surface as a confusing
    runtime error deep in the loader.
    """
    expected = [
        "student_id",
        "target_opp_id",
        "match_type",
        "full_name",
        "major",
        "academic_year",
        "skills_and_experience",
        "project_summary",
    ]
    frame = pd.read_csv(REPO_ROOT / "data" / "student_profiles.csv")
    assert list(frame.columns) == expected
    assert frame.isnull().sum().sum() == 0
    assert frame["student_id"].is_unique


def test_synthetic_v2_stratum_filter_reduces_queries_but_not_corpus() -> None:
    """
    Filtering by match_type must subset QUERIES only. Labels stay scoped to the
    full corpus, otherwise recall denominators would be computed against a
    truncated candidate set.
    """
    import research_matching_pipeline as rmp

    full = rmp.load_dataset("synthetic_v2")
    filtered = rmp.load_dataset("synthetic_v2", match_types=("semantic_analogy",))

    assert len(filtered.opportunities) == len(full.opportunities)
    assert 0 < len(filtered.queries) < len(full.queries)
    assert filtered.grade_matrix.shape[1] == full.grade_matrix.shape[1]
