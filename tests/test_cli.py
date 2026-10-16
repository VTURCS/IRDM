"""
CLI and end-to-end smoke tests.

These check the properties a reader actually relies on: that the documented
commands work, that the export schema is stable, and that bad input fails
loudly rather than producing a half-written result.
"""

from __future__ import annotations

import pandas as pd
import pytest

import research_matching_pipeline as rmp

REQUIRED_RESULT_COLUMNS = [
    "profile_id",
    "method",
    "opportunity_id",
    "rank",
    "score",
    "title",
    "description",
]


def test_default_invocation_succeeds(tmp_path) -> None:
    """The command in the README must run and exit 0."""
    exit_code = rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path)])
    assert exit_code == 0


def test_exports_have_expected_schema(tmp_path) -> None:
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path)])
    out = tmp_path / "prototype"

    results = pd.read_csv(out / "match_results.csv")
    per_profile = pd.read_csv(out / "evaluation_per_profile.csv")
    summary = pd.read_csv(out / "evaluation_summary.csv")

    # Column order is part of the contract for downstream consumers.
    assert list(results.columns) == REQUIRED_RESULT_COLUMNS
    assert {"profile_id", "method", "ndcg@k", "precision@k", "recall@k", "scorable"} <= set(
        per_profile.columns
    )
    assert {"method", "mean_ndcg@k", "mean_precision@k", "mean_recall@k", "n_profiles"} <= set(
        summary.columns
    )
    assert len(summary) == 3  # one row per condition


def test_export_row_count_covers_profiles_methods_and_ranks(tmp_path) -> None:
    """4 profiles x 3 conditions x k=5 = 60 rows for the prototype dataset."""
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path)])
    results = pd.read_csv(tmp_path / "prototype" / "match_results.csv")

    dataset = rmp.load_dataset("prototype")
    assert len(results) == len(dataset.queries) * 3 * 5
    assert results["rank"].min() == 1
    assert results["rank"].max() == 5


def test_ranks_are_contiguous_per_profile_and_method(tmp_path) -> None:
    """Each ranked list must be 1..k with no gaps or duplicates."""
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path)])
    results = pd.read_csv(tmp_path / "prototype" / "match_results.csv")

    for (profile_id, method), group in results.groupby(["profile_id", "method"]):
        assert sorted(group["rank"]) == list(range(1, len(group) + 1)), (profile_id, method)


def test_output_directories_are_isolated_per_dataset(tmp_path) -> None:
    """Runs must not overwrite each other's exports."""
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path)])
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path), "--dataset", "synthetic_v2"])

    assert (tmp_path / "prototype" / "evaluation_summary.csv").exists()
    assert (tmp_path / "synthetic_v2" / "evaluation_summary.csv").exists()


def test_stratum_run_gets_its_own_directory(tmp_path) -> None:
    """
    A stratified run must not clobber the full run's CSVs, since both are named
    after the dataset otherwise.
    """
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path), "--dataset", "synthetic_v2"])
    rmp.main(
        [
            "--offline-embeddings",
            "--output-dir", str(tmp_path),
            "--dataset", "synthetic_v2",
            "--match-type", "semantic_analogy",
        ]
    )
    assert (tmp_path / "synthetic_v2" / "evaluation_summary.csv").exists()
    assert (tmp_path / "synthetic_v2__semantic_analogy" / "evaluation_summary.csv").exists()


def test_top_k_override_is_respected(tmp_path) -> None:
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path), "--top-k", "3"])
    results = pd.read_csv(tmp_path / "prototype" / "match_results.csv")
    assert results["rank"].max() == 3


def test_bm25_weight_override_changes_hybrid_label(tmp_path) -> None:
    """An overridden run must be labelled with the weights actually used."""
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path), "--bm25-weight", "0.7"])
    summary = pd.read_csv(tmp_path / "prototype" / "evaluation_summary.csv")
    hybrid = summary[summary["method"].str.contains("Hybrid")]
    assert "0.7" in hybrid["method"].iloc[0]


def test_invalid_dataset_fails_before_loading_encoder(capsys) -> None:
    """A bad --dataset must fail fast, not after an expensive model download."""
    with pytest.raises(SystemExit) as excinfo:
        rmp.main(["--dataset", "not_a_dataset"])
    assert excinfo.value.code == 2  # argparse usage error


def test_invalid_bm25_weight_is_rejected(tmp_path) -> None:
    """
    A weight outside [0, 1] must be rejected at the CLI boundary.

    Regression guard: `--bm25-weight 1.5` derives `semantic_weight = -0.5`, which
    SUMS to 1.0 and so slipped past a sum-only check. A negative weight encodes a
    nonsense rule (negatively weighted semantic signal) while still producing a
    plausible-looking ranking, so the range is validated explicitly.
    """
    with pytest.raises(SystemExit) as excinfo:
        rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path), "--bm25-weight", "1.5"])
    assert excinfo.value.code == 2  # argparse usage error


def test_negative_weight_is_rejected_by_config() -> None:
    """The programmatic entry point must reject it too, not just the CLI."""
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        rmp.Config(bm25_weight=1.5, semantic_weight=-0.5, dataset="prototype")


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        rmp.Config(top_k=0, dataset="prototype")


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        rmp.Config(batch_size=0, dataset="prototype")


def test_scores_are_rounded_to_six_places(tmp_path) -> None:
    """Export rounding is part of the reproducibility contract."""
    rmp.main(["--offline-embeddings", "--output-dir", str(tmp_path)])
    scores = pd.read_csv(tmp_path / "prototype" / "match_results.csv")["score"]
    assert (scores.round(6) == scores).all()
