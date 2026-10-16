"""
Metric correctness tests.

These are the tests that matter most for a ranking evaluation: if the metric
implementation is wrong, every published number is wrong, and a broken metric
often fails *silently* by producing plausible-looking values.

The core test compares the vectorised production metrics against a
straightforward brute-force reference computed with plain Python loops. The
reference is deliberately naive - it is much easier to see that a naive loop is
correct than it is to audit vectorised index arithmetic.

This harness has already earned its keep once: the first vectorised version of
`evaluate_metrics` gathered grades from the wrong axis and made all three
retrieval methods score identically. See docs/verification_log.md.
"""

from __future__ import annotations

import numpy as np
import pytest

import research_matching_pipeline as rmp
from conftest import make_random_dataset


# ---------------------------------------------------------------------------
# Brute-force reference implementations (plain Python, no numpy vectorisation)
# ---------------------------------------------------------------------------


def ref_dcg(grades: list[float], k: int) -> float:
    """DCG@k with graded gain 2^rel - 1 and log2 position discount."""
    return sum((2.0**g - 1.0) / np.log2(i + 1.0) for i, g in enumerate(list(grades)[:k], start=1))


def ref_ndcg(ranked_ids, relevance: dict[str, int], k: int) -> float:
    dcg = ref_dcg([relevance.get(i, 0) for i in ranked_ids][:k], k)
    idcg = ref_dcg(sorted(relevance.values(), reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def ref_precision(ranked_ids, relevance: dict[str, int], k: int, threshold: int = 1) -> float:
    ranked = list(ranked_ids)[:k]
    if not ranked:
        return 0.0
    return sum(1 for i in ranked if relevance.get(i, 0) >= threshold) / float(k)


def ref_recall(ranked_ids, relevance: dict[str, int], k: int, threshold: int = 1) -> float:
    ranked = list(ranked_ids)[:k]
    n_relevant = sum(1 for g in relevance.values() if g >= threshold)
    if n_relevant == 0:
        return 0.0
    return len({i for i in ranked if relevance.get(i, 0) >= threshold}) / float(n_relevant)


# ---------------------------------------------------------------------------
# Vectorised == brute force
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_vectorised_metrics_match_brute_force(seed: int, encoder) -> None:
    """
    Production metrics must equal the naive reference for every row.

    Sweeps a range of query counts, corpus sizes and k values so the comparison
    exercises k < corpus, k == corpus and k > corpus (the last is where the
    Precision@k denominator convention matters).
    """
    rng = np.random.default_rng(seed + 1000)
    n_queries = int(rng.integers(3, 20))
    n_candidates = int(rng.integers(3, 18))
    k = int(rng.integers(1, n_candidates + 4))

    dataset = make_random_dataset(n_queries, n_candidates, seed)
    cfg = rmp.Config(top_k=k, dataset="prototype")

    index = rmp.build_index(cfg, dataset, encoder, "test::deterministic")
    table = rmp.rank_candidates(index, cfg)
    per_profile, _ = rmp.evaluate_metrics(table, dataset, k=k)

    checked = 0
    for _, row in per_profile.iterrows():
        profile_id, method = row["profile_id"], row["method"]
        mask = (table.methods == method) & (table.profile_ids == profile_id)
        ranked_ids = [table.opportunity_ids[c] for c in table.top_indices[mask][0]]
        relevance = dataset.relevance_for(profile_id)

        assert row["ndcg@k"] == pytest.approx(ref_ndcg(ranked_ids, relevance, k), abs=1e-4), (
            f"nDCG mismatch: profile={profile_id} method={method} k={k} "
            f"corpus={n_candidates}"
        )
        assert row["precision@k"] == pytest.approx(ref_precision(ranked_ids, relevance, k), abs=1e-4)
        assert row["recall@k"] == pytest.approx(ref_recall(ranked_ids, relevance, k), abs=1e-4)
        checked += 1

    assert checked > 0, "test compared no rows"


def test_precision_denominator_is_requested_k() -> None:
    """
    Precision@k divides by the REQUESTED k, not by the list length.

    Regression guard: an early vectorised rewrite divided by the effective list
    length, which silently inflated precision whenever k exceeded the corpus
    size. Matches the original scalar implementation and the standard convention
    that a short list should be penalised.
    """
    # 2 candidates, k = 4, both relevant -> 2/4 = 0.5, not 2/2 = 1.0
    grades = np.array([[2, 1]], dtype=float)
    result = rmp.precision_at_k_matrix(grades, k=4)
    assert result[0] == pytest.approx(0.5)

    # Sanity: the same row scored at k = 2 gives full precision.
    assert rmp.precision_at_k_matrix(grades, k=2)[0] == pytest.approx(1.0)


def test_methods_produce_distinct_rankings(encoder) -> None:
    """
    Retrieval methods must not collapse to the same ranking.

    Regression guard for the axis bug: because BM25 and cosine similarity are
    computed from different signals, their top-k orders should differ on data
    where both have signal. If they are identical, grades are likely being
    gathered from the wrong source.
    """
    dataset = make_random_dataset(n_queries=8, n_candidates=15, seed=42)
    cfg = rmp.Config(top_k=5, dataset="prototype")
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, "t::d"), cfg)

    bm25_rows = table.methods == table.method_names[0]
    semantic_rows = table.methods == table.method_names[1]
    assert not np.array_equal(table.top_indices[bm25_rows], table.top_indices[semantic_rows])


def test_graded_gain_rewards_strong_matches() -> None:
    """
    Gain must be convex in the grade, so a rank-1 strong match beats a rank-1
    partial match by more than the linear gap.

    This is the reason the rubric uses 2^rel - 1 rather than rel: surfacing the
    single best-fitting listing should dominate returning lukewarm options.
    """
    assert rmp.gain(np.array([2.0]))[0] == pytest.approx(3.0)
    assert rmp.gain(np.array([1.0]))[0] == pytest.approx(1.0)
    assert rmp.gain(np.array([0.0]))[0] == pytest.approx(0.0)


def test_ndcg_is_one_for_ideal_ranking() -> None:
    """nDCG must saturate at 1.0 when the retrieved order is the ideal order."""
    ideal = np.array([[2, 1, 0, 0]], dtype=float)
    # Already sorted descending, so it IS the ideal ordering.
    assert rmp.ndcg_at_k_matrix(ideal, ideal, k=4)[0] == pytest.approx(1.0)


def test_ndcg_is_zero_for_worst_ranking() -> None:
    """nDCG must be 0.0 when no retrieved item is relevant."""
    retrieved = np.array([[0, 0, 0, 0]], dtype=float)
    ideal = np.array([[2, 1, 0, 0]], dtype=float)
    assert rmp.ndcg_at_k_matrix(retrieved, ideal, k=4)[0] == pytest.approx(0.0)


def test_unscorable_query_scores_zero_without_dividing_by_zero() -> None:
    """A query with no relevant listing must yield 0.0, not NaN/inf."""
    retrieved = np.array([[0, 0]], dtype=float)
    ideal = np.array([[0, 0]], dtype=float)
    result = rmp.ndcg_at_k_matrix(retrieved, ideal, k=2)
    assert result[0] == 0.0
    assert np.isfinite(result[0])


def test_k_larger_than_corpus_is_handled() -> None:
    """Ranking with k > corpus size must not raise or over-read."""
    dataset = make_random_dataset(n_queries=4, n_candidates=4, seed=7)
    cfg = rmp.Config(top_k=50, dataset="prototype")
    encoder = rmp.HashedBoWEncoder()
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, "t::hashed"), cfg)
    # Ranked list is clamped to the number of available candidates.
    assert table.k == 4
    per_profile, _ = rmp.evaluate_metrics(table, dataset, k=50)
    assert np.isfinite(per_profile["ndcg@k"]).all()
