"""
Retrieval behaviour tests.

These pin down the properties the research write-up depends on: that the three
conditions are genuinely different mechanisms, that the hybrid is a weighted
blend on a comparable [0, 1] scale, and that ranking is deterministic (tie
handling) so a run can be reproduced exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import research_matching_pipeline as rmp
from conftest import make_random_dataset


def test_min_max_normalisation_spans_unit_interval() -> None:
    scores = np.array([2.0, 4.0, 8.0])
    out = rmp.min_max_normalize(scores)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_min_max_normalisation_handles_constant_scores() -> None:
    """
    Constant scores must normalise to zeros rather than dividing by zero.

    A query with no lexical overlap produces an all-zero BM25 vector; this case
    must degrade gracefully so the hybrid falls back to the semantic signal.
    """
    out = rmp.min_max_normalize(np.array([3.0, 3.0, 3.0]))
    assert np.allclose(out, 0.0)
    assert np.isfinite(out).all()


def test_min_max_normalisation_handles_nan_and_inf() -> None:
    out = rmp.min_max_normalize(np.array([1.0, np.nan, np.inf, 5.0]))
    assert np.isfinite(out).all()


def test_rowwise_normalisation_is_per_query() -> None:
    """
    Normalisation must be row-wise: each query's scores are scaled against that
    query's own min/max. Scaling per corpus instead would make scores
    incomparable across queries and silently change the hybrid ranking.
    """
    scores = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = rmp.min_max_normalize_2d(scores)
    assert np.allclose(out[0], [0.0, 0.5, 1.0])
    assert np.allclose(out[1], [0.0, 0.5, 1.0])


def test_row_constant_normalisation_yields_zeros() -> None:
    scores = np.array([[5.0, 5.0], [1.0, 2.0]])
    out = rmp.min_max_normalize_2d(scores)
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[1], [0.0, 1.0])


def test_hybrid_matches_manual_weighted_sum() -> None:
    """
    Hybrid = w_bm25 * bm25_norm + w_sem * sem_norm, verified against a manual
    computation with the same weights.
    """
    bm25 = np.array([1.0, 4.0, 2.0])
    semantic = np.array([0.10, 0.30, 0.20])
    cfg = rmp.Config(bm25_weight=0.6, semantic_weight=0.4, dataset="prototype")

    actual = rmp.hybrid_scores_matrix(bm25, semantic, cfg)
    expected = 0.6 * rmp.min_max_normalize(bm25) + 0.4 * rmp.min_max_normalize(semantic)
    assert np.allclose(actual, expected)


def test_hybrid_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        rmp.Config(bm25_weight=0.9, semantic_weight=0.9, dataset="prototype")


def test_keyword_priority_changes_ranking() -> None:
    """
    The weights must actually matter: an all-keyword hybrid should reproduce the
    BM25 ordering and an all-semantic hybrid the semantic ordering. If this
    fails, the weighting is not being applied.
    """
    dataset = make_random_dataset(n_queries=5, n_candidates=12, seed=3)
    encoder = rmp.HashedBoWEncoder()

    def top_for(weight: float) -> np.ndarray:
        cfg = rmp.Config(top_k=4, bm25_weight=weight, semantic_weight=1.0 - weight, dataset="prototype")
        table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, "t::hashed"), cfg)
        hybrid_rows = table.methods == table.method_names[2]
        return table.top_indices[hybrid_rows]

    keyword_only = top_for(1.0)
    semantic_only = top_for(0.0)

    cfg = rmp.Config(top_k=4, dataset="prototype")
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, "t::hashed"), cfg)
    bm25_rows = table.methods == table.method_names[0]
    semantic_rows = table.methods == table.method_names[1]

    # weight=1.0 hybrid must equal pure BM25; weight=0.0 must equal pure semantic.
    assert np.array_equal(keyword_only, table.top_indices[bm25_rows])
    assert np.array_equal(semantic_only, table.top_indices[semantic_rows])


def test_ties_break_on_opportunity_id() -> None:
    """
    Equal scores must be ordered by opportunity id, so a run is reproducible.

    The evaluation harness depends on this: without a deterministic tie rule,
    metric values would drift between runs on corpora with tied scores.
    """
    scores = np.zeros((1, 4))
    ids = np.array(["C", "A", "D", "B"])
    top = rmp.top_k_indices(scores, k=4, tiebreak_keys=ids)
    assert [ids[i] for i in top[0]] == ["A", "B", "C", "D"]


def test_scores_sorted_descending_within_row() -> None:
    dataset = make_random_dataset(n_queries=6, n_candidates=10, seed=11)
    cfg = rmp.Config(top_k=5, dataset="prototype")
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, rmp.HashedBoWEncoder(), "t::h"), cfg)
    # Each row's gathered scores must be non-increasing.
    assert (np.diff(table.top_scores, axis=1) <= 1e-9).all()


def test_ranking_is_deterministic_across_runs() -> None:
    """Two identical runs must produce byte-identical rankings."""
    dataset = make_random_dataset(n_queries=6, n_candidates=10, seed=5)
    cfg = rmp.Config(top_k=5, dataset="prototype")

    first = rmp.rank_candidates(rmp.build_index(cfg, dataset, rmp.HashedBoWEncoder(), "t::h"), cfg)
    second = rmp.rank_candidates(rmp.build_index(cfg, dataset, rmp.HashedBoWEncoder(), "t::h"), cfg)
    assert np.array_equal(first.top_indices, second.top_indices)


def test_semantic_scores_are_cosine_similarity() -> None:
    """
    Cosine similarity must lie within [-1, 1].

    A row outside that range means the embeddings were not L2-normalised, so the
    inner product is not actually cosine similarity.
    """
    dataset = make_random_dataset(n_queries=4, n_candidates=8, seed=9)
    cfg = rmp.Config(top_k=3, dataset="prototype")
    index = rmp.build_index(cfg, dataset, rmp.HashedBoWEncoder(), "t::h")
    sims = index.semantic.score_batch([q.text for q in dataset.queries])
    assert sims.min() >= -1.0 - 1e-6
    assert sims.max() <= 1.0 + 1e-6


def test_document_text_excludes_eligibility_by_default() -> None:
    """
    Baseline constraint: eligibility text must stay out of the searchable body,
    so no condition can benefit from major/year keyword overlap.
    """
    dataset = rmp.load_dataset("prototype")
    cfg = rmp.Config(dataset="prototype")
    documents = rmp.build_documents(dataset.opportunities, cfg)
    eligibility = dataset.opportunities[0].eligibility

    assert eligibility, "fixture should carry eligibility metadata"
    assert eligibility.lower() not in documents[0].lower()

    # ...but it IS included when the ablation switch is explicitly enabled.
    cfg_on = rmp.Config(dataset="prototype", include_eligibility_in_text=True)
    assert eligibility.lower() in rmp.build_documents(dataset.opportunities, cfg_on)[0].lower()


def test_offline_encoder_is_labelled_as_a_proxy() -> None:
    """
    The offline fallback must never be labelled as MiniLM.

    Misreporting a lexical proxy as a semantic model would invalidate any
    comparison against the keyword condition.
    """
    labels = rmp.method_labels(rmp.Config(dataset="prototype"), "offline-forced::hashed-bow")
    assert "HashedBoW" in labels[1]
    assert "MiniLM" not in labels[1]

    real = rmp.method_labels(rmp.Config(dataset="prototype"), "sentence-transformers::all-MiniLM-L6-v2")
    assert "MiniLM" in real[1]


def test_hybrid_label_reflects_configured_weights() -> None:
    """Labels must print the real weights, so an overridden run is not mislabelled."""
    cfg = rmp.Config(bm25_weight=0.7, semantic_weight=0.3, dataset="prototype")
    labels = rmp.method_labels(cfg, "sentence-transformers::all-MiniLM-L6-v2")
    assert "0.7" in labels[2] and "0.3" in labels[2]
