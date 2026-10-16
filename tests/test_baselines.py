"""
Baseline reproduction tests.

These pin the numbers reported in docs/RESULTS.md. They exist for three reasons:

1. Reproducibility (OPEN_ISSUES T8). Anyone can run `pytest -m baseline` and get
   the same figures, on any machine, without reading prose.
2. Refactor safety. Any change to the tokenizer, normalisation or ranking that
   moves a published number fails CI instead of silently invalidating a result.
3. Honesty about the encoder. The offline hashed-BoW proxy and real MiniLM
   produce DIFFERENT numbers, so they are pinned separately and the backend
   string is asserted. A lexical proxy must never be filed under a MiniLM result.

Requires no model download: the deterministic proxy is used throughout. The real
MiniLM baselines are marked `slow` and skip cleanly when the model is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

import research_matching_pipeline as rmp

# ---------------------------------------------------------------------------
# Pinned values. Update ONLY alongside a documented, intentional result change.
# ---------------------------------------------------------------------------

PROTOTYPE_OFFLINE = {
    "Keyword (BM25)": (0.7046, 0.55, 0.7417),
    # The proxy is lexical, so it does NOT tie with BM25 - it lands below it.
    "Semantic (HashedBoW proxy)": (0.6818, 0.60, 0.8042),
    "Hybrid (0.6 KW / 0.4 SEM)": (0.7210, 0.60, 0.8042),
}

PROTOTYPE_MINILM = {
    "Keyword (BM25)": (0.7046, 0.55, 0.7417),
    "Semantic (all-MiniLM-L6-v2)": (0.8846, 0.65, 0.8875),
    "Hybrid (0.6 KW / 0.4 SEM)": (0.8364, 0.65, 0.8875),
}

SYNTHETIC_V2_MINILM = {
    "Keyword (BM25)": 0.9667,
    "Semantic (all-MiniLM-L6-v2)": 0.9421,
    "Hybrid (0.6 KW / 0.4 SEM)": 0.9754,
}


def _summarise(dataset_name: str, encoder, backend: str, top_k: int = 5):
    """Run the full pipeline and return (summary DataFrame, method_names)."""
    dataset = rmp.load_dataset(dataset_name)
    cfg = rmp.Config(top_k=top_k, dataset=dataset_name)
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg)
    _, summary = rmp.evaluate_metrics(table, dataset, k=top_k)
    return summary, table.method_names


# ---------------------------------------------------------------------------
# Offline proxy baselines (always run, no download)
# ---------------------------------------------------------------------------


def test_prototype_offline_baseline_is_stable() -> None:
    """
    Pin the offline-proxy run separately from the real-model run.

    The hashed-BoW proxy is a bag of words, so it is not a semantic model and
    produces different numbers from MiniLM (here it scores BELOW BM25). Pinning
    it separately is what stops a proxy result being filed as a semantic one.
    """
    summary, _ = _summarise("prototype", rmp.HashedBoWEncoder(), "offline-forced::hashed-bow")
    by_method = summary.set_index("method")

    for method, (ndcg, precision, recall) in PROTOTYPE_OFFLINE.items():
        assert method in by_method.index, f"missing method label: {method}"
        row = by_method.loc[method]
        assert row["mean_ndcg@k"] == pytest.approx(ndcg, abs=1e-4), method
        assert row["mean_precision@k"] == pytest.approx(precision, abs=1e-4), method
        assert row["mean_recall@k"] == pytest.approx(recall, abs=1e-4), method


def test_offline_run_labels_backend_as_proxy() -> None:
    """The semantic condition must not be labelled MiniLM in offline mode."""
    _, method_names = _summarise("prototype", rmp.HashedBoWEncoder(), "offline-forced::hashed-bow")
    assert "HashedBoW" in method_names[1]
    assert "MiniLM" not in method_names[1]


def test_prototype_bm25_row_is_backend_independent() -> None:
    """BM25 depends only on text, so its row must be identical either way."""
    offline, _ = _summarise("prototype", rmp.HashedBoWEncoder(), "offline-forced::hashed-bow")
    baseline = offline.set_index("method").loc["Keyword (BM25)"]
    assert baseline["mean_ndcg@k"] == pytest.approx(PROTOTYPE_OFFLINE["Keyword (BM25)"][0], abs=1e-4)


# ---------------------------------------------------------------------------
# Real MiniLM baselines (skipped when the model is unavailable)
# ---------------------------------------------------------------------------


def _minilm_or_skip():
    try:
        encoder, backend = rmp.load_encoder("all-MiniLM-L6-v2", allow_fallback=False)
    except Exception as exc:  # import error, offline, no cache
        pytest.skip(f"MiniLM unavailable ({type(exc).__name__}); baseline not checked")
    return encoder, backend


@pytest.mark.slow
def test_prototype_minilm_baseline_is_stable() -> None:
    """The headline prototype result: semantic leads, hybrid sits between."""
    encoder, backend = _minilm_or_skip()
    summary, method_names = _summarise("prototype", encoder, backend)

    assert "MiniLM" in method_names[1], "expected the real model to be used"
    by_method = summary.set_index("method")

    for method, (ndcg, precision, recall) in PROTOTYPE_MINILM.items():
        row = by_method.loc[method]
        assert row["mean_ndcg@k"] == pytest.approx(ndcg, abs=1e-4), method
        assert row["mean_precision@k"] == pytest.approx(precision, abs=1e-4), method
        assert row["mean_recall@k"] == pytest.approx(recall, abs=1e-4), method

    # The qualitative claim in RESULTS.md, asserted explicitly.
    assert by_method.loc["Semantic (all-MiniLM-L6-v2)", "mean_ndcg@k"] > by_method.loc[
        "Hybrid (0.6 KW / 0.4 SEM)", "mean_ndcg@k"
    ] > by_method.loc["Keyword (BM25)", "mean_ndcg@k"]


@pytest.mark.slow
def test_synthetic_v2_minilm_baseline_is_stable() -> None:
    """Secondary result. Documented as a lower bound: labels are derived."""
    encoder, backend = _minilm_or_skip()
    summary, _ = _summarise("synthetic_v2", encoder, backend)
    by_method = summary.set_index("method")

    for method, ndcg in SYNTHETIC_V2_MINILM.items():
        assert by_method.loc[method, "mean_ndcg@k"] == pytest.approx(ndcg, abs=1e-4), method


@pytest.mark.slow
def test_scorable_aggregation_excludes_unscorable_profiles() -> None:
    """
    On synthetic_v2 the distractor strata have no relevant listing and must be
    excluded from means: they score 0 for every method by construction, so
    including them would add a constant zero rather than information.
    """
    encoder, backend = _minilm_or_skip()
    dataset = rmp.load_dataset("synthetic_v2")
    cfg = rmp.Config(top_k=5, dataset="synthetic_v2")
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg)
    per_profile, summary = rmp.evaluate_metrics(table, dataset, k=5)

    n_unscorable = int((~per_profile["scorable"]).sum())
    assert n_unscorable > 0, "fixture should contain unscorable distractor profiles"

    # n_profiles counts only scorable rows, and is well below the full row count.
    assert summary["n_profiles"].max() < len(per_profile)
    scorable = per_profile[per_profile["scorable"]]
    assert summary["n_profiles"].iloc[0] == len(scorable) // 3


@pytest.mark.slow
def test_distractor_negative_control_is_computable() -> None:
    """
    Unscorable distractor profiles are evaluated via `target_rank` instead of
    nDCG. That column must be populated, otherwise the negative-control section
    of the report silently degrades to "no data".
    """
    encoder, backend = _minilm_or_skip()
    dataset = rmp.load_dataset("synthetic_v2")
    cfg = rmp.Config(top_k=5, dataset="synthetic_v2")
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg)
    per_profile, _ = rmp.evaluate_metrics(table, dataset, k=5)

    distractors = per_profile[~per_profile["scorable"]]
    assert len(distractors) > 0
    assert distractors["target_rank"].notna().any(), "target_rank column is empty"


# ---------------------------------------------------------------------------
# Weight sweep: the experiment the write-up says comes next (OPEN_ISSUES M4)
# ---------------------------------------------------------------------------


def test_bm25_weight_sweep_spans_pure_conditions() -> None:
    """
    A bm25_weight of 1.0 must reproduce the keyword condition and 0.0 the
    semantic condition, with intermediate weights interpolating. This is the
    mechanism M4 proposes to tune, so its endpoints are asserted here.
    """
    dataset = rmp.load_dataset("prototype")
    encoder = rmp.HashedBoWEncoder()

    def hybrid_ndcg(weight: float) -> float:
        cfg = rmp.Config(top_k=5, bm25_weight=weight, semantic_weight=1.0 - weight, dataset="prototype")
        table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, "t::h"), cfg)
        _, summary = rmp.evaluate_metrics(table, dataset, k=5)
        labels = table.method_names
        return float(summary.set_index("method").loc[labels[2], "mean_ndcg@k"])

    keyword_only = hybrid_ndcg(1.0)
    semantic_only = hybrid_ndcg(0.0)
    assert np.isfinite([keyword_only, semantic_only]).all()

    cfg = rmp.Config(top_k=5, dataset="prototype")
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, "t::h"), cfg)
    _, summary = rmp.evaluate_metrics(table, dataset, k=5)
    by_method = summary.set_index("method")

    assert keyword_only == pytest.approx(by_method.loc[table.method_names[0], "mean_ndcg@k"], abs=1e-4)
    assert semantic_only == pytest.approx(by_method.loc[table.method_names[1], "mean_ndcg@k"], abs=1e-4)


def test_sweep_weight_actually_moves_results() -> None:
    """
    The weight must have an observable effect on the metric, not just on the label.

    Guards against a wiring bug where the hybrid silently ignores `bm25_weight`
    (which would still print the right label but compute the same ranking).
    """
    dataset = rmp.load_dataset("prototype")
    encoder = rmp.HashedBoWEncoder()

    def hybrid_ndcg(weight: float) -> float:
        cfg = rmp.Config(top_k=5, bm25_weight=weight, semantic_weight=1.0 - weight, dataset="prototype")
        table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, "t::h"), cfg)
        _, summary = rmp.evaluate_metrics(table, dataset, k=5)
        return float(summary.set_index("method").loc[table.method_names[2], "mean_ndcg@k"])

    # The two extremes reproduce different pure conditions, so they must differ.
    assert hybrid_ndcg(0.0) != pytest.approx(hybrid_ndcg(1.0), abs=1e-6)


@pytest.mark.slow
def test_minilm_sweep_default_is_not_optimal() -> None:
    """
    Pin the M4 finding: on the prototype set the default 0.6 keyword weight is
    mistuned relative to a milder 0.4.

    This is a *documented* result (docs/RESULTS.md), so it is asserted rather
    than left to a prose claim. Note the caveat recorded there: with 4 profiles
    the ~0.07 gap is small in rank terms, so this pins the direction, and the
    stable region (0.2-0.4) is the trustworthy part of the curve.
    """
    encoder, backend = _minilm_or_skip()
    dataset = rmp.load_dataset("prototype")

    def hybrid_ndcg(weight: float) -> float:
        cfg = rmp.Config(top_k=5, bm25_weight=weight, semantic_weight=1.0 - weight, dataset="prototype")
        table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg)
        _, summary = rmp.evaluate_metrics(table, dataset, k=5)
        return float(summary.set_index("method").loc[table.method_names[2], "mean_ndcg@k"])

    at_default = hybrid_ndcg(0.6)
    at_mild = hybrid_ndcg(0.4)

    assert at_mild > at_default, (
        f"expected the mild keyword weight to beat the 0.6 default "
        f"(0.4 -> {at_mild:.4f}, 0.6 -> {at_default:.4f})"
    )
    assert at_default == pytest.approx(0.8364, abs=1e-4)
    assert at_mild == pytest.approx(0.9030, abs=1e-4)
