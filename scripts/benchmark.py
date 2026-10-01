"""
Phase-timing benchmark for the IRDM matching pipeline.

Purpose: produce a BEFORE/AFTER comparison so performance claims are measured
rather than asserted. Run it before and after a refactor and diff the tables.

    python scripts/benchmark.py
    python scripts/benchmark.py --scale 2000        # larger synthetic corpus

Design notes
------------
* Each phase is timed independently so a total-speedup number cannot hide a
  phase that got *slower*.
* `--offline` uses the deterministic HashedBoW encoder. That isolates Python- and
  NumPy-level overhead from transformer inference, which is what we can actually
  optimise in this codebase.
* The scaled case is deliberately synthetic: the real datasets are tiny (4-20
  queries), and at that size everything is dominated by fixed overhead. The
  asymptotic behaviour only shows up at scale.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research_matching_pipeline as rmp  # noqa: E402
from data import load_dataset  # noqa: E402
from data.schema import Dataset, Opportunity, Query  # noqa: E402


def timeit(fn: Callable[[], object], repeat: int = 3) -> Tuple[float, object]:
    """Return (best-of-`repeat` seconds, last result). Best-of reduces noise."""
    best = float("inf")
    result: object = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def build_scaled_dataset(n_queries: int, n_listings: int) -> Dataset:
    """
    Build a large synthetic Dataset to expose asymptotic behaviour.

    Listings and queries are generated from a fixed vocabulary with overlapping
    skill terms, so BM25 and the encoder both do non-trivial work. Labels give
    each query 3 relevant listings (grade 2, 1, 1) so Recall@k is not degenerate.
    """
    rng = np.random.default_rng(0)
    vocab = [f"skill{i}" for i in range(120)] + ["python", "docker", "pytorch", "linux"]

    opportunities: List[Opportunity] = []
    for i in range(n_listings):
        n_terms = int(rng.integers(4, 10))
        skills = [str(t) for t in rng.choice(vocab, size=n_terms, replace=False)]
        opportunities.append(
            Opportunity(
                id=f"OPP-{i:05d}",
                title=f"Research Project {i} on {skills[0]}",
                description=" ".join(str(t) for t in rng.choice(vocab, size=40, replace=True)),
                skills=skills,
                eligibility="Any, Freshman+",
            )
        )

    queries: List[Query] = []
    for j in range(n_queries):
        queries.append(
            Query(
                profile_id=f"STU-{j:05d}",
                text=" ".join(str(t) for t in rng.choice(vocab, size=25, replace=True)),
                major="Computer Science",
                year="Junior",
            )
        )

    # 3 relevant listings per query, deterministic.
    relevance_map: Dict[Tuple[str, str], int] = {}
    for j, query in enumerate(queries):
        relevant = {
            opportunities[(j * 7) % n_listings].id: 2,
            opportunities[(j * 7 + 1) % n_listings].id: 1,
            opportunities[(j * 7 + 2) % n_listings].id: 1,
        }
        for opportunity in opportunities:
            relevance_map[(query.profile_id, opportunity.id)] = relevant.get(opportunity.id, 0)

    return Dataset(
        name=f"scaled_{n_queries}x{n_listings}",
        description=f"synthetic scale test: {n_queries} queries x {n_listings} listings",
        opportunities=opportunities,
        queries=queries,
        relevance_map=relevance_map,
        suggested_top_k=10,
    )


def bench_dataset(dataset: Dataset, cfg: rmp.Config, encoder, backend: str) -> Dict[str, float]:
    """Time each pipeline phase for one dataset. Returns phase -> seconds."""
    timings: Dict[str, float] = {}

    # Phase 1: document text assembly
    timings["1_build_documents"], documents = timeit(
        lambda: rmp.build_documents(dataset.opportunities, cfg)
    )
    assert isinstance(documents, list)

    # Phase 2: lexical index construction (tokenize + BM25 fit)
    timings["2_build_keyword_index"], _ = timeit(lambda: rmp.KeywordSearcher(documents, cfg))

    # Phase 3: dense index construction (document embeddings)
    timings["3_encode_documents"], _ = timeit(lambda: rmp.SemanticSearcher(documents, encoder, cfg))

    keyword = rmp.KeywordSearcher(documents, cfg)
    semantic = rmp.SemanticSearcher(documents, encoder, cfg)
    query_texts = [q.text for q in dataset.queries]

    # Phase 4: query encoding, batched - the previously dominant ML bottleneck
    timings["4_encode_queries_batched"], _ = timeit(
        lambda: semantic.score_batch(query_texts)
    )
    # Reference: the old per-query path, kept for the before/after comparison.
    timings["4b_encode_queries_percall"], _ = timeit(
        lambda: [semantic.score(text) for text in query_texts]
    )

    # Phase 5: full pipeline via the public API (index -> rank), which is what
    # actually replaced the old `run_pipeline` row loop.
    with_api, _ = timeit(lambda: rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg))
    timings["5_build_index_and_rank"] = with_api

    index = rmp.build_index(cfg, dataset, encoder, backend)
    table = rmp.rank_candidates(index, cfg)

    # Phase 6: metric evaluation
    timings["6_evaluate_metrics"], _ = timeit(lambda: rmp.evaluate_metrics(table, dataset, k=cfg.top_k))

    # Phase 7: tidy frame materialisation (export path)
    timings["7_to_dataframe"], _ = timeit(lambda: table.to_frame(dataset.opportunities))

    timings["TOTAL_index_rank_metrics"] = (
        timings["5_build_index_and_rank"] + timings["6_evaluate_metrics"]
    )
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-timing benchmark")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--model", action="store_true", help="use the real MiniLM model")
    parser.add_argument("--scale", type=int, default=0, help="also time an N-query synthetic corpus")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.model:
        encoder, backend = rmp.load_encoder("all-MiniLM-L6-v2", allow_fallback=True)
    else:
        encoder, backend = rmp.HashedBoWEncoder(), "bench::hashed-bow"

    print(f"encoder: {backend}\n")

    cases = [("prototype", load_dataset("prototype")), ("synthetic_v2", load_dataset("synthetic_v2"))]
    if args.scale:
        cases.append((f"scaled", build_scaled_dataset(args.scale, max(50, args.scale // 10))))

    all_timings: Dict[str, Dict[str, float]] = {}
    for label, dataset in cases:
        cfg = rmp.Config(
            top_k=min(dataset.suggested_top_k, len(dataset.opportunities)),
            dataset=dataset.name if dataset.name in __import__("data").available_datasets() else rmp.DEFAULT_DATASET,
        )
        print(f"--- {label}: {len(dataset.queries)} queries x {len(dataset.opportunities)} listings ---")
        try:
            timings = bench_dataset(dataset, cfg, encoder, backend)
        except Exception as exc:  # a phase may not exist yet
            print(f"    FAILED: {type(exc).__name__}: {exc}\n")
            continue
        all_timings[label] = timings
        for phase, seconds in timings.items():
            print(f"    {phase:<28} {seconds * 1000:9.2f} ms")
        print()

    # Machine-readable block for diffing before/after.
    print("=== CSV ===")
    phases = sorted({p for t in all_timings.values() for p in t})
    print("case,phase,ms")
    for label, timings in all_timings.items():
        for phase in phases:
            if phase in timings:
                print(f"{label},{phase},{timings[phase] * 1000:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
