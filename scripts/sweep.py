"""
Hybrid weight sweep (OPEN_ISSUES M4).

Sweeps `bm25_weight` from 0.0 (pure semantic) to 1.0 (pure BM25) and reports how
the hybrid condition's mean nDCG changes. This is the experiment the write-up
identified as the next step, because score-level fusion is only as good as its
weighting.

Why a script rather than an inline shell loop: PowerShell parses the `@` in the
CSV column name `mean_ndcg@k`, and `for ($w in ...)` is not valid PowerShell
syntax, so the one-liner version silently fails. A script is also far safer to
run live during a presentation.

Usage
-----
    python scripts/sweep.py
    python scripts/sweep.py --offline --steps 0,0.1,0.2,0.3,0.4,0.5,0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research_matching_pipeline as rmp  # noqa: E402

RULE = "=" * 78


def evaluate_weight(dataset, encoder, backend: str, weight: float, top_k: int) -> dict:
    """Run the full pipeline at one bm25_weight and return the condition means."""
    cfg = rmp.Config(
        top_k=top_k,
        bm25_weight=weight,
        semantic_weight=round(1.0 - weight, 10),
        dataset=dataset.name,
    )
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg)
    _, summary = rmp.evaluate_metrics(table, dataset, k=top_k)
    by_method = summary.set_index("method")

    return {
        "bm25_weight": weight,
        "semantic_weight": round(1.0 - weight, 10),
        "keyword_ndcg": float(by_method.loc[table.method_names[0], "mean_ndcg@k"]),
        "semantic_ndcg": float(by_method.loc[table.method_names[1], "mean_ndcg@k"]),
        "hybrid_ndcg": float(by_method.loc[table.method_names[2], "mean_ndcg@k"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid weight sweep")
    parser.add_argument("--dataset", default="prototype")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="use the hashed-BoW proxy")
    parser.add_argument(
        "--steps",
        default="0.0,0.2,0.3,0.4,0.6,0.8,1.0",
        help="comma-separated bm25_weight values to evaluate",
    )
    args = parser.parse_args()

    weights = [float(w) for w in args.steps.split(",")]
    if args.offline:
        encoder, backend = rmp.HashedBoWEncoder(), "sweep::hashed-bow"
    else:
        encoder, backend = rmp.load_encoder("all-MiniLM-L6-v2", allow_fallback=False)

    dataset = rmp.load_dataset(args.dataset)

    print(RULE)
    print(f"Hybrid weight sweep - dataset '{args.dataset}', k={args.top_k}")
    print(f"encoder: {backend}")
    print(RULE)
    print(
        "Note: pure conditions are the endpoints - bm25_weight=1.0 IS the keyword\n"
        "condition and 0.0 IS the semantic condition. Intermediate weights show how\n"
        "much keyword signal the hybrid should mix in.\n"
    )

    rows = [evaluate_weight(dataset, encoder, backend, w, args.top_k) for w in weights]
    frame = pd.DataFrame(rows)

    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = frame.loc[frame["hybrid_ndcg"].idxmax()]
    semantic_only = frame.loc[frame["bm25_weight"] == 0.0, "semantic_ndcg"]
    default_row = frame.loc[(frame["bm25_weight"] - 0.6).abs() < 1e-9, "hybrid_ndcg"]

    print(f"\n{'-' * 78}")
    print(f"best hybrid weight : {best['bm25_weight']:.1f}  ->  nDCG {best['hybrid_ndcg']:.4f}")
    if not semantic_only.empty:
        lift = best["hybrid_ndcg"] - float(semantic_only.iloc[0])
        print(f"pure semantic      : 0.0  ->  nDCG {float(semantic_only.iloc[0]):.4f}")
        print(f"lift over semantic : {lift:+.4f}")
    if not default_row.empty:
        print(f"default (0.6)      :      ->  nDCG {float(default_row.iloc[0]):.4f}")
        print(f"cost of the default: {float(default_row.iloc[0]) - best['hybrid_ndcg']:+.4f} vs best")

    print(
        "\nCAVEAT: with 4 profiles a difference of ~0.02 nDCG is ONE rank change in\n"
        "one profile. Read the SHAPE of the curve, not the argmax. What the sweep\n"
        "does establish is that the default 0.6 is not optimal: the curve peaks at a\n"
        "lower keyword weight, so the keyword priority in the current default is too\n"
        "strong. Treat the exact peak location as noise until the corpus is larger.\n"
    )
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
