"""
Evaluate keyword, semantic and hybrid retrieval on one dataset.

    python evaluate.py                      # the default dataset (prototype)
    python evaluate.py heldout              # any folder name in data/
    python evaluate.py path/to/my_dataset   # or any folder in the same CSV format
    python evaluate.py prototype --sweep    # also sweep the hybrid's keyword weight

Prints mean scores per method and writes three CSVs to outputs/<dataset>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import irdm

SWEEP_WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)


def _weight(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be within [0, 1], got {text}")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate keyword, semantic and hybrid retrieval on one dataset.")
    parser.add_argument("dataset", nargs="?", default=irdm.DEFAULT_DATASET,
                        help=f"folder name in data/ ({', '.join(irdm.list_datasets())}) or a folder path")
    parser.add_argument("--top-k", type=_positive, default=5, help="ranked-list length and metric cut-off (default 5)")
    parser.add_argument("--bm25-weight", type=_weight, default=0.6, help="hybrid keyword weight, 0 to 1 (default 0.6)")
    parser.add_argument("--sweep", action="store_true", help="also report hybrid nDCG across keyword weights")
    parser.add_argument("--output-dir", default="outputs", help="where to write the CSVs (default outputs)")
    args = parser.parse_args(argv)

    try:
        dataset = irdm.load_dataset(args.dataset)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None
    k = args.top_k
    results, per_profile, summary = irdm.evaluate(dataset, k, args.bm25_weight)
    n_profiles, n_listings = dataset.grades.shape

    print(f"Dataset {dataset.name}: {n_listings} listings, {n_profiles} profiles, k = {k}, "
          f"model {irdm.MODEL_NAME}@{irdm.MODEL_REVISION[:7]}")
    _section(f"Mean scores per method (k = {k})")
    print(summary.to_string(index=False, float_format="%.4f"))
    unscorable = per_profile.loc[~per_profile["scorable"], "profile_id"].unique()
    if len(unscorable):
        print(f"\n{len(unscorable)} profile(s) have no relevant listing and are left out of the means: "
              f"{', '.join(unscorable)}")

    by_type = irdm.by_match_type(dataset, per_profile)
    if by_type is not None:
        _section(f"Mean nDCG@{k} by profile type")
        print(by_type.to_string(float_format="%.4f"))

    # Distractor profiles have no relevant listing, so instead ask how highly each
    # method ranked the listing they were written against. A high rank = fooled.
    if "target_id" in dataset.profiles and len(unscorable):
        distractors = per_profile[~per_profile["scorable"]].groupby("method", sort=False)["target_rank"]
        _section("Negative control: where each method ranked a distractor's target listing")
        for method, ranks in distractors:
            print(f"  {method:<30} in top {k}: {ranks.notna().mean():.0%}, mean rank there {ranks.mean():.2f}")
        print(f"  (random ordering would give a mean rank of {(n_listings + 1) / 2:.1f})")

    if args.sweep:
        _section(f"Hybrid nDCG@{k} by keyword weight")
        for weight in SWEEP_WEIGHTS:
            ndcg = irdm.evaluate(dataset, k, weight)[2]["mean_ndcg@k"].iloc[2]
            note = {0.0: "  (= semantic only)", 1.0: "  (= keyword only)"}.get(weight, "")
            print(f"  {weight:.1f}  {ndcg:.4f}  {'#' * round(ndcg * 40)}{note}")

    out = Path(args.output_dir) / dataset.name
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("match_results", results), ("evaluation_per_profile", per_profile),
                        ("evaluation_summary", summary)):
        table.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    print(f"\nCSVs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
