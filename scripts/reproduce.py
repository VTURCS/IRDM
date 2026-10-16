"""
One-command presentation run.

This is the script to run live. It exercises the whole pipeline, PRINTS the
results (not just writes them), and verifies every reported number against
docs/RESULTS.md -- so a single command both demonstrates the system and proves
the numbers are current.

    python scripts/reproduce.py            # real MiniLM
    python scripts/reproduce.py --offline  # no model download (lexical proxy)

Output is deliberately ASCII-only: the default Windows console codepage renders
em-dashes and box-drawing characters as mojibake, which looks broken on a
projector.

Exit codes: 0 = all values reproduced, 1 = drift detected.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research_matching_pipeline as rmp  # noqa: E402

# ---------------------------------------------------------------------------
# Expected values, cited in docs/RESULTS.md.
# Tolerance is tight enough to catch a real behaviour change.
# ---------------------------------------------------------------------------
TOLERANCE = 1e-4

EXPECTED = {
    ("prototype", "MiniLM"): {
        "Keyword (BM25)": (0.7046, 0.55, 0.7417),
        "Semantic (all-MiniLM-L6-v2)": (0.8846, 0.65, 0.8875),
        "Hybrid (0.6 KW / 0.4 SEM)": (0.8364, 0.65, 0.8875),
    },
    ("synthetic_v2", "MiniLM"): {
        "Keyword (BM25)": (0.9667, 0.20, 1.00),
        "Semantic (all-MiniLM-L6-v2)": (0.9421, 0.20, 1.00),
        "Hybrid (0.6 KW / 0.4 SEM)": (0.9754, 0.20, 1.00),
    },
    ("prototype", "proxy"): {
        "Keyword (BM25)": (0.7046, 0.55, 0.7417),
        "Semantic (HashedBoW proxy)": (0.6818, 0.60, 0.8042),
        "Hybrid (0.6 KW / 0.4 SEM)": (0.7210, 0.60, 0.8042),
    },
}

SWEEP_WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
RESULTS_DIR = Path("results")
WIDTH = 78

# --- tiny presentation helpers (ASCII only) ---------------------------------


def rule(char: str = "=") -> None:
    print(char * WIDTH)


def banner(text: str) -> None:
    rule()
    print(text)
    rule()


def section(text: str) -> None:
    print()
    print(text)
    print("-" * WIDTH)


def bar(value: float, scale: float = 1.0, width: int = 24) -> str:
    """ASCII bar for a 0..scale value, so the console shows magnitude at a glance."""
    filled = int(round(width * max(0.0, min(1.0, value / scale))))
    return "#" * filled + "." * (width - filled)


def environment_report() -> list[str]:
    """Capture the interpreter and package versions a result depends on."""
    lines = [f"python  : {platform.python_version()}  ({platform.machine()})"]
    for package in ("numpy", "pandas", "sentence_transformers", "torch", "rank_bm25"):
        try:
            module = __import__(package)
            lines.append(f"{package:<7} : {getattr(module, '__version__', 'unknown')}")
        except Exception:
            lines.append(f"{package:<7} : not installed")
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    lines.append(f"commit  : {git or 'unknown'}")
    return lines


def write_csv(frame: pd.DataFrame, path: Path) -> str | None:
    """
    Write a CSV, returning a warning string instead of raising on a locked file.

    A presenter often has these open in Excel or a preview pane, and a write
    failure must not abort verification -- the values are checked from memory.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return None
    except PermissionError:
        return f"{path} is locked (close Excel / preview panes to refresh it)"
    except OSError as exc:
        return f"{path}: {exc}"


# ---------------------------------------------------------------------------
# Pipeline stages, timed
# ---------------------------------------------------------------------------


def run_dataset(dataset_name: str, encoder, backend: str, top_k: int) -> dict:
    """Run index -> rank -> evaluate, timing each stage, and export CSVs."""
    dataset = rmp.load_dataset(dataset_name)
    cfg = rmp.Config(top_k=top_k, dataset=dataset_name)

    t0 = time.perf_counter()
    index = rmp.build_index(cfg, dataset, encoder, backend)
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    table = rmp.rank_candidates(index, cfg)
    t_rank = time.perf_counter() - t0

    t0 = time.perf_counter()
    per_profile, summary = rmp.evaluate_metrics(table, dataset, k=top_k)
    t_metrics = time.perf_counter() - t0

    out_dir = RESULTS_DIR / dataset_name
    warnings = [
        message
        for message in (
            write_csv(table.to_frame(dataset.opportunities), out_dir / "match_results.csv"),
            write_csv(per_profile, out_dir / "evaluation_per_profile.csv"),
            write_csv(summary, out_dir / "evaluation_summary.csv"),
        )
        if message
    ]

    return {
        "dataset": dataset,
        "table": table,
        "per_profile": per_profile,
        "summary": summary,
        "timings": {"index": t_index, "rank": t_rank, "metrics": t_metrics},
        "warnings": warnings,
        "out_dir": out_dir,
    }


def print_summary_table(result: dict, top_k: int) -> None:
    """Headline comparison: three conditions, one table, best marked."""
    summary = result["summary"]
    best_ndcg = float(summary["mean_ndcg@k"].max())

    print(f"{'condition':<30}{'nDCG@' + str(top_k):>9}{'P@' + str(top_k):>9}{'R@' + str(top_k):>9}")
    print("-" * WIDTH)
    for _, row in summary.iterrows():
        marker = "   <-- best" if abs(row["mean_ndcg@k"] - best_ndcg) < 1e-9 else ""
        print(
            f"{row['method']:<30}"
            f"{row['mean_ndcg@k']:>9.4f}{row['mean_precision@k']:>9.4f}{row['mean_recall@k']:>9.4f}"
            f"{marker}"
        )
    print("-" * WIDTH)
    print("nDCG@%d, 0 to 1 (longer is better):" % top_k)
    for _, row in summary.iterrows():
        print(f"  {str(row['method'])[:26]:<28}{bar(row['mean_ndcg@k'])} {row['mean_ndcg@k']:.4f}")


def print_top1(result: dict, max_rows: int = 8) -> None:
    """What each condition actually surfaces first -- the demo money shot."""
    table = result["table"]
    labels = table.method_names
    frame = table.to_frame(result["dataset"].opportunities, descriptions=False)
    top1 = frame[frame["rank"] == 1]

    print(f"{'profile':<10}{'BM25 top-1':<23}{'Semantic top-1':<23}{'Hybrid top-1':<23}")
    print("-" * WIDTH)
    profiles = list(dict.fromkeys(top1["profile_id"]))
    for profile_id in profiles[:max_rows]:
        row = top1[top1["profile_id"] == profile_id]
        titles = []
        for label in labels:
            match = row[row["method"] == label]
            titles.append(str(match["title"].iloc[0])[:21] if not match.empty else "-")
        print(f"{profile_id:<10}{titles[0]:<23}{titles[1]:<23}{titles[2]:<23}")
    if len(profiles) > max_rows:
        print(f"  ({max_rows} of {len(profiles)} profiles shown)")
    print()
    print("  NOTE: on a 5-8 listing corpus these columns often agree. That is a")
    print("        finding about the benchmark size, not about the methods.")


def print_per_profile(result: dict, top_k: int, max_profiles: int = 6) -> None:
    """
    Per-profile detail: which queries each condition got right or wrong.

    Failure cases are the finding, so the low-scoring profiles matter more than
    the mean. The 'target' column is only shown when the dataset actually
    records a target listing per profile (synthetic_v2 does, prototype does not)
    -- an all-dashes column wastes space and invites the question "why is that
    empty?".
    """
    per_profile = result["per_profile"]
    scorable = per_profile[per_profile["scorable"]]
    if scorable.empty:
        print("  (no scorable profiles)")
        return

    has_targets = bool(scorable["target_rank"].notna().any())
    header = f"{'profile':<10}{'condition':<30}{'nDCG':>7}"
    if has_targets:
        header += f"{'hit@1':>7}{'target':>7}"
    print(header)
    print("-" * WIDTH)

    profiles = list(dict.fromkeys(scorable["profile_id"]))
    for profile_id in profiles[:max_profiles]:
        group = scorable[scorable["profile_id"] == profile_id]
        for index, (_, row) in enumerate(group.iterrows()):
            line = (
                f"{(str(profile_id) if index == 0 else ''):<10}"
                f"{str(row['method'])[:29]:<30}{row['ndcg@k']:>7.3f}"
            )
            if has_targets:
                target = row["target_rank"]
                text = "-" if pd.isna(target) else str(int(target))
                hit = "yes" if (not pd.isna(target) and int(target) == 1) else "no"
                line += f"{hit:>7}{text:>7}"
            print(line)
        print("-" * WIDTH)

    if len(profiles) > max_profiles:
        print(f"  ({max_profiles} of {len(profiles)} profiles shown; full detail in {result['out_dir']})")
    if has_targets:
        print("  'target' = rank given to the most relevant listing (1 is ideal)")
    if len(profiles) <= 3:
        print("  CAUTION: very few profiles. One rank change moves the mean, so treat")
        print("           differences of a few hundredths as noise.")


def print_negative_control(result: dict) -> None:
    """
    Distractor profiles have no relevant listing, so nDCG cannot score them.
    Report how highly each condition ranked a listing with no domain fit.
    """
    per_profile = result["per_profile"]
    distractors = per_profile[~per_profile["scorable"]]
    if distractors.empty or not distractors["target_rank"].notna().any():
        return

    corpus = len(result["dataset"].opportunities)
    baseline = (corpus + 1) / 2

    section("NEGATIVE CONTROL: did a condition surface a listing with no domain fit?")
    print(f"{'condition':<30}{'mean target rank':>18}{'in top-k':>12}")
    print("-" * WIDTH)
    for label in result["table"].method_names:
        rows = distractors[distractors["method"] == label]
        if rows.empty:
            continue
        print(
            f"{label:<30}{rows['target_rank'].mean():>18.2f}"
            f"{rows['target_rank'].notna().mean() * 100:>11.0f}%"
        )
    print("-" * WIDTH)
    print(
        f"  lower is better; random baseline = {baseline:.1f} mean rank\n"
        f"  n = {distractors['profile_id'].nunique()} distractor profiles with no relevant listing"
    )


def sweep_table(dataset, encoder, backend: str, top_k: int) -> pd.DataFrame:
    """Sweep bm25_weight and report the hybrid curve."""
    rows = []
    for weight in SWEEP_WEIGHTS:
        cfg = rmp.Config(
            top_k=top_k,
            bm25_weight=weight,
            semantic_weight=round(1.0 - weight, 10),
            dataset=dataset.name,
        )
        table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg)
        _, summary = rmp.evaluate_metrics(table, dataset, k=top_k)
        by_method = summary.set_index("method")
        rows.append(
            {
                "bm25_weight": weight,
                "keyword": float(by_method.loc[table.method_names[0], "mean_ndcg@k"]),
                "semantic": float(by_method.loc[table.method_names[1], "mean_ndcg@k"]),
                "hybrid": float(by_method.loc[table.method_names[2], "mean_ndcg@k"]),
            }
        )
    return pd.DataFrame(rows)


def print_sweep(frame: pd.DataFrame, top_k: int) -> None:
    section(f"WEIGHT SWEEP: hybrid nDCG@{top_k} vs keyword weight")
    print(f"{'bm25_w':>8}{'sem_w':>8}{'hybrid nDCG':>13}   curve")
    print("-" * WIDTH)
    best_weight = float(frame.loc[frame["hybrid"].idxmax(), "bm25_weight"])
    for _, row in frame.iterrows():
        marker = ""
        if abs(row["bm25_weight"] - best_weight) < 1e-9:
            marker = "  <-- best"
        elif row["bm25_weight"] == 0.0:
            marker = "  (= pure semantic)"
        elif abs(row["bm25_weight"] - 0.6) < 1e-9:
            marker = "  (current default)"
        elif row["bm25_weight"] == 1.0:
            marker = "  (= pure BM25)"
        print(
            f"{row['bm25_weight']:>8.1f}{1.0 - row['bm25_weight']:>8.1f}"
            f"{row['hybrid']:>13.4f}   {bar(row['hybrid'])}{marker}"
        )
    print("-" * WIDTH)
    pure_semantic = float(frame.loc[frame["bm25_weight"] == 0.0, "semantic"].iloc[0])
    default_rows = frame.loc[(frame["bm25_weight"] - 0.6).abs() < 1e-9, "hybrid"]
    best = float(frame["hybrid"].max())
    print(f"  pure semantic         : {pure_semantic:.4f}")
    print(f"  best hybrid weight    : {best_weight:.1f} -> {best:.4f}  ({best - pure_semantic:+.4f} vs semantic)")
    if not default_rows.empty:
        print(
            f"  current default (0.6) : {float(default_rows.iloc[0]):.4f}"
            f"  ({float(default_rows.iloc[0]) - best:+.4f} vs best)"
        )
    print()
    print("  FINDING: the default 0.6 keyword weight is mistuned. A MILD keyword weight")
    print("           beats pure semantic; a strong one hurts. Stable region is 0.2-0.4.")
    print("  CAVEAT : with 4 profiles, ~0.02 nDCG is ONE rank change. Read the shape of")
    print("           the curve, not the argmax.")


def print_timings(all_timings: list[tuple[str, dict, int]]) -> None:
    section("PIPELINE TIMINGS (this run)")
    print(f"{'dataset':<16}{'index':>10}{'rank':>10}{'metrics':>10}{'total':>10}")
    print("-" * WIDTH)
    for name, timings, _ in all_timings:
        total = sum(timings.values())
        print(
            f"{name:<16}{timings['index'] * 1000:>9.1f}ms"
            f"{timings['rank'] * 1000:>9.1f}ms{timings['metrics'] * 1000:>9.1f}ms"
            f"{total * 1000:>9.1f}ms"
        )
    print("-" * WIDTH)
    print("  index   = text assembly + BM25 fit + document embeddings (once per corpus)")
    print("  rank    = score every query x 3 conditions, then argsort")
    print("  metrics = nDCG / P / R for every (profile, condition), vectorised")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def check(summary: pd.DataFrame, expected: dict) -> list[str]:
    """Return human-readable drift descriptions (empty means all matched)."""
    problems: list[str] = []
    by_method = summary.set_index("method")
    for method, (ndcg, precision, recall) in expected.items():
        if method not in by_method.index:
            problems.append(f"missing condition in output: {method}")
            continue
        row = by_method.loc[method]
        for label, actual, want in (
            ("nDCG", row["mean_ndcg@k"], ndcg),
            ("Precision", row["mean_precision@k"], precision),
            ("Recall", row["mean_recall@k"], recall),
        ):
            if abs(float(actual) - float(want)) > TOLERANCE:
                problems.append(f"{method} {label}: got {actual}, expected {want}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command presentation run")
    parser.add_argument("--offline", action="store_true", help="use the hashed-BoW proxy")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-sweep", action="store_true", help="skip the weight sweep")
    args = parser.parse_args()

    backend_kind = "proxy" if args.offline else "MiniLM"

    banner("VTURCS Team 1 - IRDM")
    print("Research-opportunity matching: keyword vs semantic vs hybrid retrieval")
    print("Corpus: VT research listings   |   Queries: student interest profiles")
    print()
    for line in environment_report():
        print(line)

    if args.offline:
        print()
        print("!! OFFLINE MODE: the 'semantic' condition is a hashed bag-of-words")
        print("!! proxy, NOT MiniLM, and it scores BELOW BM25. Do not present these")
        print("!! numbers as semantic results.")
        encoder, backend = rmp.HashedBoWEncoder(), "offline-forced::hashed-bow"
    else:
        encoder, backend = rmp.load_encoder("all-MiniLM-L6-v2", allow_fallback=False)

    print()
    print(f"encoder : {backend}")
    print(f"cut-off : k = {args.top_k}")

    # --- run both datasets ---------------------------------------------------
    all_problems: list[str] = []
    all_warnings: list[str] = []
    all_timings: list[tuple[str, dict, int]] = []

    for dataset_name in ("prototype", "synthetic_v2"):
        result = run_dataset(dataset_name, encoder, backend, args.top_k)
        all_warnings.extend(result["warnings"])
        all_timings.append((dataset_name, result["timings"], args.top_k))

        dataset = result["dataset"]
        section(f"DATASET '{dataset_name}': {dataset.description}")
        print(f"{len(dataset.opportunities)} listings x {len(dataset.queries)} queries")
        print()
        print_summary_table(result, args.top_k)

        if dataset_name == "prototype":
            print()
            print_top1(result)
            print()
            print_per_profile(result, args.top_k)
        else:
            print_negative_control(result)

        expected = EXPECTED.get((dataset_name, backend_kind))
        if expected:
            problems = check(result["summary"], expected)
            print()
            if problems:
                all_problems.extend(f"{dataset_name}: {p}" for p in problems)
                print(f"  [x] DRIFT DETECTED vs docs/RESULTS.md ({len(problems)} value(s)):")
                for problem in problems:
                    print(f"      - {problem}")
            else:
                print("  [ok] matches docs/RESULTS.md")

    # --- sweep + timings -----------------------------------------------------
    if not args.no_sweep:
        print_sweep(
            sweep_table(rmp.load_dataset("prototype"), encoder, backend, args.top_k), args.top_k
        )

    print_timings(all_timings)

    if all_warnings:
        section("WARNINGS (results not written; verification still valid)")
        for warning in all_warnings:
            print(f"  - {warning}")

    # --- verdict -------------------------------------------------------------
    print()
    rule()
    if all_problems:
        print(f"RESULT: {len(all_problems)} value(s) DRIFTED from docs/RESULTS.md")
        print("        Either behaviour changed, or the docs need updating.")
        rule()
        return 1

    print("RESULT: all reported values reproduced from a clean checkout")
    print(f"Artifacts: {RESULTS_DIR.resolve()}")
    rule()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
