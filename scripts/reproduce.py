"""
Reproduce every reported number in one command.

This is the script a reviewer or teammate should run before a presentation. It:

  1. runs both datasets through all three retrieval conditions,
  2. writes the CSVs into `results/` so they can be committed as an artifact,
  3. prints the metric summary tables next to the values cited in docs/,
  4. records the exact environment (interpreter + package versions).

Usage
-----
    python scripts/reproduce.py                 # real MiniLM
    python scripts/reproduce.py --offline       # no model download

Exit code is 1 if any headline value drifts from its pinned expectation, so this
doubles as a lightweight regression gate.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research_matching_pipeline as rmp  # noqa: E402

# Values cited in docs/RESULTS.md. Tolerance is generous enough for float
# formatting differences but tight enough to catch a real change in behaviour.
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
        "Semantic (HashedBoW proxy)": (0.7046, 0.55, 0.7417),
        "Hybrid (0.6 KW / 0.4 SEM)": (0.7210, 0.60, 0.8042),
    },
}

RESULTS_DIR = Path("results")
RULE = "=" * 78


def environment_report() -> str:
    """Capture the exact environment so a result can be attributed to a setup."""
    lines = [
        f"python     : {platform.python_version()} ({platform.machine()})",
        f"platform   : {platform.platform()}",
    ]
    for package in ("numpy", "pandas", "rank_bm25", "sentence_transformers", "torch"):
        try:
            module = __import__(package)
            lines.append(f"{package:<11}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:  # not installed
            lines.append(f"{package:<11}: NOT INSTALLED ({type(exc).__name__})")

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        lines.append(f"git commit : {sha or 'unknown'}")
    except Exception:
        lines.append("git commit : unavailable")
    return "\n".join(lines)


def write_csv(frame: pd.DataFrame, path: Path) -> str | None:
    """
    Write a CSV, returning an error message instead of raising on a locked file.

    On Windows a results CSV left open in Excel or a preview pane raises
    PermissionError. That must not abort the run: the purpose of this script is
    to CHECK values, and verification does not need write access. The write
    failure is reported as a warning once the verification result is known, so a
    presenter who has the CSV open still gets the answer they came for.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return None
    except PermissionError:
        return f"{path} is locked by another process (close Excel / preview panes)"
    except OSError as exc:
        return f"{path}: {exc}"


def run_case(dataset_name: str, encoder, backend: str, top_k: int):
    """
    Run one dataset through the pipeline and return
    (summary, method_names, write_warnings).
    """
    dataset = rmp.load_dataset(dataset_name)
    cfg = rmp.Config(top_k=top_k, dataset=dataset_name)
    table = rmp.rank_candidates(rmp.build_index(cfg, dataset, encoder, backend), cfg)
    per_profile, summary = rmp.evaluate_metrics(table, dataset, k=top_k)

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
    return summary, table.method_names, out_dir, warnings


def check(summary: pd.DataFrame, expected: dict) -> list[str]:
    """Return a list of drift descriptions (empty means all matched)."""
    problems: list[str] = []
    by_method = summary.set_index("method")
    for method, (ndcg, precision, recall) in expected.items():
        if method not in by_method.index:
            problems.append(f"missing method in output: {method}")
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
    parser = argparse.ArgumentParser(description="Reproduce all reported numbers")
    parser.add_argument("--offline", action="store_true", help="use the hashed-BoW proxy")
    args = parser.parse_args()

    backend_kind = "proxy" if args.offline else "MiniLM"

    print(RULE)
    print("VTURCS Team 1 - IRDM: reproduction run")
    print(RULE)
    print(environment_report())
    print()

    if args.offline:
        encoder, backend = rmp.HashedBoWEncoder(), "offline-forced::hashed-bow"
        print("[note] offline mode: the 'semantic' condition is a hashed bag-of-words")
        print("[note] proxy, NOT MiniLM. Do not report these as semantic results.\n")
    else:
        encoder, backend = rmp.load_encoder("all-MiniLM-L6-v2", allow_fallback=False)

    all_problems: list[str] = []
    all_warnings: list[str] = []
    for dataset_name in ("prototype", "synthetic_v2"):
        summary, method_names, out_dir, warnings = run_case(dataset_name, encoder, backend, top_k=5)
        all_warnings.extend(warnings)

        print(f"--- {dataset_name} ---")
        # Print from the in-memory summary, not the file: the file may not have
        # been written if it was locked, and the values are what matter here.
        print(summary.to_string(index=False))
        print()

        expected = EXPECTED.get((dataset_name, backend_kind))
        if expected is None:
            print("  (no pinned values for this backend/dataset combination)\n")
            continue

        problems = check(summary, expected)
        if problems:
            all_problems.extend(f"{dataset_name}: {p}" for p in problems)
            print("  DRIFT detected:")
            for problem in problems:
                print(f"    - {problem}")
        else:
            print("  all pinned values matched")
        print()

    if all_warnings:
        print("WARNINGS (results could not be written; verification still valid):")
        for warning in all_warnings:
            print(f"  - {warning}")
        print()

    print(RULE)
    if all_problems:
        print(f"RESULT: {len(all_problems)} value(s) drifted from docs/RESULTS.md")
        print("        Either a real change occurred, or the docs need updating.")
        print(RULE)
        return 1

    print("RESULT: all reported values reproduced")
    print(f"Artifacts: {RESULTS_DIR.resolve()}")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
