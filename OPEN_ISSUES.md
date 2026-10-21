# Open Issues Log: Prototype v1

Last updated: 2026-10-21 | Severity: H / M / L | Owner: [TBD]

## Technical
| # | Issue | Sev | Next step |
|---|---|---|---|
| T1 | Corpus is 8 listings (`prototype`) and 5 (`synthetic_v2`), so Top-10 would be nearly the whole corpus. Prototype v1 reports k=5, not the required @10. | H | Expand to >=50 listings (README target), then report @10. Until then, document the deviation. |
| T2 | Two implementations disagree: RRF hybrid (`main`) vs min-max hybrid (`Sai_Prototype`); P@10/nDCG@10 (`main`) vs @5 + Recall (`Sai_Prototype`). | H | Team decision on one hybrid for the primary condition; keep the other as an ablation. |
| T3 | ~~No committed tests.~~ **RESOLVED**: `tests/test_irdm.py` covers the published numbers on all four datasets, the weight sweep, nDCG by hand, tie-breaking, CSV validation, the CLI and the web UI. | M | Done. Run `docker compose run --rm irdm pytest`. |
| T4 | ~~Results CSVs are git-ignored, so no results artifact is committed.~~ **RESOLVED**: `results/` is committed (un-ignored) and produced by `python evaluate.py <dataset> --output-dir results`. | M | Done. |
| T5 | ~~`--offline-embeddings` replaces MiniLM with a lexical proxy; misreporting risk.~~ **RESOLVED**: the offline proxy was removed. The Docker image ships the pinned model, so no run can silently fall back. | M | Done. |
| T6 | ~~Two files named `benchmark.py` with different purposes.~~ **RESOLVED**: the timing harness was removed in the simplification. | L | Done. |
| T7 | ~~Python guidance conflicts across docs.~~ **RESOLVED**: the Docker image pins Python 3.13, and `requirements.txt` pins the verified package versions. | L | Done. |
| T8 | ~~MiniLM-based numbers are unreproduced by a second teammate.~~ **RESOLVED**: `tests/test_irdm.py` pins every reported value, and the tests pass both in Docker and on Windows. | M | Done. A teammate should still run it on a second machine to confirm. |
| T9 | Presentation materials were spread across five docs with no run order or demo script. | L | **Addressed**: `docs/PRESENTATION.md` adds a runbook, demo sequence, cheat sheet, and offline-trap warning. |

## Methodological
| # | Issue | Sev | Next step |
|---|---|---|---|
| M1 | `synthetic_v2` labels are derived from `match_type`: exactly one relevant listing per profile, all other pairs assumed 0. This biases absolute metrics downward and penalizes valid cross-matches. | H | Hand-grade multiple listings per profile. |
| M2 | Structural degeneracy in `synthetic_v2`: P@k ~ 1/k, Recall is binary, nDCG reduces to rank-1 success. Only nDCG is informative. | M | Resolve via M1. |
| M3 | Keyword leakage in the `semantic_analogy` stratum (a substring hit let BM25 win). A 5-document corpus cannot support claims about keyword vs semantic. | H | Larger corpus; remove leakage from profiles. |
| M4 | Hybrid is min-max score blending with keyword priority (0.6/0.4). **Sweep run (2026-10-16)**: the default 0.6 is mistuned - nDCG@5 peaks at weight 0.4 (0.9030) vs 0.8846 for pure semantic and 0.8364 at 0.6. | M | Decide whether to change the default to ~0.3-0.4, and re-run the sweep on the expanded corpus before treating the argmax as real. See docs/RESULTS.md. |
| M5 | Labels are single-rater and hand-written; no inter-rater agreement. | M | Second rater on a subset; compute agreement. |
| M6 | Corpus is synthetic, not real public VT listings, while the RQ concerns VT opportunities. | H | Collect real public listings. |
| M7 | Only 4 profiles in the hand-graded set; one rank change moves means noticeably. | M | More profiles; report per-profile results. |

## Human-subjects dependencies
| # | Issue | Sev | Next step |
|---|---|---|---|
| H1 | Prototype v1 uses fictional, non-participant profiles. IRB status for later phases is **[TBD]**. | H | Confirm with faculty advisor whether IRB review/exemption is needed before any real student profiles are collected. |
| H2 | ~~`data/student_profiles.csv` includes a `full_name` column plus major and year.~~ **RESOLVED**: the name column was dropped when `synthetic_v2` moved to `data/synthetic_v2/`; profiles carry no names, only ids, query text and metadata. | M | Never add real names. |
| H3 | Recruitment, consent, data storage, and retention for participant studies are **[TBD]**. | H | Draft once H1 is answered. |
| H4 | Relevance judgments by team members or students may count as human data. **[TBD]** | M | Check with advisor. |

## Fixed during presentation prep

Issues found while getting this branch ready to present. Listed here so the
reasoning is auditable rather than buried in a commit message.

| # | Issue | Resolution |
|---|---|---|
| F1 | `--bm25-weight 1.5` was accepted: it derives `semantic_weight = -0.5`, which sums to 1.0 and so passed a sum-only check. A negative weight encodes a nonsense rule (negatively weighted semantic signal) while still producing a plausible-looking ranking. | The CLI range-checks the weight to [0, 1] and `--top-k` to >= 1, so a bad value is a usage error (exit 2) instead of a traceback; the web UI falls back to the defaults. Covered by tests. |
| F2 | Presentation docs had no run order, demo script, or offline warning. | Added `docs/PRESENTATION.md`. |
| F3 | `docs/RESULTS.md` carried a `[TBD: replace with a committed evaluation_summary.csv]` marker, meaning the headline numbers were not reproducible from the repo. | Resolved via `results/` and pinned baseline tests (see T4/T8). |
