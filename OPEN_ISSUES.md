# Open Issues Log: Prototype v1

Last updated: 2026-10-16 | Severity: H / M / L | Owner: [TBD]

## Technical
| # | Issue | Sev | Next step |
|---|---|---|---|
| T1 | Corpus is 8 listings (`prototype`) and 5 (`synthetic_v2`), so Top-10 would be nearly the whole corpus. Prototype v1 reports k=5, not the required @10. | H | Expand to >=50 listings (README target), then report @10. Until then, document the deviation. |
| T2 | Two implementations disagree: RRF hybrid (`main`) vs min-max hybrid (`Sai_Prototype`); P@10/nDCG@10 (`main`) vs @5 + Recall (`Sai_Prototype`). | H | Team decision on one hybrid for the primary condition; keep the other as an ablation. |
| T3 | ~~No committed tests.~~ **RESOLVED**: `tests/` added - 69 tests covering metric equivalence vs brute force, retrieval properties, dataset validation contracts, pinned baselines, and CLI/export behaviour. | M | Done. Run `pytest -q`. |
| T4 | ~~Results CSVs are git-ignored, so no results artifact is committed.~~ **RESOLVED**: `results/` is committed (un-ignored) and produced by `scripts/reproduce.py`. | M | Done. |
| T5 | `--offline-embeddings` replaces MiniLM with a lexical proxy; misreporting risk. | M | Keep backend label in all exports and tables. (Labels verified by test; see docs/PRESENTATION.md sec 6.) |
| T6 | Two files named `benchmark.py` with different purposes (`main` vs `scripts/benchmark.py`). | L | Rename the timing harness (e.g. `scripts/timing.py`). |
| T7 | Python guidance conflicts: docs say verified on 3.13, requirements.txt recommends 3.10-3.12. | L | Pin one tested version. |
| T8 | ~~MiniLM-based numbers are unreproduced by a second teammate.~~ **RESOLVED**: `scripts/reproduce.py` regenerates and verifies all reported values, and `tests/test_baselines.py` pins them. Committed environment recorded in docs/RESULTS.md. | M | Done. A teammate should still run it on a second machine to confirm. |
| T9 | Presentation materials were spread across five docs with no run order or demo script. | L | **Addressed**: `docs/PRESENTATION.md` adds a runbook, demo sequence, cheat sheet, and offline-trap warning. |

## Methodological
| # | Issue | Sev | Next step |
|---|---|---|---|
| M1 | `synthetic_v2` labels are derived from `match_type`: exactly one relevant listing per profile, all other pairs assumed 0. This biases absolute metrics downward and penalizes valid cross-matches. | H | Hand-grade multiple listings per profile. |
| M2 | Structural degeneracy in `synthetic_v2`: P@k ~ 1/k, Recall is binary, nDCG reduces to rank-1 success. Only nDCG is informative. | M | Resolve via M1. |
| M3 | Keyword leakage in the `semantic_analogy` stratum (a substring hit let BM25 win). A 5-document corpus cannot support claims about keyword vs semantic. | H | Larger corpus; remove leakage from profiles. |
| M4 | Hybrid is min-max score blending with keyword priority (0.6/0.4). On the 4-profile prototype, hybrid scores below pure semantic. Weight is not yet tuned. | M | Run `--bm25-weight` sweep (0.0-1.0) before drawing conclusions. |
| M5 | Labels are single-rater and hand-written; no inter-rater agreement. | M | Second rater on a subset; compute agreement. |
| M6 | Corpus is synthetic, not real public VT listings, while the RQ concerns VT opportunities. | H | Collect real public listings. |
| M7 | Only 4 profiles in the hand-graded set; one rank change moves means noticeably. | M | More profiles; report per-profile results. |

## Human-subjects dependencies
| # | Issue | Sev | Next step |
|---|---|---|---|
| H1 | Prototype v1 uses fictional, non-participant profiles. IRB status for later phases is **[TBD]**. | H | Confirm with faculty advisor whether IRB review/exemption is needed before any real student profiles are collected. |
| H2 | `data/student_profiles.csv` includes a `full_name` column plus major and year. Confirm all are fictional. | M | Label as synthetic in the file/README; drop the name column or never replace it with real names. |
| H3 | Recruitment, consent, data storage, and retention for participant studies are **[TBD]**. | H | Draft once H1 is answered. |
| H4 | Relevance judgments by team members or students may count as human data. **[TBD]** | M | Check with advisor. |

## Fixed during presentation prep

Issues found while getting this branch ready to present. Listed here so the
reasoning is auditable rather than buried in a commit message.

| # | Issue | Resolution |
|---|---|---|
| F1 | `--bm25-weight 1.5` was accepted: it derives `semantic_weight = -0.5`, which sums to 1.0 and so passed a sum-only check. A negative weight encodes a nonsense rule (negatively weighted semantic signal) while still producing a plausible-looking ranking. | Weights are now range-checked to [0, 1] in `Config` and at the CLI boundary, so a bad value is a usage error (exit 2) instead of a traceback. `top_k` and `batch_size` are range-checked too. Covering tests added. |
| F2 | Presentation docs had no run order, demo script, or offline warning. | Added `docs/PRESENTATION.md`. |
| F3 | `docs/RESULTS.md` carried a `[TBD: replace with a committed evaluation_summary.csv]` marker, meaning the headline numbers were not reproducible from the repo. | Resolved via `results/` + `scripts/reproduce.py` + pinned baseline tests (see T4/T8). |
