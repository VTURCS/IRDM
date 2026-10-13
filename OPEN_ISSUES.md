# Open Issues Log: Prototype v1

Last updated: 2026-10-12 | Severity: H / M / L | Owner: [TBD]

## Technical
| # | Issue | Sev | Next step |
|---|---|---|---|
| T1 | Corpus is 8 listings (`prototype`) and 5 (`synthetic_v2`), so Top-10 would be nearly the whole corpus. Prototype v1 reports k=5, not the required @10. | H | Expand to >=50 listings (README target), then report @10. Until then, document the deviation. |
| T2 | Two implementations disagree: RRF hybrid (`main`) vs min-max hybrid (`Sai_Prototype`); P@10/nDCG@10 (`main`) vs @5 + Recall (`Sai_Prototype`). | H | Team decision on one hybrid for the primary condition; keep the other as an ablation. |
| T3 | No committed tests. The brute-force metric verification described in docs/verification_log.md is not in the repo. | M | Add `tests/` with a metric sanity check and a smoke run. |
| T4 | Results CSVs are git-ignored, so no results artifact is committed. | M | Commit a snapshot under `docs/` or `results/`. |
| T5 | `--offline-embeddings` replaces MiniLM with a lexical proxy; misreporting risk. | M | Keep backend label in all exports and tables. |
| T6 | Two files named `benchmark.py` with different purposes (`main` vs `scripts/benchmark.py`). | L | Rename the timing harness (e.g. `scripts/timing.py`). |
| T7 | Python guidance conflicts: docs say verified on 3.13, requirements.txt recommends 3.10-3.12. | L | Pin one tested version. |
| T8 | MiniLM-based numbers are unreproduced by a second teammate. | M | Re-run with real MiniLM, commit versions (`pip freeze`). |

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
