# Week 5 pilot results and team handoff

## Cases and preliminary results

Verified October 12, 2026, using Python 3.12 and `requirements-lock.txt`. These are software checks on 12 fictional opportunities and 60 illustrative labels, not independently validated VT research findings.

| Profile | Case | Illustrative relevant opportunities |
| --- | --- | --- |
| p01 | Specific robotics / computer vision interest | Robot navigation (strong), image recognition (partial). |
| p02 | Specific environmental chemistry interest | Stream water quality (strong), watershed mapping (partial). |
| p03 | Different vocabulary: “Automata perceiving surroundings” | Robot navigation (strong), image recognition (partial). No keyword overlap. |
| p04 | Undecided area, enjoys outdoor questions | Water quality, watershed mapping, insect ecology (partial). |
| p05 | Quantum cryptography, absent from the corpus | None. |

| Method | Precision@10 (5 profiles) | Recall@10 (4 profiles) | nDCG@10 (4 profiles) |
| --- | ---: | ---: | ---: |
| BM25 | 0.1400 | 0.7917 | 0.8157 |
| Semantic | 0.1800 | 1.0000 | 0.9919 |
| Hybrid | 0.1800 | 1.0000 | 0.9714 |

The no-match case has precision zero and undefined recall/nDCG. The system still returns ten results; it does not abstain. High recall is easy in this tiny corpus because ten of twelve listings are returned. p03 gets perfect BM25 nDCG purely because ID tie-breaking happens to rank the relevant items first—not because BM25 understands the query. Hybrid omits keyword credit when all BM25 scores are zero; partially tied keyword rankings can still affect fusion.

Nine tests cover hand-calculated metrics, missing judgments, no-match cases, invalid labels, deterministic ties, weighted fusion, zero-keyword fallback, and complete demo fixtures. The full MiniLM run produces 150 rankings and 15 metric rows. The CLI was also checked for stale-metric replacement when labels are absent and rejection of empty study inputs.

Rerun the demo to obtain the per-profile CSVs. Model revision is pinned in code, and the lock file preserves the tested Windows dependencies; `requirements.txt` remains an unpinned alternative. Preserve input/output CSVs and the code revision for each study run. Failed runs can leave old outputs: check the exit status. Small numerical differences across hardware remain possible.

## Proposed labeling rules

These require team agreement before the real study. John coordinates evaluation; two raters and a third reviewer still need to be named. Independently score ten calibration pairs without seeing method, rank, or score; discuss disagreements and clarify rules. Then independently label the full small corpus for each profile. Preserve original grades, evidence, missing information, and adjudication notes. Resolve disagreements by discussion, referring unresolved cases to the designated third reviewer. Export final agreed grades to `relevance.csv` and report pre-resolution agreement.

| Grade | Proposed rule |
| --- | --- |
| 0 — unsuitable | No supported topic connection, or an explicit mandatory requirement is unmet. |
| 1 — partial | Broad/indirect topic fit, or a relevant topic with an important mandatory requirement unverified. |
| 2 — strong | Clear topic fit and stated mandatory requirements met. If none are stated, flag eligibility as unspecified rather than inventing requirements. |

Distinguish required from preferred skills and training offered. Missing information is not disqualification; explicit ineligibility cannot be offset by topic similarity. No 50/30/20 rubric is implemented. The demo does not test eligibility rules because its listings lack mandatory requirements. For vague profiles, judge stated interests only and document uncertainty; query expansion remains deferred. Keep evaluation at ten; displaying 3–5 recommendations is a future interface decision.

## Ownership and contributions

Planned ownership from the Week 4 specification—not a claim that each task is complete:

| Person | Responsibility / integration |
| --- | --- |
| Sai | Source-linked VT listings and profiles shared by all methods. |
| Sandilya | BM25 rankings supplied to hybrid and evaluation. |
| Vaidehi | MiniLM and weighted hybrid rankings. |
| John | Rubric, label coordination, metrics, integrated run and report. |

Baseline `663c3a7` provided the single-script benchmark and two synthetic cases. This iteration, prepared with Codex at John's request, adds Recall@10, zero-keyword fusion handling, three cases, focused tests, pinned model/dependencies, and Week 5 documentation. Each teammate must confirm their actual completed artifact, checks, and integration point before submission.

## Remaining work

Collect real public VT listings with source links and availability/eligibility evidence; agree raters and rubric; preserve independent labels and disagreement decisions; freeze inputs and run the comparison. Research-interest pages do not establish open positions. No live student data is needed for this pilot. General effectiveness, student satisfaction, and optimal hybrid weights are not established by these synthetic results.
