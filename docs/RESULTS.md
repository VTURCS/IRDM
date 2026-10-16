# Initial Results: Prototype v1 (DRAFT)

Dataset: `prototype` | 8 synthetic listings | 4 hand-graded profiles | **k = 5** (not 10)
Labels: 0-2 rubric. Precision and Recall treat grade >= 1 as relevant.

| Condition | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword (BM25) | 0.7046 | 0.55 | 0.7417 |
| Semantic (MiniLM) | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 BM25 / 0.4 MiniLM) | 0.8364 | 0.65 | 0.8875 |

**Provenance (resolved).** All three rows are now independently reproduced from a clean
checkout. The artifact is committed at `results/prototype/evaluation_summary.csv`, generated
by `scripts/reproduce.py`, which re-runs the pipeline and fails if any value drifts from the
table above. Reproduce with:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce.py
```

Environment for the committed artifact: Python 3.13.14, numpy 2.5.3, pandas 3.0.6,
sentence-transformers 6.1.0, torch 2.14.1+cpu. The pinned baselines are also enforced by
`tests/test_baselines.py`, so a regression fails the test suite rather than silently
changing a published number.

## Secondary: `synthetic_v2` (5 listings, 20 profiles, derived labels), mean nDCG@5
| Condition | nDCG@5 |
|---|---|
| Keyword (BM25) | 0.9667 |
| Semantic (MiniLM) | 0.9421 |
| Hybrid (0.6/0.4) | 0.9754 |

Artifact: `results/synthetic_v2/evaluation_summary.csv`.
Interpret with caution: derived labels and a 5-document corpus (OPEN_ISSUES M1-M3).
Note that every condition reaches ~1.0 — all three methods put the target listing at rank 1
in almost every case, which means this corpus cannot distinguish the conditions at all.

## Reading the results
- On the prototype set, semantic leads and the hybrid sits between the two methods.
  This is consistent with keyword-priority weighting hurting on short, vague queries,
  but with 4 profiles and 8 listings it is a hypothesis to test, not a finding.
- The `--bm25-weight` sweep is the next experiment (OPEN_ISSUES M4).
- Absolute values are not the finding; relative ordering and failure cases are.
