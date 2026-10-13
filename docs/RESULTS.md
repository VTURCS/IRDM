# Initial Results: Prototype v1 (DRAFT)

Dataset: `prototype` | 8 synthetic listings | 4 hand-graded profiles | **k = 5** (not 10)
Labels: 0-2 rubric. Precision and Recall treat grade >= 1 as relevant.

| Condition | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword (BM25) | 0.7046 | 0.55 | 0.74 |
| Semantic (MiniLM) | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 BM25 / 0.4 MiniLM) | 0.8364 | 0.65 | 0.8875 |

Provenance: values as reported in the previous `Sai_Prototype` README (real MiniLM backend).
Only the BM25 row was independently reproduced during review; the MiniLM and hybrid rows
were not (no model download available). [TBD: replace with a committed
`evaluation_summary.csv` from a fresh run.]

## Secondary: `synthetic_v2` (5 listings, 20 profiles, derived labels), mean nDCG@5
| Condition | nDCG@5 |
|---|---|
| Keyword (BM25) | 0.9667 |
| Semantic (MiniLM) | 0.9421 |
| Hybrid (0.6/0.4) | 0.9754 |

Interpret with caution: derived labels and a 5-document corpus (OPEN_ISSUES M1-M3).

## Reading the results
- On the prototype set, semantic leads and the hybrid sits between the two methods.
  This is consistent with keyword-priority weighting hurting on short, vague queries,
  but with 4 profiles and 8 listings it is a hypothesis to test, not a finding.
- The `--bm25-weight` sweep is the next experiment (OPEN_ISSUES M4).
- Absolute values are not the finding; relative ordering and failure cases are.
