# Hybrid design, baselines, and metrics

> Moved from the original README (unchanged text).

## Pipeline overview

```
opportunities (mock, VT-001..VT-008)
        │
        └─ build_document()  ──►  identical text blob for every method
                                   (eligibility excluded — see baselines)
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
  KeywordSearcher                  SemanticSearcher                  hybrid_scores()
  BM25Okapi (negative               all-MiniLM-L6-v2                0.6*minmax(BM25)
  IDFs floored at 0)                + cosine similarity             + 0.4*minmax(cosine)
        │                                   │                                   │
        └───────────────────────────────────┼───────────────────────────────────┘
                                            ▼
                                 rank_scores() → Top-k per profile
                                            ▼
                          evaluate() → nDCG@k, Precision@k, Recall@k
                                            ▼
                     outputs/*.csv  +  console comparison report
```


## Baselines held constant

All three methods run against **the exact same corpus and the same verbatim queries**:

| Switch | Default | Rationale |
| --- | --- | --- |
| `metadata_filtering` | `False` | `eligibility` (major/year) never filters or re-ranks, so no method secretly benefits from eligibility overlap. |
| `query_expansion` | `False` | No synonyms, no PRF/RM3, no LLM rewriting — raw vague queries only. |
| `include_eligibility_in_text` | `False` | Eligibility text is not even in the searchable body. |

These are the first three planned ablations for the full study; they live in `Config` so they
cannot silently drift.


## How hybrid ranking works

BM25 is unbounded (≈0–15) while cosine similarity sits in a narrow band (≈0.1–0.7), so raw
averaging would let scale — not design intent — decide the winner. Both signals are
min-max normalized **per query across all candidates** to `[0, 1]`, then combined:

```
final = 0.6 * bm25_norm + 0.4 * semantic_norm
```

Documented consequences:
- This is **rank fusion by score scaling**, so the hybrid inherits BM25's brittleness to
  vocabulary mismatch: a purely semantic hit BM25 cannot see still gets `bm25_norm = 0` and
  keeps only 40% of its signal. That is the intended keyword-prioritized behaviour, and the
  planned `--bm25-weight` sweep (0.0 → 1.0) is what quantifies it.
- Normalization is per query, so scores are comparable **within** a query but not across.
- If a query produces all-zero BM25 scores, normalization returns zeros and the hybrid
  degrades gracefully to the semantic signal instead of dividing by zero.
- Ties break on `opportunity_id`, so runs are byte-identical and reproducible.


## Metrics

Labels are graded and defined in `RELEVANCE_LABELS`:

| Grade | Meaning |
| --- | --- |
| 0 | Unsuitable — unrelated topic/skills |
| 1 | Partial Match — adjacent topic or transferable skills |
| 2 | Strong Match — interests and skills line up with core work |

Labels are assigned from **content only** and deliberately ignore `eligibility` (since
filtering is off, a Sophomore can legitimately score 2 on a "Junior+" listing). The label
matrix covers the **whole corpus**, not just retrieved items — otherwise Recall@k would be
silently inflated.

- **nDCG@k (primary)** — graded gain `2^rel − 1` over `log2(i+1)`, normalized by the ideal
  ordering. Rewards putting the single best lab at rank 1 far more than three lukewarm options.
- **Precision@k** — binary (grade ≥ 1), always divided by `k`.
- **Recall@k** — binary (grade ≥ 1) over all relevant listings in the ground truth.

