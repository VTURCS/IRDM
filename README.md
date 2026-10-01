# VTURCS Team 1 — IRDM Prototype

Prototype research-matching evaluation pipeline: **how does a keyword-prioritized hybrid
retriever compare to keyword-only and semantic-only retrieval when matching short, vague
undergraduate profiles to Virginia Tech research opportunity listings?**

## Files

| File | Purpose |
| --- | --- |
| `research_matching_pipeline.py` | The whole pipeline: mock data, 3 retrievers, metrics, report, CSV export |
| `requirements.txt` | Dependencies |
| `outputs/` | Generated CSVs (created on first run, git-ignored) |

## Quick start

The project uses a local virtual environment (`.venv`) so PyTorch and
`sentence-transformers` stay isolated from the system interpreter.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe research_matching_pipeline.py
```

Verified working on **Python 3.13** with `torch 2.14.1+cpu` and
`sentence-transformers 6.1.0`. The first run downloads the ~90 MB `all-MiniLM-L6-v2`
model into the Hugging Face cache; later runs are offline and instant.

If `sentence-transformers` / PyTorch can't be installed or downloaded (restricted network,
air-gapped cluster), the pipeline still runs end to end:

```powershell
.\.venv\Scripts\python.exe research_matching_pipeline.py --offline-embeddings
```

Useful flags:

```powershell
.\.venv\Scripts\python.exe research_matching_pipeline.py --top-k 8 --bm25-weight 0.7 --output-dir outputs/run2
```

> **Fallback caveat:** with `--offline-embeddings` the "semantic" method is a hashed
> bag-of-words **lexical proxy, not MiniLM**. The method label and the exported CSVs record
> which backend ran, so a report can never accidentally claim MiniLM results it doesn't have.

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

## Outputs

| File | Columns |
| --- | --- |
| `match_results.csv` | `profile_id, method, opportunity_id, rank, score, title, description` |
| `evaluation_per_profile.csv` | metrics per profile × method (+ `n_relevant`, `n_strong`) |
| `evaluation_summary.csv` | mean nDCG / Precision / Recall per method — the headline table |

## Prototype finding (mock data, k = 5)

With real MiniLM embeddings the ranking **does not** favour the keyword-prioritized hybrid:

| Method | mean nDCG@5 | mean P@5 | mean R@5 |
| --- | --- | --- | --- |
| Keyword (BM25) | 0.7046 | 0.55 | 0.74 |
| **Semantic (MiniLM)** | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 / 0.4) | 0.8364 | 0.65 | 0.8875 |

Semantic wins on every profile; the hybrid lands *between* the two, strictly worse than pure
semantic. The mechanism is in the fusion rule: a listing that is a strong semantic match but
shares no vocabulary with the vague query receives `bm25_norm = 0`, so the 0.4 semantic weight
is all it keeps — while a lexicon-overlap-only listing gets `bm25_norm = 1` and outranks it.

This is exactly the failure mode the study is set up to detect, and it makes the
**`--bm25-weight` sweep (0.0 → 1.0) the highest-value ablation**: for short, vague student
queries, a keyword *priority* may simply be the wrong prior. Re-run the sweep before drawing
conclusions — with a synthetic corpus of 8 items and hand-written labels, the absolute numbers
above are not the finding, and a single listing changing its rank moves the means noticeably.

## Prototype caveat

The corpus is **synthetic** and labels are **hand-written**. Absolute metric values are not
the finding — the relative ordering of methods and the failure cases are what this pilot
exists to surface before the study scales to real listings. Where the pilot already delivers:
it produced a concrete, mechanistically-explained hypothesis (keyword priority hurts on vague
queries) and a measurable next experiment to test it.
