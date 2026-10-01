# VTURCS Team 1 — IRDM Prototype

Prototype research-matching evaluation pipeline: **how does a keyword-prioritized hybrid
retriever compare to keyword-only and semantic-only retrieval when matching short, vague
undergraduate profiles to Virginia Tech research opportunity listings?**

## Files

| File | Purpose |
| --- | --- |
| `research_matching_pipeline.py` | **Pure logic** — retrievers, hybrid fusion, metrics, report, CSV export. Contains no data. |
| `data/` | All datasets + the canonical schema (see below) |
| `scripts/benchmark.py` | Phase-by-phase timing harness (before/after comparison) |
| `requirements.txt` | Dependencies |
| `outputs/<dataset>/` | Generated CSVs (created on first run, git-ignored) |

## Architecture

The pipeline is three explicit stages, so the expensive corpus-side work happens exactly once:

```
load_dataset(name)                    data package, canonical schema
        │
   build_index(cfg, dataset, encoder)   ──► SearchIndex  (immutable)
        │                                  text assembly, BM25 fit,
        │                                  doc embeddings: ONCE per corpus
        │
   rank_candidates(index, cfg)          ──► ResultTable  (numpy arrays)
        │                                  (3 x Q x C) score stack -> argsort
        │
   evaluate_metrics(table, dataset, k)  ──► per_profile, summary  (vectorised)
        │
   ResultTable.to_frame(listings)       ──► tidy CSV (built once, on export)
```

Key types: `SearchIndex` (cached retrieval state), `ResultTable` (ranked numpy arrays),
`Dataset` (data + precomputed grade matrix). All are frozen dataclasses.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --scale 2000    # phase timings
```

## Data layout

The pipeline holds **no** corpus, queries or labels — it resolves them by name from `data/`:

| File | Role |
| --- | --- |
| `data/opportunities.py` | 5 listings for the v2 benchmark (data as code) |
| `data/student_profiles.csv` | 20 student profiles (data as CSV) |
| `data/schema.py` | `Opportunity` / `Query` / `Dataset` dataclasses + validation |
| `data/prototype_dataset.py` | `prototype` — 8 listings, 4 short vague queries, hand-graded |
| `data/synthetic_v2_dataset.py` | `synthetic_v2` — 5 listings, 20 profiles, derived labels |
| `data/datasets.py` | Registry: dataset name → loader |

Adding a dataset = writing one `load() -> Dataset`. The retrievers, metrics and report
never change, and every loader normalises its own field names (`id` vs `opp_id`,
`required_skills` vs `key_terms`) so the pipeline only ever sees the canonical schema.

`Dataset.__post_init__` rejects two things that would silently corrupt results: duplicate
ids, and a `relevance_map` that does **not** cover every (query, listing) pair — the latter
because Recall@k divides by the relevant-set size and a partial map inflates it.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe research_matching_pipeline.py                          # prototype (default)
.\.venv\Scripts\python.exe research_matching_pipeline.py --dataset synthetic_v2    # the 5x4 benchmark
```

Verified on **Python 3.13** with `torch 2.14.1+cpu` and `sentence-transformers 6.1.0`.
The first run downloads the ~90 MB `all-MiniLM-L6-v2` model; later runs are offline.

Other flags:

```powershell
--match-type semantic_analogy distractor   # label-stratified ablation (synthetic_v2)
--top-k 8                                  # override the dataset's suggested k
--bm25-weight 0.7                          # hybrid weight sweep
--offline-embeddings                       # no model download; hashed-BoW proxy
--output-dir outputs/run2
```

Each run writes to `outputs/<dataset>[/__<strata>]/` so runs never overwrite each other.

> **Fallback caveat:** with `--offline-embeddings` the "semantic" method is a hashed
> bag-of-words **lexical proxy, not MiniLM**. The method label and exported CSVs record which
> backend ran, so a report can never claim MiniLM results it doesn't have.

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
| `evaluation_per_profile.csv` | metrics per profile × method (+ `n_relevant`, `n_strong`, `scorable`, `target_rank`) |
| `evaluation_summary.csv` | mean nDCG / Precision / Recall per method over **scorable** profiles |

## Datasets

| | `prototype` (default) | `synthetic_v2` |
| --- | --- | --- |
| Listings | 8 | 5 |
| Profiles | 4 short, vague | 20 long-form |
| Labels | hand-graded per pair | **derived** from `match_type` |
| Positive labels/profile | 3–5 | exactly 1 (0 for distractors) |

`prototype` reproduces the original verified numbers. `synthetic_v2` is the purpose-built
benchmark: 5 listings × 4 match types (`exact_keyword`, `semantic_analogy`, `distractor`,
`noisy_edge`), designed so the lexical/semantic contrast is measurable by stratum.

## Two measurement problems in `synthetic_v2` (read before citing its numbers)

**1. Derived labels are a lower bound, not ground truth.** Each profile is graded 2 at its
`target_opp_id` (1 for `noisy_edge`, 0 for `distractor`) and **0 against every other
listing**. Those negatives are an *assumption*, not an assessment — nobody judged whether a
UAV-focused student would also suit the IoT lab. A retriever that finds a genuinely good
cross-listing match is penalised for it, so absolute metrics are biased downward.
`prototype`'s hand-graded labels do not have this problem.

**2. Structural degeneracy.** Because relevance is concentrated on exactly one listing,
every relevant set has size 1, so:
- Recall@k is binary — 1.0 if the target is in Top-k, else 0.0.
- Precision@k collapses to a near-constant 1/k.
- nDCG@k reduces to "did the target land at rank 1" (with partial credit for rank 2).

Only **nDCG@k** carries real signal; treat P/R as coarse sanity checks. To fix this properly,
each profile needs graded labels against *several* listings — which means hand-grading, as
`prototype` does.

### How the harness handles the null-information rows

Distractor profiles have no relevant listing, so every method scores them 0.0 *by
construction*. Those rows cannot discriminate between retrievers and would dilute the mean,
so they are:
- **excluded from all reported means** (marked `scorable = False`, still written to CSV), and
- re-evaluated as a **negative control** instead: the rank each method gave a listing the
  profile has no domain fit for. Lower is better; the useful comparison is the random
  baseline `(corpus_size + 1) / 2`.

Distractors legitimately belong in the benchmark as a false-positive probe — it was the
*scoring* of them that was wrong, not their inclusion.

## Result: the new benchmark contradicts the hypothesis

`synthetic_v2`, real MiniLM embeddings, k = 5:

```
match_type        Keyword (BM25)   Semantic (MiniLM)   Hybrid (0.6/0.4)
exact_keyword          1.0000            1.0000             1.0000
semantic_analogy       1.0000            0.9262             1.0000
noisy_edge             0.9000            0.9000             0.9262
```

**BM25 wins the `semantic_analogy` stratum outright**, which was supposed to be the case it
should lose. The mechanism is not that BM25 is secretly doing semantics — it is that the
stratum is contaminated at the *corpus* level:

> STU-009's circumlocutions (`home hypervisor`, `spaced copies`) match **no** building block
> of the `VT-LNX-03` listing, but `.7` — the substring `proxmox` inside `Proxmox` in the
> listing title — is *unique to that one listing in a 5-document corpus*. BM25's
> anti-discrimination guard floors its negative IDF at 0, so a discriminant of `.7` is still
> the single largest cosine contribution to that document. With only 5 listings, one stray
> substring hit is enough to win.

Empirically, *every* method put the target at rank 1 for 4 of 5 semantic profiles; the
sequence was 1-1-1-1-2 against BM25's 1-1-1-1-1. So this is not evidence that keywords beat
semantics — it is evidence that **a 5-document corpus cannot support the claim.** Rank-based
similarity saturates, IDFs are unreliable, and rare-substring effects dominate.

**Before this benchmark can test the hypothesis it needs** (a) substantially more listings
(≥50, ideally with near-duplicate or overlapping-skill pairs), (b) hand-graded labels across
multiple listings per profile, and (c) resolution of the `semantic_analogy` stratum's own
keyword leakage (see `data/synthetic_v2_dataset.py`).

## Performance

Measured with `scripts/benchmark.py`, which times every pipeline phase independently so
a total-speedup figure cannot hide a phase that regressed. The scale case is a synthetic
2000-query x 200-listing corpus, because the real datasets are far too small for
asymptotic behaviour to show up.

| Phase | Before | After | Speedup |
| --- | --- | --- | --- |
| **Metric evaluation** (2000 x 200) | 46,650 ms | **12.9 ms** | **3,613x** |
| Metric evaluation (20 queries) | 10.5 ms | 1.6 ms | 6.5x |
| Metric evaluation (4 queries) | 2.8 ms | 1.6 ms | 1.8x |
| **Query encoding**, real MiniLM, 512 queries | 5,081 ms | **1,474 ms** | **3.4x** |
| Rank + metrics, end to end (2000 x 200) | 48,041 ms | 1,008 ms | 47.6x |

### What was actually slow, and why

**`evaluate_metrics` was quadratic, and it dominated everything.** `Dataset.relevance_for()`
rescanned the entire `relevance_map` for every (profile, method) pair. At 2000 queries x 200
listings that is 6,000 pairs x 400,000 entries ≈ **2.4 billion dict lookups** — 97% of total
runtime. The fix is structural, not incremental: `Dataset` now materialises a dense
(queries x listings) `int8` grade matrix once in `__post_init__`, making every lookup an
O(1) array index. The ideal-DCG ordering is also precomputed once per dataset instead of
being re-sorted per row.

**Query encoding called the transformer once per query.** Each `encoder.encode([text])` call
pays fixed overhead (tokenizer setup, tensor allocation) regardless of input length, so
batching is nearly free performance. Query embeddings are now computed in batches and cosine
similarity for the whole batch is a single matmul:

```
sims = query_embeddings @ doc_embeddings.T        # (Q x D) @ (D x C)
```

Both sides are L2-normalised, so the inner product *is* cosine similarity.

**The tidy DataFrame was built per row.** `run_pipeline` accumulated one dict per
(query x method x rank) result — 60 rows for the prototype, but 60,000 at the scale case.
Results are now carried as aligned numpy arrays (`ResultTable`) and the long DataFrame is
constructed once, with vectorised column ops, only on export.

### Correctness was verified, not assumed

Metrics were checked against a brute-force reference implementation (a direct port of the
original scalar functions) across **20 randomised datasets with varying query/corpus sizes
and k, 843 scored rows: 0 mismatches**. Ranking order is unchanged — ties still break by
opportunity id, so runs remain byte-identical and reproducible.

A verification harness earned its keep here. The first vectorised version of
`evaluate_metrics` sliced the first *k columns* of the grade matrix instead of gathering
grades at the *retrieved* indices — so all three retrievers were scored against the same
listings and reported identical metrics. It looked like a plausible result. The stress test
caught it, and `precision_at_k_matrix` was also divided by `k_eff` instead of the requested
`k`, a silent discrepancy whenever `k` exceeds the corpus size. Both are fixed, with comments
at the call sites explaining why the index gather is load-bearing.

## Verification log (what has actually been checked)

| Claim | Method | Result |
| --- | --- | --- |
| Refactor preserves metrics | run prototype + synthetic_v2, diff summary CSVs | byte-identical |
| Vectorised metrics are correct | 20 random datasets, 843 rows vs brute-force reference | 0 mismatches |
| Methods are not degenerate | check BM25 vs Semantic rankings differ | 20/20 cases differ |
| `prototype` baseline (real MiniLM) | `mean_ndcg@k` | 0.7046 / 0.8846 / 0.8364 |
| `synthetic_v2` baseline (real MiniLM) | `mean_ndcg@k` | 0.9667 / 0.9421 / 0.9754 |
| Export schema unchanged | inspect `match_results.csv` header | 7 expected columns |
| Error handling | `--dataset nope` | clean argparse failure, exit 2 |

Known reproduction command for the headline number:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --scale 2000 --repeat 3
```

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
