# Performance work

> Moved from the original README (unchanged text).

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

