# Architecture, files, data layout, datasets, flags

> Moved from the original README (unchanged text).

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

