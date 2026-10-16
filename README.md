# VTURCS Team 1: IRDM Prototype v1

**Research question:** How do BM25, semantic (MiniLM), and weighted hybrid retrieval
compare when matching short student-interest profiles to VT research opportunities?

## Conditions (identical corpus, identical verbatim queries)
| Condition | Method |
|---|---|
| Keyword | BM25Okapi over `title + description` (lowercase tokens) |
| Semantic | `all-MiniLM-L6-v2` embeddings, cosine similarity |
| Hybrid | `0.6 * minmax(BM25) + 0.4 * minmax(cosine)`, normalized per query |

Held constant: no metadata filtering, no query expansion, eligibility text excluded
from the searchable body. Ties break on opportunity ID, so runs are reproducible.

## Inputs
- **Opportunities:** synthetic VT-style listings (`id, title, description`, plus metadata).
- **Profiles:** short, vague student-interest queries. Fictional, non-participant data.
- **Relevance labels:** 0 = unsuitable, 1 = partial match, 2 = strong match, for every
  (profile, listing) pair. The dataset loader rejects incomplete label sets.
- Datasets: `prototype` (8 listings, 4 hand-graded profiles; default) and
  `synthetic_v2` (5 listings, 20 profiles, derived labels; see docs/).

## Outputs (written to `outputs/<dataset>/`)
`match_results.csv` (ranked lists), `evaluation_per_profile.csv`, `evaluation_summary.csv`.

## Evaluation
- **nDCG@k** (primary): gain `2^grade - 1`, normalized by the ideal ranking.
- **Precision@k:** grade >= 1 counts as relevant, divided by the requested k.
- **Recall@k:** grade >= 1 relevant listings found in top k, over all relevant listings.
- **k = 5 for Prototype v1**: the requirement is @10, but Top-10 would be nearly the whole
  corpus (4 profiles x 8 listings), so @5 is reported and @10 is deferred until the corpus
  is expanded (see OPEN_ISSUES T1). Means cover scorable profiles only.

## Verify (do this before presenting)
```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 69 tests
.\.venv\Scripts\python.exe scripts\reproduce.py  # checks all reported values
```
`scripts/reproduce.py` re-runs both datasets, writes `results/`, and **fails if any number has
drifted** from `docs/RESULTS.md`. `pytest -m "not slow"` skips the tests needing the MiniLM
model (65 tests, no download).

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # pytest
.\.venv\Scripts\python.exe research_matching_pipeline.py                        # prototype
.\.venv\Scripts\python.exe research_matching_pipeline.py --dataset synthetic_v2
.\.venv\Scripts\python.exe research_matching_pipeline.py --bm25-weight 0.0        # weight sweep
```
Verified on **Python 3.13** with numpy 2.5.3, pandas 3.0.6, torch 2.14.1+cpu.
The first run downloads MiniLM (~90 MB); later runs are offline.

## Test
Smoke test with no model download (`--offline-embeddings`):
```powershell
.\.venv\Scripts\python.exe research_matching_pipeline.py --offline-embeddings --output-dir outputs/smoke
```
Expect three CSVs in `outputs/smoke/prototype/`. **Caution:** in this mode the "semantic"
condition is a hashed bag-of-words proxy, not MiniLM. It scores *below* BM25 (0.6818 vs
0.7046 nDCG@5), so an offline run is not comparable to the headline numbers. Check the
method label before quoting any value.

## More
**Presentation: `docs/PRESENTATION.md`** (runbook, demo script, cheat sheet)
Results: `docs/RESULTS.md` | Limitations: `OPEN_ISSUES.md` | Artifact: `results/`
Design notes, verification log, performance work: `docs/`
