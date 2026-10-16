# VTURCS Team 1: IRDM Prototype v1

**Research question:** How do BM25, semantic (MiniLM), and weighted hybrid retrieval
compare when matching short student-interest profiles to VT research opportunities?

## Conditions (identical corpus, identical verbatim queries)
| Condition | Method |
|---|---|
| Keyword | BM25Okapi over `title + description + skills` (lowercase tokens, stop-words removed) |
| Semantic | `all-MiniLM-L6-v2` embeddings at pinned revision `1110a24`, cosine similarity |
| Hybrid | `0.6 * minmax(BM25) + 0.4 * minmax(cosine)`, normalized per query |

Held constant: no metadata filtering, no query expansion, eligibility text excluded
from the searchable body. Ties break on opportunity ID, so runs are reproducible.

## Inputs
- **Opportunities:** synthetic VT-style listings (`id, title, description`, plus metadata).
- **Profiles:** short, vague student-interest queries. Fictional, non-participant data.
- **Relevance labels:** 0 = unsuitable, 1 = partial match, 2 = strong match, for every
  (profile, listing) pair. The dataset loader rejects incomplete label sets.
- Datasets (`--dataset NAME`):
  - `prototype`: 8 listings, 4 hand-graded profiles (default).
  - `synthetic_v2`: 5 listings, 20 profiles, derived labels (see docs/).
  - `main_demo`: 12 listings, 5 profiles, the `main` branch demo carried over verbatim.
  - `heldout`: 30 listings, 24 profiles tagged by type (exact, paraphrase, vague, token,
    long, nomatch). Written for the cross-branch comparison, with labels fixed before
    either system was run on it. One rater, illustrative labels.

### CSV input format (use this for real VT listings)
Any folder holding these three files runs with `--data-dir PATH`. No Python needed.
`main_demo` and `heldout` live in `data/csv/` in this format.

| File | Required columns | Optional columns |
|---|---|---|
| `opportunities.csv` | `id, title, description` | `skills` (separated by `;`), `eligibility`, `department` |
| `profiles.csv` | `id, query` | `major, year, match_type` (enables `--match-type`) |
| `relevance.csv` | `profile_id, opportunity_id, grade` (0/1/2) | |

`relevance.csv` must grade every (profile, listing) pair, zeros included. Bad files stop
with a one-line error naming the file and line. Results go to `outputs/<folder name>/`.

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
.\.venv\Scripts\python.exe -m pytest -q          # 93 tests
.\.venv\Scripts\python.exe scripts\reproduce.py  # checks all reported values
```
`scripts/reproduce.py` is also **the command to run when presenting**: it runs `prototype`
and `synthetic_v2` through all three conditions, prints the results, sweeps the hybrid
weight, shows live timings, verifies every number against `docs/RESULTS.md`, and writes
`results/`. `pytest -m "not slow"` skips the tests needing the MiniLM model (88 tests,
no download).

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt  # exact tested versions, incl. pytest
.\.venv\Scripts\python.exe research_matching_pipeline.py                        # prototype
.\.venv\Scripts\python.exe research_matching_pipeline.py --dataset heldout
.\.venv\Scripts\python.exe research_matching_pipeline.py --data-dir path\to\folder  # your own CSVs
.\.venv\Scripts\python.exe research_matching_pipeline.py --bm25-weight 0.0        # weight sweep
```
`requirements-lock.txt` is the environment the tests and `docs/RESULTS.md` were verified in
(**Python 3.13**, Windows, CPU torch 2.14.1). `requirements.txt` + `requirements-dev.txt`
remain the unpinned alternative. The model is pinned to Hugging Face revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (`EMBEDDING_REVISION` in the pipeline), so an
upstream update cannot change results. The first run downloads MiniLM (~90 MB); later
runs are offline.

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
