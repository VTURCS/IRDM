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
- **Precision@k:** grade >= 1 counts as relevant, divided by k.
- **Recall@k:** grade >= 1 relevant listings found in top k, over all relevant listings.
- **k = 5 for Prototype v1** because the corpus has only 8 listings. @10 is deferred
  until the corpus is larger (see OPEN_ISSUES.md). Means cover scorable profiles only.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe research_matching_pipeline.py                        # prototype
.\.venv\Scripts\python.exe research_matching_pipeline.py --dataset synthetic_v2
```
First run downloads MiniLM (~90 MB). Python 3.10-3.12 is the safest interpreter.

## Test
Smoke test with no model download (`--offline-embeddings`):
```powershell
.\.venv\Scripts\python.exe research_matching_pipeline.py --offline-embeddings --output-dir outputs/smoke
```
Expect three CSVs in `outputs/smoke/prototype/`. **Caution:** in this mode the "semantic"
condition is a hashed bag-of-words proxy, not MiniLM, so never report it as a MiniLM result.

## More
Results: `docs/RESULTS.md` | Limitations: `OPEN_ISSUES.md`
Design notes, verification log, performance work: `docs/`
