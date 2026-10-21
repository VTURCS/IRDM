# IRDM: Research-Opportunity Matching

VTURCS Team 1 · Prototype v1

Students often describe their research interests in a sentence or two, and vaguely: "I like
machine learning and I know some Python." IRDM ranks Virginia Tech research listings for each
student in three ways. It then measures which ranking puts the right listings at the top,
against relevance grades written in advance.

| Method | How it ranks listings |
|---|---|
| Keyword (BM25) | Word overlap with each listing's title, description and skills |
| Semantic (MiniLM) | Similarity of meaning, using `all-MiniLM-L6-v2` sentence embeddings and cosine similarity |
| Hybrid | `0.6 × keyword + 0.4 × semantic`, each rescaled to 0–1 per query |

All data is synthetic. No real student information is used.

## Quick start

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/), or Docker Engine
with Compose. From the repository folder, run:

```
docker compose up --build
```

Then open **http://localhost:8000**. The first build downloads the dependencies and the model
and takes a few minutes; later starts take seconds. Stop the app with `Ctrl+C`.

To check that everything works:

```
docker compose run --rm irdm pytest
```

Expected output: `19 passed`. The tests include every published number, so a pass also means
the results reproduce.

**Testing team:** the scenarios to run are in [TEST_SCENARIOS.md](TEST_SCENARIOS.md).

## Web interface

- **Search:** pick a dataset, type a student's interests and press **Match**. Each method's
  top-k listings appear side by side.
- **Try a profile:** pick one of the dataset's profiles from the list instead. Each listing then
  shows its relevance grade (strong, partial or unsuitable), and each method shows its nDCG for
  that profile.
- **Settings:** **Top k** and **Keyword weight** (the hybrid's BM25 share, from 0 to 1) apply to
  the rankings and to the evaluation table.
- **How each method scores:** the table at the bottom gives mean nDCG, Precision and Recall over
  the whole dataset, plus a breakdown by profile type for datasets that tag their profiles.

## Command-line evaluation

```
docker compose run --rm irdm python evaluate.py heldout
docker compose run --rm irdm python evaluate.py prototype --sweep
```

This prints mean scores per method. Depending on the dataset it also prints a breakdown by
profile type, a negative-control check for distractor profiles, and (with `--sweep`) hybrid
nDCG across keyword weights from 0.0 to 1.0. It writes three CSVs to `outputs/<dataset>/`:

| File | Contents |
|---|---|
| `match_results.csv` | Every ranked list: `profile_id, method, opportunity_id, rank, score, title, description` |
| `evaluation_per_profile.csv` | Scores per profile and method, plus `n_relevant, n_strong, scorable, target_rank` |
| `evaluation_summary.csv` | Mean scores per method |

Options:
- `--top-k N`: ranked-list length and metric cut-off (default 5)
- `--bm25-weight W`: the hybrid's keyword weight, from 0 to 1 (default 0.6)
- `--sweep`: also report hybrid nDCG across keyword weights
- `--output-dir DIR`: where to write the CSVs

## Datasets

| Dataset | Listings × profiles | Notes |
|---|---|---|
| `prototype` (default) | 8 × 4 | Short, vague profiles. Every pair graded by hand |
| `heldout` | 30 × 24 | Profiles tagged `exact`, `paraphrase`, `vague`, `token`, `long` or `nomatch`. Grades fixed before any method was run on it |
| `main_demo` | 12 × 5 | Small demo set |
| `synthetic_v2` | 5 × 20 | Longer profiles. Grades derived from each profile's target listing (one relevant listing each), so read with caution |

### Adding your own

To add a dataset, create a folder in `data/` holding three CSV files. `data/main_demo/` is a
good starting point to copy. The new folder appears in the web interface's dataset list when you
refresh the page, and `evaluate.py <folder name>` runs on it.

| File | Required columns | Optional columns |
|---|---|---|
| `opportunities.csv` | `id, title, description` | `skills` (separated by `;`), `eligibility`, `department` |
| `profiles.csv` | `id, query` | `major, year`, `match_type` (enables the by-type table), `target_id` |
| `relevance.csv` | `profile_id, opportunity_id, grade` | |

Grades are 0 for unsuitable, 1 for a partial match and 2 for a strong match. `relevance.csv` must
grade **every** (profile, listing) pair, zeros included. Only title, description and skills are
searched; the other columns are metadata. If a file has a problem, both the web interface and
the CLI show a one-line message naming the file and line.

## Metrics

- **nDCG@k** (the main metric, from 0 to 1): rewards putting strong matches near the top. 1.0 means the ideal order.
- **Precision@k:** the share of the top k graded 1 or 2. It always divides by k.
- **Recall@k:** the share of all listings graded 1 or 2 that made the top k.

A profile with no relevant listing, such as a `nomatch` profile, scores 0 under every method. It
is therefore left out of the means.

## Results

On `prototype` with k = 5:

| Method | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword (BM25) | 0.7046 | 0.55 | 0.7417 |
| Semantic (all-MiniLM-L6-v2) | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 KW / 0.4 SEM) | 0.8364 | 0.65 | 0.8875 |

The keyword-weight sweep peaks at 0.4 (nDCG 0.9030), so the 0.6 default gives keywords too much
weight.

Results are deterministic, for four reasons:
- The model is pinned to a fixed revision and built into the image.
- Package versions are pinned in `requirements.txt`.
- Ties are broken by listing id.
- The tests fail if any published number changes.

Full results and caveats are in [docs/RESULTS.md](docs/RESULTS.md), and the CSVs behind them are
committed in [results/](results/).

## Without Docker

You need Python 3.13. Then run:

```
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                      # then open http://localhost:8000
pytest
```

The first run downloads the model (~90 MB).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it reports that it is running. |
| `port is already allocated` | Something else is using port 8000. In `docker-compose.yml`, change `"8000:8000"` to `"8080:8000"`, then open http://localhost:8080. |
| My dataset doesn't appear in the list | The folder must sit directly in `data/` and contain `opportunities.csv`. Refresh the page. |

## Project layout

| Path | What it is |
|---|---|
| `irdm.py` | Dataset loading, the three retrieval methods and the metrics |
| `app.py`, `templates/`, `static/` | The web interface |
| `evaluate.py` | Command-line evaluation |
| `tests/test_irdm.py` | Tests, including every published number |
| `data/` | Datasets, one folder each |
| `results/` | Committed CSVs behind docs/RESULTS.md |
| `Dockerfile`, `docker-compose.yml` | Container setup |

## Known limitations

- **Small, synthetic data:** 5–30 listings written to resemble VT postings.
- **k = 5, not 10:** with 8 listings, a top 10 would be the whole corpus.
- **One rater:** one person wrote the grades, and there is no agreement check yet.
- **No "no match" answer:** each method always returns k listings. When a query shares no words with any listing, BM25 returns them in id order.

The full list is in [OPEN_ISSUES.md](OPEN_ISSUES.md). The presenter runbook is
[docs/PRESENTATION.md](docs/PRESENTATION.md), and the method design is in
[docs/hybrid_design.md](docs/hybrid_design.md).
