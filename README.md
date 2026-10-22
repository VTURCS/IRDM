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

You need **Python 3.13** (`python --version` to check). Run the following from the
repository folder. On macOS/Linux, use `python3.13` for the first command if needed;
on Windows, you can use `py -3.13`.

1. Create a virtual environment:

   ```sh
   python -m venv .venv
   ```

2. Activate it in each terminal you use:

   | Shell | Command |
   |---|---|
   | macOS/Linux (bash/zsh) | `source .venv/bin/activate` |
   | Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
   | Windows Command Prompt | `.venv\Scripts\activate.bat` |

3. On Linux/Windows, install CPU-only PyTorch first to avoid downloading GPU libraries
   (skip this command on macOS; the next command installs its native PyTorch package):

   ```sh
   python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
   ```

4. Install the remaining dependencies and start the app:

   ```sh
   python -m pip install -r requirements.txt
   python app.py
   ```

Open **http://localhost:8000**. The first launch downloads the model (~90 MB) and caches it
locally. Stop the app with `Ctrl+C`. For later runs, activate the environment and run
`python app.py`; you only need to install dependencies once.

To check that everything works, run this in an activated terminal:

```sh
python -m pytest
```

Expected output: `19 passed`. The tests include every published number, so a pass also means
the results reproduce.

**Testing team:** the scenarios to run are in [TEST_SCENARIOS.md](TEST_SCENARIOS.md).

## Web interface

- **Search:** type a student's interests and press **Match**. Each method's top five
  listings appear side by side. The default dataset is `irdm.DEFAULT_DATASET`.
- **Try a profile:** expand **Evaluation** and pick one of the dataset's profiles. Each listing then
  shows its relevance grade (strong, partial or unsuitable), and each method shows its nDCG for
  that profile.
- **Defaults:** the search form uses five results and a hybrid keyword weight of 0.6.
  Use the command-line evaluation options below to experiment with these settings.
- **How each method scores:** the tables inside **Evaluation** give mean nDCG, Precision and Recall over
  the whole dataset, plus a breakdown by profile type for datasets that tag their profiles.
  These benchmarks measure predefined profiles, not the current search.

## Testing guide

After completing the quick start, open **http://localhost:8000/**. Copy a sample below
into **Student interests**, then click **Match**. Repeat with another sample and compare
the order and scores in the Keyword, Semantic and Hybrid columns.

### Sample student interests

These are the exact profiles from `data/prototype/profiles.csv`. The observations below
apply to the current synthetic `prototype` dataset with five results and keyword weight
0.6. Update these examples and expected results when the real dataset replaces it.

**P1 — Computer Science**

```text
I like machine learning and I know some Python. I want to work on AI stuff.
```

**P2 — Biology**

```text
Biology major interested in data and health. I don't code very much.
```

**P3 — Mechanical Engineering**

```text
I have built robots before, I like hardware and control systems.
```

**P4 — Psychology**

```text
I am an HCI and design person, I like studying people and technology.
```

### What to look for

| Sample | Current behavior |
|---|---|
| P1 — AI and Python | Semantic ranks **LLM-Based Tutoring Chatbot for CS1 Courses** first; Keyword and Hybrid rank **Smart Grid Load Forecasting with Time-Series ML** first. |
| P2 — Biology and health | Semantic currently ranks **Usable Security: Evaluating Password Manager UX** first. This is a useful failure case to inspect, not an example of a strong biology match. |
| P3 — Robotics | All three methods rank **Human-Robot Interaction for Assistive Robotics** first. |
| P4 — HCI and design | Semantic and Hybrid rank **Usable Security: Evaluating Password Manager UX** first; Keyword ranks **Thermal Management for Electric Vehicle Battery Packs** first. |

Pasting a sample exactly enables its predefined relevance labels and per-query nDCG.
Those labels are human-assigned evaluation data, not predictions from the model.

Also try your own queries:

- `I like biology`: Semantic should rank **Genomic Data Pipelines for Precision Agriculture**
  first. Keyword scores are all zero, so its displayed order is just the listing ID order.
- `I want to study medieval French poetry.`: the app still returns five listings even though
  the dataset has no suitable opportunity. This demonstrates the current lack of a relevance cutoff.

The app always returns up to five listings, so appearing in the results does not by itself
mean a listing is relevant. Similarity scores are not confidence percentages, and scores
from different methods are not directly comparable.

The collapsed **Evaluation** section contains the sample-profile picker and aggregate
benchmark tables. Those tables evaluate the predefined profiles, so changing only your
search text will not change them.

For automated checks, run `python -m pytest` in the activated environment. More detailed
scenarios are in [TEST_SCENARIOS.md](TEST_SCENARIOS.md).

## Command-line evaluation

Run these in an activated terminal from the repository folder:

```
python evaluate.py heldout
python evaluate.py prototype --sweep
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
good starting point to copy. Run `python evaluate.py <folder name>` to evaluate it. Set `DEFAULT_DATASET` in `irdm.py`
and restart the app to use it as the default in the web interface.

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
- The model is pinned to a fixed revision and cached locally after download.
- Package versions are pinned in `requirements.txt`.
- Ties are broken by listing id.
- The tests fail if any published number changes.

Full results and caveats are in [docs/RESULTS.md](docs/RESULTS.md), and the CSVs behind them are
committed in [results/](results/).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Address already in use` | Run `python -m flask --app app run --port 8080`, then open http://localhost:8080. |
| `ModuleNotFoundError` | Activate `.venv` in this terminal and run `python -m pip install -r requirements.txt`. |
| The app still uses the old dataset | Set `DEFAULT_DATASET` in `irdm.py` to the new folder name under `data/`, restart the app, and open `/` without old query parameters. |

## Project layout

| Path | What it is |
|---|---|
| `irdm.py` | Dataset loading, the three retrieval methods and the metrics |
| `app.py`, `templates/`, `static/` | The web interface |
| `evaluate.py` | Command-line evaluation |
| `tests/test_irdm.py` | Tests, including every published number |
| `data/` | Datasets, one folder each |
| `results/` | Committed CSVs behind docs/RESULTS.md |

## Known limitations

- **Small, synthetic data:** 5–30 listings written to resemble VT postings.
- **k = 5, not 10:** with 8 listings, a top 10 would be the whole corpus.
- **One rater:** one person wrote the grades, and there is no agreement check yet.
- **No "no match" answer:** each method always returns k listings. When a query shares no words with any listing, BM25 returns them in id order.

The full list is in [OPEN_ISSUES.md](OPEN_ISSUES.md). The presenter runbook is
[docs/PRESENTATION.md](docs/PRESENTATION.md), and the method design is in
[docs/hybrid_design.md](docs/hybrid_design.md).
