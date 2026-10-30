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

The default dataset, `vt_furf`, holds 44 real research listings from Virginia Tech's 2026–27
First Year FURF program. Each listing keeps its source ID in the form `FURF-##` (for example,
`FURF-07`); this is the project number from the FURF source list, **not a rank or match score**.
The `FURF-##` numbers identify projects; IDs break ties when scores are equal. Every student
profile is synthetic, and no real student information is used.

## Quick start

You need **Git** and **Python 3.13** (`git --version` and `python --version` to check).
Clone the repository and enter its folder:

```sh
git clone https://github.com/VTURCS/IRDM.git
cd IRDM
```

Run the remaining commands from this folder. When creating the virtual environment,
use `python3.13` on macOS/Linux or `py -3.13` on Windows if `python` does not point to
Python 3.13.

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

Expected output: `40 passed`. The tests include every published number, so a pass also means
the results reproduce.

**Testing team:** the scenarios to run are in [TEST_SCENARIOS.md](TEST_SCENARIOS.md).

## Web interface

- **Search:** type a student's interests and press **Search**. By default, one list combines the top five
  results from each method, deduplicated by project and sorted by its highest rescaled score.
  Fixed score and similarity cutoffs filter weak matches in the combined list. Turn on **Compare methods**
  to see the three original rankings side by side. The page uses `vt_furf` (real VT listings) by default; open
  http://localhost:8000/?dataset=prototype for the hand-graded synthetic set, or any other
  folder name in `data/`.
- **Try a labelled profile:** paste one of the samples from the [testing guide](#testing-guide)
  below. When the text matches one of the dataset's own profiles exactly, each listing also
  shows its relevance grade (strong, partial or unsuitable) and comparison mode shows each method’s nDCG.
- **Defaults:** the search form uses five results and a hybrid keyword weight of 0.6.
  Use the command-line evaluation options below to experiment with these settings.
- **How each method scores:** the tables inside **Evaluation** give mean nDCG, Precision and Recall over
  the whole dataset, plus a breakdown by profile type for datasets that tag their profiles.
  These benchmarks measure the dataset's labelled profiles, not the current search.

In the single list, each entry shows its method beside its score. Scores are rescaled per
method and query using `(score - minimum) / (maximum - minimum)` across all listings, before
taking the top five. A method with equal scores for every listing gets zeros. These values
show relative position within a method; they are not calibrated probabilities or evidence
that the best result is relevant. Each project appears once using its highest score; the label names the method that supplied it
(or all tied methods). A project must meet the fixed minimum score of 0.5 and minimum semantic
similarity of 0.30. Equal-scoring projects are ordered by listing ID. Comparison mode keeps the original scores
and all three lists; it does not apply the combined-list cutoffs. Because each method’s best nonconstant score rescales
to 1, the combined list also checks the original
semantic cosine similarity for that project, before rescaling. Every candidate, including
Keyword and Hybrid results, must pass both cutoffs. Similarity is shown beside each result.
The 0.30 cutoff is a heuristic, not a validated relevance boundary. It can remove
useful keyword matches where the semantic model performs poorly; use comparison mode to
inspect candidates. The cutoffs do not change benchmark evaluation.

## Testing guide

After completing the quick start, open **http://localhost:8000/**. Copy a sample below
into **Student interests**, then click **Search**. Repeat with another sample and compare
the results. Enable **Compare methods** to follow the column-by-column examples below. Each result includes its
`FURF-##` source project number; those numbers identify postings and do not indicate rank.

### Sample student interests

These are exact profiles from `data/vt_furf/profiles.csv`, matched against the real VT
listings. The observations below are for five results and keyword weight 0.6. Every dataset's
profiles live in its own `data/<dataset>/profiles.csv`, so you can copy queries for any dataset;
the `prototype` samples are in [TEST_SCENARIOS.md](TEST_SCENARIOS.md) (scenario 1).

**S03 — Environmental Science** (uses the posting's own words)

```text
Microplastic pollution in water samples and FTIR analysis
```

**S08 — Animal and Poultry Sciences** (everyday words for a technical topic)

```text
Helping pets stay calm when their paws are handled
```

**S12 — English** (everyday words for a technical topic)

```text
How the way people talk changes from one part of the state to another
```

**S22 — Aerospace Engineering** (a long, mixed paragraph)

```text
I'm a first-year aerospace engineering student. I've been building and launching model rockets since middle school and I took AP Physics last year. I'd love to work on real hardware that goes to space, or at least on the design process behind it. I can do some basic programming and I want to get better at it.
```

**S24 — Theatre Arts** (reviewed as a partial match to one listing)

```text
I want to study classical ballet choreography
```

### What to look for

| Sample | Current behavior |
|---|---|
| S03 — Microplastics | All three methods rank the **Watamu Marine National Park microplastics** study first (strong). When a student uses the posting's own words, every method works. |
| S08 — Calm pets | Semantic ranks **Nailed it: Exploring Effective Training for Nail Trims Using Scratch Boards** first (strong). Keyword ranks **Engineering workforce development in the era of emerging technologies** first and Hybrid ranks **Optimistic chickens** first, both unsuitable. No word in the request appears in the posting, so keyword search has nothing to match, and the hybrid inherits much of that failure. |
| S12 — Dialects | Semantic ranks the **Southwest Virginia Language (SWVAL) Project** first (strong). Keyword and Hybrid rank a cognitive-psychology study first (unsuitable). |
| S22 — Space hardware | The reverse: Keyword and Hybrid rank **VT Cube Satellite: Hack-a-Sat** first (strong), on the words "design", "hardware" and "space". Semantic ranks the engineering-workforce project first (unsuitable). Semantic is not always better. |
| S24 — Ballet | Human review changed FURF-15 to a partial match. S24 is now included in the means. Its original `nomatch` profile-type tag is retained as provenance, not a statement that all current grades are zero. |

Pasting a sample exactly shows its reference grades; comparison mode also shows per-query nDCG.
For `vt_furf`, the AI-drafted grades now have one human review per pair. Independent rater
agreement remains unmeasured. The `prototype` grades were written by a person.

Also try your own queries:

- `I want to work with bees and insects outdoors`: all three methods rank the pollinator
  conservation project first.
- `I like biology`: no listing contains the word "biology", so every keyword score is 0 and the
  Keyword column is just listing-id order. Semantic ranks **Biomechanics of animals** first.

The app always returns up to five listings, so appearing in the results does not by itself
mean a listing is relevant. Similarity scores are not confidence percentages, and scores
from different methods are not directly comparable.

The collapsed **Evaluation** section holds the aggregate benchmark tables. Those tables
score the dataset's labelled profiles, so changing your search text will not change them.

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
| `prototype` | 8 × 4 | Short, vague profiles. Every pair graded by hand |
| `heldout` | 30 × 24 | Profiles tagged `exact`, `paraphrase`, `vague`, `token`, `long` or `nomatch`. Grades fixed before any method was run on it |
| `main_demo` | 12 × 5 | Small demo set |
| `synthetic_v2` | 5 × 20 | Longer profiles. Grades derived from each profile's target listing (one relevant listing each), so read with caution |
| `vt_furf` (default) | 44 × 25 | **Real VT listings** from the 2026–27 First Year FURF projects list, copied word for word, with synthetic profiles. AI-drafted grades have one human review per pair across two disjoint halves; agreement has not been measured. See the [FURF data provenance](#furf-data-provenance) section below |

To use a dataset other than the default in the web interface, add it to the URL, for example
http://localhost:8000/?dataset=vt_furf. Searches keep it.

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

## FURF data provenance

The default `vt_furf` dataset contains 44 listings from Virginia Tech's **2026–2027 First
Year Fralin Undergraduate Research Fellowship (FURF) Projects List**, alongside 25 fictional
student profiles. No real student data is included.

- `opportunities.csv`: project titles and descriptions are copied from the source list. Skills
  are normalized from its experience and skills section; eligibility and college are metadata.
  The source's research goals and mentoring style are not included in the searchable text.
- `profiles.csv`: 25 synthetic first-year student profiles.
- `relevance.csv`: 1,100 AI-drafted grades reviewed in two halves by Vaidehi and Sandilya.
  There is one human review per pair, not two independent ratings. See [review provenance](docs/HUMAN_REVIEW.md).

**Listing IDs:** `FURF-##` is the project's ID from the original source list—for example,
`FURF-07`. It identifies the posting; it is not a rank or match score. The source list has no
project number 34. IDs break ties when scores are equal.

The draft grades use 0 for unsuitable, 1 for partial and 2 for strong. To review them, compare
each profile in `profiles.csv` against every listing in `opportunities.csv`, then check all
corresponding rows in `relevance.csv`. Confirm both that every non-zero grade is justified and
that no zero should be relevant. Ideally, two reviewers should grade independently and compare.
After editing the grades, run `python -m pytest` to check that all profile/listing pairs are
present and valid.

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

The reviewed `vt_furf` results use 24 scorable profiles (only S25 is excluded):

| Method | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword | 0.4852 | 0.3083 | 0.4608 |
| Semantic | **0.8079** | **0.4417** | **0.7192** |
| Hybrid (0.6/0.4) | 0.6228 | 0.3917 | 0.6326 |

These values are pinned in tests and saved in `results/vt_furf/`. Reviews cover separate
halves of the labels; they do not establish inter-rater agreement. See [review report](docs/HUMAN_REVIEW.md).

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
| `benchmark.py`, `WEEK5.md`, `tests/test_benchmark.py` | The Week 5 standalone benchmark (weighted reciprocal-rank fusion, scored at 10), its report and labeling-rule proposal, and its tests. It reads `opportunities.csv`, `profiles.csv` and `relevance.csv` from the repository root (header-only templates) or `--data-dir` |

## Known limitations

- **Mostly small, synthetic data:** four datasets have 5–30 listings written to resemble VT postings. `vt_furf` has 44 real listings, with one human review per AI-drafted grade and no measured rater agreement.
- **k = 5, not 10:** with 8 listings, a top 10 would be the whole corpus.
- **One human review per pair:** two people reviewed separate halves of `vt_furf`; no independent overlapping ratings or agreement measure are available.
- **No "no match" answer:** each method always returns k listings. When a query shares no words with any listing, BM25 returns them in id order.

The full list is in [OPEN_ISSUES.md](OPEN_ISSUES.md). The presenter runbook is
[docs/PRESENTATION.md](docs/PRESENTATION.md), and the method design is in
[docs/hybrid_design.md](docs/hybrid_design.md).
