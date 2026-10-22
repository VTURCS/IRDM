# Test Scenarios: IRDM Prototype v1

This document gives five scenarios for the testing team. Each is a realistic situation built on
synthetic data, with the steps to follow and the output to expect. All five take about 30
minutes.

**Before you start:**
Follow the [README quick start](README.md#quick-start), leave the app running and open
**http://localhost:8000**.

**How the page works.** The only control is the search box. You choose the dataset, the
number of results (`k`) and the hybrid keyword weight through the URL instead:

```
http://localhost:8000/?dataset=heldout&k=5&weight=0.4
```

A search keeps whichever of those are already in the URL, so you can set them once and then
search freely. The aggregate tables live in the collapsed **Evaluation** section at the
bottom of the page.

For commands marked *second terminal*, open a new terminal in the repository folder and
activate the same virtual environment as described in the quick start. Keep the app running
in the first terminal.

| # | Scenario | Checks that… | Time |
|---|---|---|---|
| 1 | Students with vague interests | the three methods run, and the UI shows rankings, grades and scores | 5 min |
| 2 | Paraphrased and vague requests on unseen data | results hold on a larger held-out set, broken down by request type | 5 min |
| 3 | A coordinator adds their own listings | a new dataset folder works, and bad data is reported clearly | 10 min |
| 4 | Tuning the hybrid | the keyword weight changes the hybrid as expected | 5 min |
| 5 | Automated checks | the test suite passes and the CLI reproduces the published numbers | 5 min |

---

## Scenario 1: Students with vague interests

**Situation.** Four students describe themselves in a sentence or two, the way they might in an
email to an advisor. The system has 8 research listings and must rank them for each student.

**Steps**
1. Expand **Evaluation** at the bottom of the page and read **How each method scores on prototype**.
2. Paste this into **Student interests** and press **Match** (profile **P1**):
   `I like machine learning and I know some Python. I want to work on AI stuff.`
3. Repeat with **P2** and **P4**, copied from the
   [README testing guide](README.md#sample-student-interests):
   - P2: `Biology major interested in data and health. I don't code very much.`
   - P4: `I am an HCI and design person, I like studying people and technology.`
4. Type your own description (for example, a few words about your own interests) into
   **Student interests** and press **Match**.

Paste each query exactly, including punctuation. An exact match to one of the dataset's
profiles is what makes the grade badges and the per-query nDCG appear.

**Expected**
- At step 1, the evaluation table shows:

  | Method | nDCG@5 | Precision@5 | Recall@5 |
  |---|---|---|---|
  | Keyword (BM25) | 0.7046 | 0.5500 | 0.7417 |
  | **Semantic (all-MiniLM-L6-v2)** | **0.8846** | **0.6500** | **0.8875** |
  | Hybrid (0.6 KW / 0.4 SEM) | 0.8364 | 0.6500 | 0.8875 |

- At steps 2 and 3, three columns of 5 listings appear, each listing with a grade badge:

  | Profile | Method | nDCG@5 | Top listing (grade) |
  |---|---|---|---|
  | P1 | Keyword | 0.770 | Smart Grid Load Forecasting with Time-Series ML (partial) |
  | P1 | Semantic | 0.844 | LLM-Based Tutoring Chatbot for CS1 Courses (strong) |
  | P1 | Hybrid | 0.788 | Smart Grid Load Forecasting with Time-Series ML (partial) |
  | P2 | Keyword | 0.673 | Human-Robot Interaction for Assistive Robotics (unsuitable) |
  | P2 | Semantic | 0.805 | Usable Security: Evaluating Password Manager UX (partial) |
  | P4 | Keyword | 0.579 | Thermal Management for Electric Vehicle Battery Packs (unsuitable) |
  | P4 | Semantic | 1.000 | Usable Security: Evaluating Password Manager UX (strong) |

- At step 4, three ranked lists appear without grade badges or nDCG, because a free-text query
  has no grades.

**Pass if** the numbers match, every profile shows three ranked lists with grades, and your own
query returns results.

**Worth noting.** For a vague query, the keyword method's top pick can be a listing graded
unsuitable. It picks robotics for a biology student (P2) and battery cooling for an HCI student
(P4); semantic avoids both.

---

## Scenario 2: Paraphrased and vague requests on unseen data

**Situation.** This is a larger set: 30 listings and 24 student requests, written before any
method was run on them. Each request is tagged with a type:

| Type | Example request |
|---|---|
| `exact` | "Stream water pollution and nitrate sampling" (uses the listing's own words) |
| `paraphrase` | "Machines that walk on legs" (for a legged-robot locomotion listing) |
| `vague` | "I like being outside and want to help nature" |
| `token` | "ML for MRI" (short technical terms) |
| `long` | A paragraph-length self-description |
| `nomatch` | "Sculpture and ceramic art" (no listing fits) |

**Steps**
1. Open **http://localhost:8000/?dataset=heldout** to switch datasets, then expand **Evaluation**.
2. Paste `Machines that walk on legs` into **Student interests** and press **Match** (profile
   **Q02**), then compare the three columns.
3. *Second terminal:* `python evaluate.py heldout`

Pressing **Match** keeps you on `heldout`, because the form submits the dataset with your query.

**Expected**
- The evaluation table, over 22 scorable profiles (the 2 `nomatch` profiles have no correct answer and are left out):

  | Method | nDCG@5 | Precision@5 | Recall@5 |
  |---|---|---|---|
  | Keyword (BM25) | 0.5820 | 0.2364 | 0.5455 |
  | Semantic | 0.8649 | 0.3909 | 0.8583 |
  | Hybrid | 0.8320 | 0.3818 | 0.8356 |

- The **nDCG@5 by profile type** table:

  | Type | Keyword | Semantic | Hybrid |
  |---|---|---|---|
  | exact | 0.9524 | 0.9732 | 0.9800 |
  | long | 0.9832 | 0.8790 | 0.9832 |
  | paraphrase | 0.4710 | 0.8914 | 0.8157 |
  | token | 0.9211 | 0.9718 | 0.9718 |
  | vague | 0.2015 | 0.6635 | 0.6289 |

- Step 3 prints the same tables and names the 2 excluded profiles (Q10, Q23). It writes three
  CSVs to `outputs/heldout/` on your machine.

**Pass if** the UI and the CLI show the same numbers, and `outputs/heldout/` contains
`match_results.csv`, `evaluation_per_profile.csv` and `evaluation_summary.csv`.

**Worth noting.** Keyword matching holds up when students use the listing's own words (`exact`,
`long`). It collapses when they don't: 0.47 for `paraphrase` and 0.20 for `vague`. That gap is
the central question of the project.

---

## Scenario 3: A coordinator adds their own listings

**Situation.** A lab coordinator wants to try the system on their own postings using a
spreadsheet, without touching any Python. They forget to grade a new listing, fix that, and then
make a typo.

**Step 1. Copy the example dataset.**
- Windows (PowerShell): `Copy-Item -Recurse data/main_demo data/my_listings`
- macOS/Linux: `cp -r data/main_demo data/my_listings`

Open **http://localhost:8000/?dataset=my_listings** and expand **Evaluation**. Expect 12
listings, 5 profiles and:

| Method | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword (BM25) | 0.8454 | 0.4000 | 0.9167 |
| Semantic | 1.0000 | 0.4500 | 1.0000 |
| Hybrid | 1.0000 | 0.4500 | 1.0000 |

**Step 2. Add a listing but forget to grade it.** Open `data/my_listings/opportunities.csv` in a
text editor or Excel, add this row at the end, and save:
```
o13,Drone delivery logistics,Plan delivery routes for campus drones using Python and optimization.
```
Refresh the page. Expected: a red message instead of results, naming the file (the path will
be your own repository folder):
```
...\data\my_listings\relevance.csv: 5 (profile, opportunity) pairs have no grade, starting with ('p01', 'o13'). Every pair needs a grade, zeros included.
```

**Step 3. Grade it.** Add these rows to the end of `data/my_listings/relevance.csv`. Profile
p01 ("Python computer vision for autonomous robots") is a partial match; the others are
unrelated.
```
p01,o13,1
p02,o13,0
p03,o13,0
p04,o13,0
p05,o13,0
```
Refresh. Expected: 13 listings. nDCG and recall are unchanged, and precision rises to **0.45 /
0.50 / 0.50**, because the new listing is a partial match that lands in p01's top 5.

**Step 4. Make a typo.** In `relevance.csv`, change the row `p01,o01,2` to `p01,o01,3`, then
refresh. Expected:
```
...\data\my_listings\relevance.csv:2: grade must be 0, 1 or 2 (row: p01,o01,3)
```

**Pass if** two things hold:
- Steps 1 and 3 show the numbers above.
- Steps 2 and 4 show a one-line message naming the file and the line, and the page still works: you can open another dataset.

Delete `data/my_listings/` when you're done.

**Worth noting.** On **http://localhost:8000/?dataset=main_demo**, search for
`Quantum cryptography`. No listing contains those words, so every keyword score is 0 and BM25
returns listings in id order (o01, o02, o03…). The system never answers "no good match"; it
always returns k listings.

---

## Scenario 4: Tuning the hybrid

**Situation.** The hybrid mixes keyword and semantic scores. A researcher wants to see how the
mix affects quality.

**Steps**
1. Open **http://localhost:8000/?weight=0**, expand **Evaluation**, and read the Hybrid row.
2. Repeat with `?weight=0.4` and `?weight=1`.
3. *Second terminal:* `python evaluate.py prototype --sweep`

**Expected**

| Keyword weight | Hybrid nDCG@5 | Note |
|---|---|---|
| 0 | 0.8846 | Same as the Semantic row |
| 0.4 | **0.9030** | Best: beats both pure methods |
| 0.6 (default) | 0.8364 | |
| 1 | 0.7046 | Same as the Keyword row |

Step 3 prints the full curve: 0.8846, 0.9011, 0.9004, 0.9030, 0.8364, 0.8239, 0.7046 for
weights 0.0, 0.2, 0.3, 0.4, 0.6, 0.8 and 1.0.

**Pass if** the endpoints equal the pure methods and the values match.

**Worth noting.** A light keyword weight helps, and a heavy one hurts, so the 0.6 default gives
keywords too much weight. With only 4 profiles, a 0.02 difference is a single rank change, so
the shape of the curve matters more than its exact peak.

---

## Scenario 5: Automated checks

**Situation.** A reviewer wants proof that the code is working and that the published numbers
come from it.

**Steps** (*second terminal*)
```
python -m pytest -v
```

**Expected:** `20 passed`. The tests cover:
- the published results on all four datasets, and the weight sweep
- the nDCG calculation, worked by hand
- ties and the no-overlap case
- rejection of five kinds of bad CSV
- the CLI
- the web interface, including a broken dataset folder

**Pass if** all 20 pass. If any fail, please copy the failure summary into your report.

---

## Recording results

For the test report, one row per scenario is enough:

| # | Ran? | Output matched? | Problems encountered | Recommendations |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

Please also note your OS, your Python version (`python --version`) and how long the initial setup
took. If anything fails, the full terminal output is the most useful thing to include.
