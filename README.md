# Team 1 retrieval benchmark

One script compares BM25, semantic search, and keyword-prioritized hybrid retrieval.

## Run

### Quick test with synthetic data

The `demo` folder contains 12 fictional opportunities, 2 queries, and complete illustrative labels. These are for checking the script, not research findings. After installing dependencies using the commands below, run:

```powershell
.\.venv\Scripts\python.exe benchmark.py --data-dir demo
```

Expect `demo/results.csv` with 60 data rows (2 profiles x 3 methods x 10 results), and `demo/metrics.csv` with 6 data rows. Each metric status should be `complete`, with numeric metrics between 0 and 1. Robot navigation should be a strong result for p01, and Stream water quality for p02. Exact rankings may vary between methods; a perfect score is not required for a successful test.

To test incomplete judgments, back up `demo/relevance.csv`, remove the `p01,o01,2` row, and rerun. p01 nDCG must be blank with a missing-label status. Restore the row afterward.

### Setup and study run

From this folder, create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe benchmark.py
```

Fill `opportunities.csv` and `profiles.csv` first; the supplied files are empty templates, not research data. Quote CSV fields containing commas or newlines. Extra columns such as source URL, eligibility, and skills are allowed but are not used for retrieval. Include any information you want searched in the description.

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`; embedding computation runs locally on CPU. The script embeds the title plus description and compares normalized embeddings by cosine similarity. BM25 uses lowercase word tokens. Hybrid uses weighted reciprocal-rank fusion: `0.6 / (60 + BM25 rank) + 0.4 / (60 + semantic rank)`. Ranks start at 1. All methods search the same corpus with no filters, expansion, or reranking. Scores are method-specific and should not be compared across methods.

## Judge and evaluate

`results.csv` contains up to 10 results per profile per method. Manually fill `relevance.csv` with one row per profile/opportunity pair: `0` unsuitable, `1` partial match, `2` strong match. Establish shared topic, eligibility, and skill rules before scoring. Do not infer ground truth from a method's score. Labels apply across methods.

Run again to produce `metrics.csv` and print method averages with the number of evaluated profiles. Precision treats grades 1 and 2 as relevant. It requires judgments for every returned result. nDCG uses gains `2^grade - 1` and requires labels for **every opportunity for that profile** to establish the ideal ranking. Missing labels remain missing, never implicit zeros. Profiles with no relevant opportunities have undefined nDCG, left blank. If the corpus has fewer than 10 listings, metrics use the returned count, recorded in `returned`.

Compare averages on the same profiles; incomplete judgments can give different evaluated subsets. A header-only relevance file produces flagged, blank metrics. Removing relevance.csv skips evaluation. Each successful run replaces results.csv and metrics.csv.

Use `--data-dir PATH` to read inputs and write outputs in another existing folder. Keep the model and hybrid constants fixed for the first experiment. Record dependency versions with `python -m pip freeze` for the study's reproducibility. Real public VT listings and team-reviewed relevance labels are still needed.
