# Presentation Runbook: Prototype v1

Everything needed to present this branch. Read the **Presenter cheat sheet** first;
it is the part to have memorised before you stand up.

---

## THE SINGLE COMMAND TO RUN

Once setup (section 1) is done, this one command does the whole presentation:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\reproduce.py
```

It runs both datasets through all three conditions, **prints** the results, sweeps
the hybrid weight, shows live timings, verifies every number against `docs/RESULTS.md`,
and writes the CSVs to `results/`. Exit code 0 means the numbers shown are current.

Add `--offline` only if there is no model cache (see section 6 - it changes the results).

## 1. One-command setup

```powershell
# From the repo root, in PowerShell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # pytest
```

First pipeline run downloads `all-MiniLM-L6-v2` (~90 MB) into the Hugging Face
cache. Do this **before** presenting, on the machine you will present from.

> If you are offline or on a fresh machine with no model cache, add
> `--offline-embeddings`. **It changes the results** (the semantic condition
> becomes a lexical proxy) — see §6.

## 2. Verify everything works (do this before the demo)

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q          # expect: 71 passed
.\.venv\Scripts\python.exe scripts\reproduce.py  # expect: "all reported values reproduced"
```

If both pass, the numbers you are about to present match `docs/RESULTS.md`
exactly. If either fails, **do not present the numbers** — they have drifted.

## 3. Demo sequence (≈4 minutes)

Rehearse this once. Each command either prints something striking or produces an artifact.

```powershell
# (a) The headline comparison: 3 conditions, one table
.\.venv\Scripts\python.exe research_matching_pipeline.py
```

```powershell
# (b) The ablation that explains the result: keyword weight 0.0 -> 1.0
.\\.venv\\Scripts\\python.exe scripts\\sweep.py
```

```powershell
# (c) The stratified view: BM25 wins on exact vocabulary, and the corpus is too small
.\.venv\Scripts\python.exe research_matching_pipeline.py --dataset synthetic_v2
```

```powershell
# (d) Performance: the phase table that shows where time actually goes
.\.venv\Scripts\python.exe scripts\benchmark.py --scale 2000
```

### What to point at

| Command | The one-sentence takeaway |
| --- | --- |
| (a) | Semantic leads (nDCG 0.885); the **default** hybrid is worse than pure semantic (0.836) - a 0.6 keyword weight is too strong. |
| (b) | The full report: all three conditions, per-profile detail, and the verification verdict. |
| (c) | The sweep peaks at 0.4, above pure semantic (0.903 vs 0.885) - so a *mild* keyword signal helps and a strong one hurts. The default is simply mistuned. |
| (d) | `evaluate_metrics` went from 46.7 s to 12.9 ms at scale. The refactor removed a quadratic, not just constant overhead. |

## 4. Presenter cheat sheet

**Research question.** How do keyword-only (BM25), semantic-only (MiniLM), and
keyword-prioritized hybrid retrieval compare when matching short, vague student
profiles to VT research opportunities?

**Conditions** (identical corpus, identical verbatim queries — no metadata
filtering, no query expansion, eligibility text excluded from the searchable body):

| Condition | Mechanism |
| --- | --- |
| Keyword | BM25Okapi with negative IDFs floored at 0 |
| Semantic | `all-MiniLM-L6-v2`, cosine similarity |
| Hybrid | `0.6 · minmax(BM25) + 0.4 · minmax(cosine)`, per-query normalisation |

**Headline numbers** (prototype: 8 listings, 4 hand-graded profiles, k = 5, real MiniLM):

| Condition | nDCG@5 | P@5 | R@5 |
| --- | --- | --- | --- |
| Keyword (BM25) | 0.7046 | 0.55 | 0.7417 |
| **Semantic (MiniLM)** | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 / 0.4) | 0.8364 | 0.65 | 0.8875 |

**The mechanism for why the default hybrid loses.** A listing that is a strong semantic
match but shares no vocabulary with a vague query gets `bm25_norm = 0`, so it keeps only its
0.4 share — while a listing that overlaps lexically gets `bm25_norm = 1` and outranks it.
Score-level fusion inherits BM25's brittleness to vocabulary mismatch by construction.

**The weight sweep corrects the naive reading.** The hybrid does *not* simply lose to
semantic — the default is mistuned:

| bm25_weight | hybrid nDCG@5 |
| --- | --- |
| 0.0 (pure semantic) | 0.8846 |
| 0.2 | 0.9011 |
| 0.3 | 0.9004 |
| **0.4** | **0.9030** |
| 0.6 (current default) | 0.8364 |
| 0.8 | 0.8239 |
| 1.0 (pure BM25) | 0.7046 |

A *mild* keyword weight beats pure semantic; a strong one hurts. The curve is stable in a
region (0.2–0.4) and drops sharply after 0.4, so the shape is more trustworthy than the
exact argmax. State the caveat: with 4 profiles a 0.02 nDCG difference is **one rank change
in one profile**, so do not claim 0.4 is optimal — claim the default is too high.

**Three honest caveats to state before anyone asks:**

1. **Corpus size invalidates the secondary claim.** `synthetic_v2` uses 5 listings;
   every method scores ~1.0 because rank-based similarity saturates. It cannot
   test keyword-vs-semantic. (OPEN_ISSUES M3)
2. **The v2 labels are derived, not judged.** Each profile is graded 2 at its
   target listing and 0 everywhere else. Those negatives are an *assumption*, so
   absolute v2 metrics are a lower bound. (M1)
3. **4 profiles is not enough to conclude anything.** One rank change moves the
   mean visibly. The result is a hypothesis plus a mechanism, not a finding. (M7)

**Answer to "is this statistically meaningful?"** — No, and we do not claim it is.
The deliverable of Prototype v1 is a working, reproducible harness plus a
mechanism worth testing at scale. That is what the corpus expansion addresses.

**Answer to "why not use RRF instead?"** — `main` uses reciprocal rank fusion and
reports @10; this branch uses min-max score blending and reports @5. The
disagreement is logged as a team decision (T2). RRF is rank-based so it is immune
to the scale mismatch that hurts score blending — that is the ablation to run.

## 5. Architecture in one diagram

```
data/            load_dataset(name) -> Dataset
                   opportunities.py | student_profiles.csv   (authored data)
                   prototype_dataset.py | synthetic_v2_dataset.py  (loaders)
                   schema.py  (Opportunity/Query/Dataset + validation)
                        |
research_matching_pipeline.py       (contains NO data)
   build_index(cfg, dataset, encoder)   -> SearchIndex    BM25 fit + doc embeddings, ONCE
   rank_candidates(index, cfg)          -> ResultTable     (3 x Q x C) numpy, one argsort
   evaluate_metrics(table, dataset, k)  -> per_profile, summary   fully vectorised
   ResultTable.to_frame(listings)       -> tidy CSV        built once, on export
```

Three modules to know: `SearchIndex` (cached retrieval state), `ResultTable`
(ranked numpy arrays), `Dataset` (data + precomputed grade matrix).

## 6. The offline trap (most likely way to mislead an audience)

With `--offline-embeddings` there is no MiniLM. The "semantic" condition becomes a
hashed bag-of-words **lexical proxy**, and the numbers change materially:

| Condition | Offline proxy nDCG@5 | Real MiniLM nDCG@5 |
| --- | --- | --- |
| Keyword (BM25) | 0.7046 | 0.7046 |
| Semantic | **0.6818** (proxy, lexical) | **0.8846** (real) |
| Hybrid | 0.7210 | 0.8364 |

The proxy ranks **below** BM25, so an offline run appears to show semantic search
*losing*. The method label and every exported CSV record which backend ran
(`Semantic (HashedBoW proxy)` vs `Semantic (all-MiniLM-L6-v2)`) — check that label
before quoting a number. Never present a proxy run as a semantic result.

## 7. Guaranteed demo (zero-dependency fallback)

If the model download fails or the machine is offline mid-presentation:

```powershell
.\.venv\Scripts\python.exe research_matching_pipeline.py --offline-embeddings
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q      # 65 passed, no model needed
```

Say explicitly: *"this is running the lexical fallback, so I am showing the
harness and the keyword numbers, not the semantic ones."* That is a recovery, not
a result.

## 8. Repo map

| Path | What it is |
| --- | --- |
| `research_matching_pipeline.py` | All retrieval + metric logic. No data. |
| `data/` | Datasets, loaders, canonical schema. |
| `tests/` | 93 tests: metrics vs brute force, retrieval properties, dataset contracts, CSV input, pinned baselines, CLI. |
| `scripts/reproduce.py` | **The presentation command.** Prints everything and verifies it. |
| `scripts/sweep.py` | Hybrid weight sweep (the M4 experiment). |
| `scripts/benchmark.py` | Phase-by-phase timing harness. |
| `results/` | **Committed** CSV artifact from the reproduction run. |
| `docs/RESULTS.md` | Reported numbers. |
| `docs/performance.md` | Where the time went, before and after. |
| `docs/verification_log.md` | What has actually been checked. |
| `docs/architecture.md`, `docs/hybrid_design.md` | Design detail. |
| `OPEN_ISSUES.md` | Known limitations, severity-ranked. |
| `docs/prototype_notes.md`, `docs/synthetic_v2_notes.md` | Dataset-specific notes. |

## 9. Known limitations to state up front

Full detail in `OPEN_ISSUES.md`. The five to volunteer:

| # | Limitation | Impact |
| --- | --- | --- |
| H1 | IRB status **[TBD]** for any real student data | Blocks the real-profile phase |
| M6 | Corpus is synthetic, not real VT listings | The RQ is about VT opportunities |
| M3 | 5-listing v2 corpus cannot resolve keyword-vs-semantic | Secondary result is illustrative only |
| M1 | v2 labels derived, not hand-judged | Absolute v2 numbers are a lower bound |
| M5 | Labels single-rater, no inter-rater agreement | No reliability estimate |

Proactively stating these is stronger than being asked about them.

## 10. Practical checks before you walk in

- [ ] `pytest -q` -> 71 passed
- [ ] `scripts/reproduce.py` -> "all reported values reproduced" (exit 0)
- [ ] MiniLM already cached: `scripts/reproduce.py` should finish in ~2 s, not ~30 s
- [ ] **Close `results/*.csv` in Excel/preview before presenting** - a locked file
      only produces a warning now, but the CSVs will not refresh
- [ ] Terminal font size increased for the room
- [ ] `results/` and `docs/RESULTS.md` open in tabs as backup if the live demo fails
- [ ] Can state the three caveats in section 4 from memory
- [ ] Know the offline trap in section 6

## 11. If the demo breaks live

| Symptom | Cause | Recovery |
| --- | --- | --- |
| `results/... is locked` warning | CSV open in Excel/preview | Harmless: values are still verified. Close it and re-run to refresh the CSVs. |
| Semantic score ≈ 0.68, below BM25 | `--offline-embeddings` is active | Drop the flag for the real model, or say you are showing the lexical fallback |
| Slow first run (~30 s hang) | MiniLM downloading | Expected only on a cold cache; pre-warm before presenting |
| A pinned value drifts | Something genuinely changed | Show `docs/RESULTS.md` and `results/` from the commit instead |
| `python: command not found` | venv not created / wrong shell | Use the full `.\\.venv\\Scripts\\python.exe` path as in section 1 |
