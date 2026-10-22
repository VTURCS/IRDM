# Presentation Runbook: Prototype v1

Everything needed to present this branch. Learn the **cheat sheet** (section 3) before you
stand up.

## 1. Before you walk in

Follow the [README quick start](../README.md#quick-start) to install dependencies and start
the app. In a second terminal, activate the same environment and run `python -m pytest`
(expect `20 passed`). Use that activated terminal for the optional CLI command below.

A pass means every number below reproduces on the machine you're presenting from. If anything
fails, **don't present the numbers**; show `docs/RESULTS.md` and `results/` instead.

- [ ] Dependencies are installed and the app has launched once to cache the model. Do this before arriving at the venue.
- [ ] Browser zoomed in for the room. Have a terminal open in the repository folder.
- [ ] You can state the three caveats in section 3 from memory.

## 2. Demo sequence (about 4 minutes)

Have the four URLs below open in tabs before you start, so you are never typing live.

| Step | Do this | Point at |
|---|---|---|
| (a) | `localhost:8000` — expand **Evaluation** | Semantic leads (nDCG 0.885). The **default** hybrid (0.836) is worse than pure semantic. |
| (b) | Paste profile **P4** and press **Match**: `I am an HCI and design person, I like studying people and technology.` | BM25's top pick is a battery-cooling listing graded *unsuitable*; semantic finds the usable-security study. That's vocabulary mismatch, live. |
| (c) | `localhost:8000/?weight=0.4` — expand **Evaluation** | The hybrid rises to 0.903, above pure semantic. A *light* keyword weight helps; the default is just mistuned. |
| (d) | `localhost:8000/?dataset=heldout` — expand **Evaluation** for the by-type table | BM25 holds up on `exact` (0.95), but collapses on `paraphrase` (0.47) and `vague` (0.20). |
| (e) | *Optional:* `python evaluate.py prototype --sweep` | The full weight curve in the terminal, plus the CSVs in `outputs/`. |

The search box is the only control on the page. Dataset, k and keyword weight are set
through the URL, and a search keeps whichever ones are already in it.

## 3. Presenter cheat sheet

**Research question.** How do keyword-only (BM25), semantic-only (MiniLM), and
keyword-prioritized hybrid retrieval compare when matching short, vague student profiles to VT
research opportunities?

**Conditions.** All three use the identical corpus and identical verbatim queries. There is no
metadata filtering and no query expansion, and eligibility text is kept out of the searchable
text.

| Condition | Mechanism |
| --- | --- |
| Keyword | BM25Okapi with negative IDFs floored at 0 |
| Semantic | `all-MiniLM-L6-v2`, cosine similarity |
| Hybrid | `0.6 · minmax(BM25) + 0.4 · minmax(cosine)`, normalised per query |

**Headline numbers** (prototype: 8 listings, 4 hand-graded profiles, k = 5):

| Condition | nDCG@5 | P@5 | R@5 |
| --- | --- | --- | --- |
| Keyword (BM25) | 0.7046 | 0.55 | 0.7417 |
| **Semantic (MiniLM)** | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 / 0.4) | 0.8364 | 0.65 | 0.8875 |

**Why the default hybrid loses.** Take a listing that's a strong semantic match but shares no
words with a vague query. It gets `bm25_norm = 0`, so it keeps only its 0.4 share. Meanwhile a
listing that only overlaps lexically gets `bm25_norm = 1` and outranks it. Score-level fusion
inherits BM25's brittleness to vocabulary mismatch by construction.

**The weight sweep corrects the naive reading:**

| bm25_weight | 0.0 (semantic) | 0.2 | 0.3 | **0.4** | 0.6 (default) | 0.8 | 1.0 (BM25) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid nDCG@5 | 0.8846 | 0.9011 | 0.9004 | **0.9030** | 0.8364 | 0.8239 | 0.7046 |

With 4 profiles, a 0.02 nDCG difference is **one rank change in one profile**. So don't claim
0.4 is optimal; claim the default is too high.

**Three caveats to state before anyone asks:**
1. **The `synthetic_v2` corpus is too small.** It has 5 listings, and every method scores ~1.0, so it can't separate the methods. (OPEN_ISSUES M3)
2. **The `synthetic_v2` labels are derived, not judged.** Each profile is graded at its target listing and 0 everywhere else, so absolute numbers are a lower bound. (M1)
3. **Four profiles prove nothing.** One rank change moves the mean. This is a hypothesis plus a mechanism, not a finding. (M7)

**"Is this statistically meaningful?"** No, and we don't claim it is. The deliverable is a
working, reproducible harness plus a mechanism worth testing at scale.

**"Why not RRF?"** `main` used reciprocal rank fusion and reported @10; this branch uses
min-max score blending at @5. That's an open team decision (T2). RRF is rank-based, so it's
immune to the scale mismatch that hurts score blending. That's the ablation to run next.

## 4. Known limitations to volunteer

| # | Limitation | Impact |
| --- | --- | --- |
| H1 | IRB status **[TBD]** for any real student data | Blocks the real-profile phase |
| M6 | The corpus is synthetic, not real VT listings | The research question is about VT opportunities |
| M3 | A 5-listing corpus can't resolve keyword vs semantic | The `synthetic_v2` result is illustrative only |
| M1 | `synthetic_v2` labels are derived, not hand-judged | Absolute `synthetic_v2` numbers are a lower bound |
| M5 | Labels come from one rater, with no agreement check | No reliability estimate |

## 5. If the demo breaks

| Symptom | Recovery |
| --- | --- |
| Startup or port error | See [README troubleshooting](../README.md#troubleshooting). |
| The page shows a red error message | A dataset CSV is malformed; the message names the file and line. Open `localhost:8000/?dataset=prototype` to get back to a working one. |
| Nothing works | Show `docs/RESULTS.md` and the committed CSVs in `results/`. |
