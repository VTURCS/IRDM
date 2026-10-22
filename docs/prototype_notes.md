# Prototype finding and caveat

> Moved from the original README (unchanged text).

## Prototype finding (mock data, k = 5)

With real MiniLM embeddings the ranking **does not** favour the keyword-prioritized hybrid:

| Method | mean nDCG@5 | mean P@5 | mean R@5 |
| --- | --- | --- | --- |
| Keyword (BM25) | 0.7046 | 0.55 | 0.74 |
| **Semantic (MiniLM)** | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 / 0.4) | 0.8364 | 0.65 | 0.8875 |

Semantic wins on every profile; the hybrid lands *between* the two, strictly worse than pure
semantic. The mechanism is in the fusion rule: a listing that is a strong semantic match but
shares no vocabulary with the vague query receives `bm25_norm = 0`, so the 0.4 semantic weight
is all it keeps — while a lexicon-overlap-only listing gets `bm25_norm = 1` and outranks it.

This is exactly the failure mode the study is set up to detect, and it makes the
**`--bm25-weight` sweep (0.0 → 1.0) the highest-value ablation**: for short, vague student
queries, a keyword *priority* may simply be the wrong prior. Re-run the sweep before drawing
conclusions — with a synthetic corpus of 8 items and hand-written labels, the absolute numbers
above are not the finding, and a single listing changing its rank moves the means noticeably.


## Prototype caveat

The corpus is **synthetic** and labels are **hand-written**. Absolute metric values are not
the finding — the relative ordering of methods and the failure cases are what this pilot
exists to surface before the study scales to real listings. Where the pilot already delivers:
it produced a concrete, mechanistically-explained hypothesis (keyword priority hurts on vague
queries) and a measurable next experiment to test it.
