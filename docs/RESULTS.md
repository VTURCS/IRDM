> Updated VT results after the October 30 human reviews are in [HUMAN_REVIEW.md](HUMAN_REVIEW.md) and `results/vt_furf/`. The prototype results below are unchanged.

# Initial Results: Prototype v1 (DRAFT)

Dataset: `prototype` | 8 synthetic listings | 4 hand-graded profiles | **k = 5** (not 10)
Labels: 0-2 rubric. Precision and Recall treat grade >= 1 as relevant.

| Condition | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword (BM25) | 0.7046 | 0.55 | 0.7417 |
| Semantic (MiniLM) | **0.8846** | **0.65** | **0.8875** |
| Hybrid (0.6 BM25 / 0.4 MiniLM) | 0.8364 | 0.65 | 0.8875 |

**Provenance.** Every number on this page is pinned in `tests/test_irdm.py`, so a change that
moves one fails the tests instead of silently changing a published result. The CSVs are
committed in `results/` (written by `python evaluate.py <dataset> --output-dir results`).
After following the [README quick start](../README.md#quick-start), reproduce in an activated terminal with:

```
python -m pytest
python evaluate.py prototype --sweep
```

Environment: Python 3.13, numpy 2.5.3, pandas 3.0.6, sentence-transformers 6.1.0,
torch 2.14.1+cpu, model `all-MiniLM-L6-v2` at revision `1110a24` (all pinned in
`requirements.txt`, `irdm.py` and the Docker image). Verified both in Docker and on Windows.

## Secondary: `synthetic_v2` (5 listings, 20 profiles, derived labels), mean nDCG@5
| Condition | nDCG@5 |
|---|---|
| Keyword (BM25) | 0.9667 |
| Semantic (MiniLM) | 0.9421 |
| Hybrid (0.6/0.4) | 0.9754 |

Artifact: `results/synthetic_v2/evaluation_summary.csv`.
Interpret with caution: derived labels and a 5-document corpus (OPEN_ISSUES M1-M3).
Note that every condition reaches ~1.0 — all three methods put the target listing at rank 1
in almost every case, which means this corpus cannot distinguish the conditions at all.

## Hybrid weight sweep (OPEN_ISSUES M4) - corrects the naive reading

`python evaluate.py prototype --sweep`, real MiniLM, k=5. The hybrid condition's mean nDCG@5
as `bm25_weight` varies (0.0 = pure semantic, 1.0 = pure BM25):

| bm25_weight | 0.0 | 0.2 | 0.3 | **0.4** | 0.6 (default) | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| hybrid nDCG@5 | 0.8846 | 0.9011 | 0.9004 | **0.9030** | 0.8364 | 0.8239 | 0.7046 |

**This changes the headline interpretation.** The hybrid is not simply worse than semantic:
the *default* 0.6 weight is mistuned. A mild keyword weight (0.2-0.4) beats pure semantic
(+0.018 nDCG at 0.4), and performance falls sharply beyond 0.4. So the finding is about the
tuning of score-level fusion, not evidence that keyword signal is harmful.

Caveats, both important:
- With 4 profiles a ~0.02 nDCG difference is **one rank change in one profile**. The stable
  region (0.2-0.4) is more trustworthy than the argmax. Do not claim 0.4 is optimal; claim
  the default 0.6 is too high.
- At 0.0 and 1.0 the hybrid reproduces the pure conditions exactly, which is a useful sanity
  check that the weighting is wired correctly.

## Reading the results
- On the prototype set, semantic (0.8846) beats BM25 (0.7046), and the hybrid at its default
  weight (0.8364) sits between them - but the sweep shows the default is mistuned rather than
  the mechanism being inherently worse.
- With 4 profiles and 8 listings this is a hypothesis plus a mechanism, not a finding.
- Absolute values are not the finding; relative ordering and failure cases are.
