# synthetic_v2 measurement notes

> Moved from the original README (unchanged text).

## Two measurement problems in `synthetic_v2` (read before citing its numbers)

**1. Derived labels are a lower bound, not ground truth.** Each profile is graded 2 at its
`target_opp_id` (1 for `noisy_edge`, 0 for `distractor`) and **0 against every other
listing**. Those negatives are an *assumption*, not an assessment — nobody judged whether a
UAV-focused student would also suit the IoT lab. A retriever that finds a genuinely good
cross-listing match is penalised for it, so absolute metrics are biased downward.
`prototype`'s hand-graded labels do not have this problem.

**2. Structural degeneracy.** Because relevance is concentrated on exactly one listing,
every relevant set has size 1, so:
- Recall@k is binary — 1.0 if the target is in Top-k, else 0.0.
- Precision@k collapses to a near-constant 1/k.
- nDCG@k reduces to "did the target land at rank 1" (with partial credit for rank 2).

Only **nDCG@k** carries real signal; treat P/R as coarse sanity checks. To fix this properly,
each profile needs graded labels against *several* listings — which means hand-grading, as
`prototype` does.

### How the harness handles the null-information rows

Distractor profiles have no relevant listing, so every method scores them 0.0 *by
construction*. Those rows cannot discriminate between retrievers and would dilute the mean,
so they are:
- **excluded from all reported means** (marked `scorable = False`, still written to CSV), and
- re-evaluated as a **negative control** instead: the rank each method gave a listing the
  profile has no domain fit for. Lower is better; the useful comparison is the random
  baseline `(corpus_size + 1) / 2`.

Distractors legitimately belong in the benchmark as a false-positive probe — it was the
*scoring* of them that was wrong, not their inclusion.


## Result: the new benchmark contradicts the hypothesis

`synthetic_v2`, real MiniLM embeddings, k = 5:

```
match_type        Keyword (BM25)   Semantic (MiniLM)   Hybrid (0.6/0.4)
exact_keyword          1.0000            1.0000             1.0000
semantic_analogy       1.0000            0.9262             1.0000
noisy_edge             0.9000            0.9000             0.9262
```

**BM25 wins the `semantic_analogy` stratum outright**, which was supposed to be the case it
should lose. The mechanism is not that BM25 is secretly doing semantics — it is that the
stratum is contaminated at the *corpus* level:

> STU-009 was written to describe Linux administration without the listing's vocabulary
> (`home hypervisor`, `spaced copies`), but it still shares three literal words with
> `VT-LNX-03`: `containerized`, `maintain` and `services`. With only 5 listings, those
> words are rare enough to carry high IDF, so BM25 scores `VT-LNX-03` at 3.59 against 2.24
> for the next listing. A few leaked words are enough to win.

Empirically, *every* method put the target at rank 1 for 4 of 5 semantic profiles; the
sequence was 1-1-1-1-2 against BM25's 1-1-1-1-1. So this is not evidence that keywords beat
semantics — it is evidence that **a 5-document corpus cannot support the claim.** Rank-based
similarity saturates, IDFs are unreliable, and a few shared words dominate.

**Before this benchmark can test the hypothesis it needs** (a) substantially more listings
(≥50, ideally with near-duplicate or overlapping-skill pairs), (b) hand-graded labels across
multiple listings per profile, and (c) removal of the literal keyword overlap still present
in the `semantic_analogy` profiles (`data/synthetic_v2/profiles.csv`).

The derived grades are written out in full in `data/synthetic_v2/relevance.csv`; each
profile's `target_id` and `match_type` are in `profiles.csv`.

