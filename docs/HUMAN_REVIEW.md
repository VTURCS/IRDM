# VT FURF human review integration

October 30, 2026. This report records human corrections to AI-drafted labels, not independent double annotation.

## Provenance and checks

- Original draft: commit `34233301ca736578f46bb909f593d9e69e65c73d`.
- Vaidehi: [PR #1](https://github.com/VTURCS/IRDM/pull/1), merged as `eaf41375897e138b415c26891defde55c3d819b4`, reports review of the first 550 pairs. Eleven grades changed.
- Sandilya (`sandykVT`): commit `8c47d2cad138806e017658817902809635285d73`, branch `secondhalf_human_generated_rows`, supplies `relevance550-1100.csv`. It contains the full 1,100-row matrix, but all 52 edits are in pairs 551–1100. Its unchanged first half is the old AI draft, not a second independent rating.
- Integration keeps main's first 550 pairs and takes the submitted second 550 pairs. Pair numbers exclude the header. The boundary is S13/FURF-22 (550), followed by S13/FURF-23 (551).
- Checked 1,100 unique pairs, identical pair order and IDs, grades in {0,1,2}, exact preservation of each reviewed half, and 63 total changed grades. The application reads `data/vt_furf/relevance.csv`; the root submission remains as source evidence.
- Change audit: [label_changes.csv](../results/vt_furf/label_changes.csv). Review coverage follows the contributors' stated scopes. Unchanged grades do not prove independent judgments.

## Reproduced comparison

Run `python evaluate.py vt_furf --top-k 5 --bm25-weight 0.6 --output-dir results`.
Model, opportunities, profiles and ranking code are unchanged. Only reference grades changed.

| Method | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword | 0.4852 | 0.3083 | 0.4608 |
| Semantic | 0.8079 | 0.4417 | 0.7192 |
| Hybrid 0.6/0.4 | 0.6228 | 0.3917 | 0.6326 |

[Rankings](../results/vt_furf/match_results.csv), [per-profile results](../results/vt_furf/evaluation_per_profile.csv), and [summary](../results/vt_furf/evaluation_summary.csv) are saved. Semantic remains highest overall. These means include 24 profiles; only S25 is excluded.

S24/FURF-15 changed from 0 to 1. S24 therefore enters the means, unlike the old 23-profile evaluation. The historical `nomatch` query-type tag remains in profiles.csv to avoid silently rewriting the study design; it no longer describes the reviewed relevance status of S24. Comparisons with old aggregate scores reflect both changed grades and changed inclusion. S24's new partial match warrants discussion before calling it a validated negative control.

Validation: `python -m pytest` completed with **40 passed** on Windows / Python 3.13.17. The VT aggregate scores are now pinned in the regression suite.

## Remaining issues

| Issue | Severity | Reproduction/evidence | Owner | Next step |
|---|---|---|---|---|
| Rater agreement absent | Medium | Reviews cover separate halves; no independent overlapping rating sheets were submitted | TBD | Collect independent ratings on a shared subset, measure agreement before resolving disagreements |
| Unsuitable top results | High | S08: keyword FURF-16, hybrid FURF-32; S22: semantic FURF-16; inspect saved rankings and labels | TBD | Document failures and evaluate changes separately from final evaluation; no ranking fix is claimed here |
| No-match behavior | High | S25 has all-zero labels but every baseline returns five items | TBD | Evaluate abstention separately from ranking quality; UI cutoffs are heuristic |
| S24 negative-control assumption changed | Medium | Original `nomatch` tag versus reviewed S24/FURF-15 grade 1 | TBD | Confirm the rationale with the reviewer and document the final negative-control definition |

No agreement statistic, blind-review procedure, adjudication record, or real-student outcome evidence is claimed. The 25 profiles remain synthetic. Topical relevance does not establish eligibility. A second human review of overlapping pairs is still required before reporting inter-rater agreement.
