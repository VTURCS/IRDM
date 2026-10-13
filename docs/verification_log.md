# Verification log

> Moved from the original README (unchanged text).

## Verification log (what has actually been checked)

| Claim | Method | Result |
| --- | --- | --- |
| Refactor preserves metrics | run prototype + synthetic_v2, diff summary CSVs | byte-identical |
| Vectorised metrics are correct | 20 random datasets, 843 rows vs brute-force reference | 0 mismatches |
| Methods are not degenerate | check BM25 vs Semantic rankings differ | 20/20 cases differ |
| `prototype` baseline (real MiniLM) | `mean_ndcg@k` | 0.7046 / 0.8846 / 0.8364 |
| `synthetic_v2` baseline (real MiniLM) | `mean_ndcg@k` | 0.9667 / 0.9421 / 0.9754 |
| Export schema unchanged | inspect `match_results.csv` header | 7 expected columns |
| Error handling | `--dataset nope` | clean argparse failure, exit 2 |

Known reproduction command for the headline number:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --scale 2000 --repeat 3
```

