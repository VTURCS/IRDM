# IRDM — Week 5 experimental prototype

**Question:** How do BM25, semantic (MiniLM), and weighted hybrid retrieval compare when matching short undergraduate-interest profiles to Virginia Tech research opportunities?

**Status:** runnable synthetic pilot, not a completed VT study. The demo has 12 fictional opportunities, five constructed profiles, and illustrative labels. Root CSVs are empty study templates. Success means a reproducible comparison, whether hybrid wins, ties, or loses.

## Run / test

Python 3.12; PowerShell from the repository root (the lock file records the tested Windows environment):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe benchmark.py --data-dir demo
```

The first run downloads MiniLM; inference runs locally on CPU. Expect 150 ranking rows and 15 metric rows. The no-match case has zero precision and undefined recall/nDCG. [Cases, results, and verification](WEEK5.md) are software checks, not research findings.

## Conditions and data

All methods use identical title + description text, queries, corpus, labels, and top-10 cutoff. No filtering, expansion, reranking, frontend, or live student data is included.

| Method | Fixed settings |
| --- | --- |
| BM25 | Lowercase word tokens; k1=1.5, b=0.75, epsilon=0.25. |
| Semantic | `sentence-transformers/all-MiniLM-L6-v2`, pinned revision in code; normalized embeddings and cosine similarity. |
| Hybrid | `0.6/(60+BM25 rank) + 0.4/(60+semantic rank)`; ranks start at 1. Omit keyword contribution when all BM25 scores are zero. |

Inputs: `opportunities.csv` (`id,title,description`), `profiles.csv` (`id,query`), `relevance.csv` (`profile_id,opportunity_id,grade`). Outputs: `results.csv` (rankings and text), `metrics.csv` (per-profile scores/status). Successful runs replace outputs; preserve separate run folders. Extra metadata columns are not searched unless included in description. Scores are method-specific. Ties break by opportunity ID.

## Evaluation / critique response

Primary: **nDCG@10**; secondary: **Precision@10, Recall@10**. Human grades are 0 unsuitable, 1 partial, 2 strong. Grades 1–2 count as relevant; nDCG gains are `2^grade-1`. Precision needs all returned labels; recall/nDCG require full-corpus labels. Missing judgments stay missing. No-relevant profiles have undefined recall/nDCG; averages report evaluated counts. Compare identical profile subsets. With fewer than ten listings, precision divides by the returned count (`returned`). An absent relevance file produces rankings and header-only metrics; a header-only relevance file flags missing labels.

The [labeling protocol](WEEK5.md#proposed-labeling-rules) proposes two independent raters, evidence-based topic/eligibility/skill rules, and recorded disagreement resolution; team approval and named raters remain pending. There is no implemented 50/30/20 rubric. Retrieval weights are separate fixed pilot choices. Evaluation stays at ten; displaying 3–5 results is a future interface proposal. Vague interests are included among the five cases.

## Week 5 deliverable status

| Required item | Evidence / gap |
| --- | --- |
| Prototype + experiment README | `benchmark.py` and this page. |
| 3–5 non-participant cases + initial results | Five demo cases; [results table](WEEK5.md). Synthetic only. |
| Contribution log + integration | [Ownership/work record](WEEK5.md#ownership-and-contributions); each member must confirm completed work. |
| Open issues | Real VT listings, agreed rubric/raters, independent full-corpus labels, real-data run, and tie/weight sensitivity. |

Next: populate the root templates with source-linked public VT listings and constructed profiles, agree labels, then run `.\.venv\Scripts\python.exe benchmark.py`. Verify opening availability separately from topic fit. Live student data collection remains outside v1 and requires adviser review of consent and applicable human-subjects requirements. Five synthetic cases do not establish general effectiveness.
