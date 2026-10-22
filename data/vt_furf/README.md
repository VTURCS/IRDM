# `vt_furf`: real Virginia Tech research listings

44 real undergraduate research opportunities from the **First Year Fralin Undergraduate Research
Fellowship (FURF) Projects List, 2026–2027**, across eight colleges, matched against 25 synthetic
student profiles.

```
python evaluate.py vt_furf
http://localhost:8000/?dataset=vt_furf
```

> **The relevance grades are a first draft written by an AI assistant, not human judgements.**
> Treat every number from this dataset as provisional until a person has reviewed
> `relevance.csv` (see [Reviewing the grades](#reviewing-the-grades)). An AI drafting the labels
> may favour the semantic method, because a language model and an embedding model judge
> similarity in related ways.

## What is real and what is not

| File | Source | Notes |
|---|---|---|
| `opportunities.csv` | **Real.** The FURF projects list | `title` and `description` are copied word for word from the source document's "Research project title" and "Description of project". Line breaks are collapsed to single spaces; nothing else is changed. A script checked all 88 fields against the original `.docx`: 0 differences. |
| `profiles.csv` | **Synthetic** | 25 fictional first-year students. No real people. |
| `relevance.csv` | **AI draft** | 1,100 grades, one for every (profile, listing) pair. |

Column mapping for `opportunities.csv`:

| Column | From the source | Searched? |
|---|---|---|
| `id` | `FURF-` plus the source's "ID Number", zero-padded (`FURF-07`). This is the posting's source project number, not its rank or match score. The source has no ID 34. | n/a |
| `title`, `description` | Verbatim, as above | Yes |
| `skills` | Concrete tools or knowledge named under "Experience or skills needed", e.g. `AutoCAD`, `R`, `Python`. Empty when the posting asks only for curiosity or reliability, which most first-year postings do. Normalised, not verbatim (the source's "MATHLAB" is written `MATLAB`). | Yes |
| `eligibility` | "Restrictions or time constraints", verbatim | No |
| `department` | "Academic College" | No |

The source document also has "Research goals" and "Mentoring style" sections. They are left out
of the searchable text on purpose. The semantic model reads at most 256 word-pieces per listing,
so long postings would be cut off for it but not for BM25, which would be unfair to the semantic
method. With descriptions only, the longest listing is 197 word-pieces and none is truncated.

The source lists no faculty names, and neither do these files.

## The profiles

25 profiles are tagged by type, like `heldout`, so the evaluation can break results down by kind
of request:

| Type | Count | Example |
|---|---|---|
| `exact` | 6 | "CRISPR gene editing in fungal pathogens" |
| `paraphrase` | 7 | "Helping pets stay calm when their paws are handled" (for the dog nail-trim project) |
| `vague` | 6 | "I love being outside and I care about the environment" |
| `token` | 2 | "TEM imaging" |
| `long` | 2 | A paragraph from a would-be veterinarian |
| `nomatch` | 2 | "I want to study classical ballet choreography" |

The paraphrase profiles were written to avoid the target listing's vocabulary, and a check
confirms they share only filler words ("how", "what") with it. That makes the paraphrase stratum
a deliberate stress test of keyword search, not a sample of how real students write.

## How the grades were drafted

- 0 = unsuitable, 1 = partial (adjacent topic or transferable skills), 2 = strong.
- Each pair was judged on the **whole posting**, including goals and skills, not just the
  description the retrievers see. A real reader would see the whole posting.
- Eligibility and time commitments were ignored: every FURF project is open to first-years from
  any major.
- When unsure between two grades, the lower one was chosen.
- Each profile has 1 to 12 relevant listings. The two `nomatch` profiles have none, so they are
  left out of the means.
- Four listings are relevant to no profile: FURF-01 (TEAM Malawi), FURF-23 (Syntactical
  Architecture), FURF-24 (Smart Coalition) and FURF-41 (meat color). They act as distractors.

## Reviewing the grades

This is the step that turns the numbers into evidence. In Excel, filter `relevance.csv` by
`profile_id`, read the profile in `profiles.csv`, and check two things:

1. Every non-zero grade is right.
2. No zero should have been a 1 or 2. Missed matches matter as much as wrong ones, because
   Recall divides by the number of relevant listings.

Ideally two people review the same profiles independently and compare. That also addresses
OPEN_ISSUES M5 (no agreement check). After editing, run `python -m pytest`: a test loads every
dataset folder, so a typo or a missing pair fails it with the file and line named.

## Provisional results (draft grades, k = 5)

| Method | nDCG@5 | Precision@5 | Recall@5 |
|---|---|---|---|
| Keyword (BM25) | 0.5059 | 0.3217 | 0.4844 |
| Semantic (all-MiniLM-L6-v2) | **0.8060** | **0.4870** | **0.7288** |
| Hybrid (0.6 KW / 0.4 SEM) | 0.6507 | 0.4087 | 0.6472 |

| Profile type | Keyword | Semantic | Hybrid |
|---|---|---|---|
| exact | 0.9206 | 0.9076 | 0.9342 |
| paraphrase | 0.1493 | 0.9383 | 0.4865 |
| vague | 0.3486 | 0.6640 | 0.4573 |
| token | 0.8634 | 0.8789 | 0.9010 |
| long | 0.6242 | 0.3915 | 0.7048 |

The hybrid keyword-weight sweep peaks at 0.2 (0.8160), against 0.6507 at the 0.6 default. These
results agree with `prototype`, but they rest on the draft grades above, and `token` and `long`
have only 2 profiles each.
