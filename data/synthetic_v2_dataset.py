"""
Synthetic evaluation dataset (v2) - the purpose-built 5x4 benchmark.

Sources (kept as separate files so they can be edited without touching code):
    data/opportunities.py       -> 5 listings (opp_id, title, department, ...)
    data/student_profiles.csv   -> 20 student profiles, 4 per listing

    20 profiles = 5 listings x 4 match types:
        exact_keyword     uses the listing's own vocabulary
        semantic_analogy  same competence, different words  <-- the interesting case
        distractor        shares only generic tooling (Python/C++/Docker), no domain fit
        noisy_edge        mixed contexts, uncalibrated skill lists

Field mapping to the canonical schema in `data/schema.py`:
    opp_id            -> Opportunity.id
    title             -> Opportunity.title
    description       -> Opportunity.description
    key_terms         -> Opportunity.skills      (exposed to the SAME text path)
    department        -> Opportunity.department
    (no eligibility field in v2 -> Opportunity.eligibility is left empty)

    skills_and_experience + project_summary -> Query.text
    major / academic_year                   -> Query.major / Query.year

===============================================================================
HOW RELEVANCE LABELS ARE DERIVED (read this before trusting the numbers)
===============================================================================
v1's labels were hand-written per (profile, listing) pair. v2 has ground truth
only in the weaker form of each profile's `target_opp_id` plus its `match_type`,
so labels are DERIVED by this rule:

    target listing, match_type exact_keyword    -> 2  (strong)
    target listing, match_type semantic_analogy -> 2  (strong)
    target listing, match_type noisy_edge       -> 1  (partial)
    target listing, match_type distractor       -> 0  (unsuitable)
    every non-target listing                    -> 0  (unsuitable)

Two consequences that MUST be stated in any write-up using these numbers:

1. The negative labels ("every non-target listing is 0") are NOT judgements.
   They are an assumption. A profile targeting the UAV listing is graded 0
   against the IoT listing purely because it was authored as a UAV example -
   nobody assessed whether it could also suit the IoT lab. Real semantic systems
   are penalised for finding genuinely good cross-listing matches, which biases
   the reported metrics downward. This is the main measurement threat here.

2. Because relevance is concentrated on exactly one listing per profile,
   every relevant set has size 1 and Recall@k is binary: it is 1.0 if the
   single target appears in the Top-k and 0.0 otherwise. Precision@k collapses
   to 1/k when the target is found. nDCG@k remains informative because it also
   rewards getting the target to rank 1. Read `mean_ndcg@k` as the headline and
   treat P/R as coarse sanity checks, not as fine-grained comparisons.

Use `--match-type` to restrict the run to one label stratum (e.g. only
`semantic_analogy`), which is the cleanest way to test the central hypothesis:
that keyword-only retrieval should collapse on the semantic stratum while dense
retrieval holds up.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from data.schema import Dataset, Opportunity, Query

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

OPPORTUNITIES_MODULE = Path("opportunities.py")
PROFILES_CSV = Path("student_profiles.csv")

TARGET_GRADE_BY_MATCH_TYPE: Dict[str, int] = {
    "exact_keyword": 2,     # uses the listing's own vocabulary
    "semantic_analogy": 2,  # same competence, different words
    "noisy_edge": 1,        # plausible but messy
    "distractor": 0,        # only generic tooling overlap
}

# Grade given to every listing a profile is NOT targeting. See caveat (1) above:
# this is an assumption, not an assessment.
NON_TARGET_GRADE = 0

DATASET_DIR = Path(__file__).resolve().parent


def _load_opportunities_module(path: Path):
    """
    Import `opportunities.py` by file path.

    `data/` is a plain script directory, not an installed package, so the module
    is loaded via importlib rather than a normal import.
    """
    spec = importlib.util.spec_from_file_location("_v2_opportunities", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load opportunities module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_profiles_csv(path: Path) -> List[Dict[str, str]]:
    """
    Read the student profile CSV.

    Parsed with the stdlib `csv` module rather than pandas so that dataset
    loading stays dependency-light and the file can be validated (quoting,
    column order) without a dataframe in the way.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path} contains no data rows.")

    required = {
        "student_id", "target_opp_id", "match_type", "full_name",
        "major", "academic_year", "skills_and_experience", "project_summary",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def load(
    allowed_match_types: Sequence[str] | None = None,
) -> Dataset:
    """
    Build the v2 dataset.

    Parameters
    ----------
    allowed_match_types
        Optional whitelist of `match_type` values. When given, only profiles of
        those types are loaded (labels are still computed against the FULL
        corpus, so recall denominators stay correct). This is the supported way
        to run a label-stratified ablation such as `--match-type semantic_analogy`.
    """
    module = _load_opportunities_module(DATASET_DIR / OPPORTUNITIES_MODULE)
    raw_opportunities = module.OPPORTUNITIES

    opportunities = [
        Opportunity(
            id=str(raw["opp_id"]),
            title=str(raw["title"]),
            description=str(raw["description"]),
            # key_terms are surfaced through `skills`, so they reach the
            # searchable text via the exact same path v1's required_skills did.
            skills=[str(term) for term in raw["key_terms"]],
            department=str(raw.get("department", "")),
            eligibility="",  # v2 has no eligibility field
        )
        for raw in raw_opportunities
    ]
    opportunity_ids = [o.id for o in opportunities]
    valid_ids = set(opportunity_ids)

    rows = _read_profiles_csv(DATASET_DIR / PROFILES_CSV)

    if allowed_match_types is not None:
        allowed = set(allowed_match_types)
        unknown = allowed - set(TARGET_GRADE_BY_MATCH_TYPE)
        if unknown:
            raise ValueError(
                f"Unknown match_type filter {sorted(unknown)}; "
                f"valid values are {sorted(TARGET_GRADE_BY_MATCH_TYPE)}"
            )
        rows = [r for r in rows if r["match_type"] in allowed]
        if not rows:
            raise ValueError(f"No profiles left after filtering match_type to {sorted(allowed)}")

    queries: List[Query] = []
    for row in rows:
        if row["target_opp_id"] not in valid_ids:
            raise ValueError(
                f"Profile {row['student_id']} targets unknown listing "
                f"'{row['target_opp_id']}'."
            )
        if row["match_type"] not in TARGET_GRADE_BY_MATCH_TYPE:
            raise ValueError(
                f"Profile {row['student_id']} has unknown match_type "
                f"'{row['match_type']}'."
            )

        # Major/year stay metadata; only the free-text fields are searchable, so
        # the retrievers see the same substance a human reviewer would.
        text = f"{row['skills_and_experience']}. {row['project_summary']}"
        queries.append(
            Query(
                profile_id=row["student_id"],
                text=text,
                major=row["major"],
                year=row["academic_year"],
                match_type=row["match_type"],
                target_opp_id=row["target_opp_id"],
            )
        )

    # --- derive labels -------------------------------------------------------
    # Each profile's own target listing gets its match_type grade; everything
    # else gets NON_TARGET_GRADE. Note that labels are built from the FULL
    # opportunity list regardless of any match_type filter, so a stratified run
    # still scores against the whole corpus.
    relevance_map: Dict[Tuple[str, str], int] = {}
    for query in queries:
        row = next(r for r in rows if r["student_id"] == query.profile_id)
        for opp_id in opportunity_ids:
            if opp_id == row["target_opp_id"]:
                relevance_map[(query.profile_id, opp_id)] = TARGET_GRADE_BY_MATCH_TYPE[row["match_type"]]
            else:
                relevance_map[(query.profile_id, opp_id)] = NON_TARGET_GRADE

    strata = sorted({q.match_type for q in queries})
    return Dataset(
        name="synthetic_v2",
        description=(
            f"Purpose-built benchmark: {len(opportunities)} listings, {len(queries)} profiles "
            f"(strata: {', '.join(strata)}), labels DERIVED from match_type."
        ),
        opportunities=opportunities,
        queries=queries,
        relevance_map=relevance_map,
        suggested_top_k=min(5, len(opportunities)),
    )
