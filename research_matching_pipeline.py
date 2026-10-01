"""
===============================================================================
VTURCS Team 1 - IRDM Prototype: Research-Opportunity Matching Evaluation
===============================================================================

Goal
----
Measure how a **keyword-prioritized hybrid retriever** compares against
**keyword-only (BM25)** and **semantic-only (dense embedding)** retrieval when
matching *short, vague undergraduate student profiles* to *Virginia Tech
research opportunity listings*.

Three retrieval methods are evaluated on the **exact same corpus and queries**
(baseline constraint - see `Config` below):

    Method 1  KW       : BM25 (Okapi) over a tokenized corpus.
    Method 2  SEM      : Cosine similarity between all-MiniLM-L6-v2 embeddings.
    Method 3  HYBRID   : 0.6 * minmax(BM25) + 0.4 * minmax(cosine).

Outputs
-------
    outputs/match_results.csv           # long table: profile x method x rank
    outputs/evaluation_per_profile.csv  # nDCG@k / P@k / R@k per profile
    outputs/evaluation_summary.csv      # mean metrics per method
    (+ a printed console report)

Run
---
    python research_matching_pipeline.py
    python research_matching_pipeline.py --top-k 3 --bm25-weight 0.7
    python research_matching_pipeline.py --offline-embeddings   # no model download

Author: prepared for VTURCS Team 1 (IRDM)
===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


# =============================================================================
# 1. CONFIGURATION  (single source of truth for the experiment)
# =============================================================================


@dataclass(frozen=True)
class Config:
    """
    All knobs for the prototype live here so the experiment is reproducible.

    BASELINE CONSTRAINTS (deliberately OFF for this prototype)
    ----------------------------------------------------------
    `metadata_filtering=False` : `eligibility` (major / year / GPA) is stored as
        metadata but is NEVER used to filter or re-rank candidates. This keeps
        all three methods retrieval-only and comparable. Turning this on is the
        first planned ablation for the full study.

    `query_expansion=False` : student queries are used verbatim (after the same
        light normalization applied to documents). No synonym lists, no PRF/RM3,
        no LLM query rewriting. Also a planned ablation.

    `include_eligibility_in_text=False` : even as plain text, eligibility
        strings are excluded from the searchable document body, otherwise
        method differences would partly reflect major/year keyword overlap
        rather than topical matching skill.
    """

    # --- retrieval / ranking -------------------------------------------------
    top_k: int = 5                      # dataset is small -> Top-5 (spec allows)
    bm25_weight: float = 0.6            # keyword priority
    semantic_weight: float = 0.4
    bm25_k1: float = 1.5                # standard Okapi defaults
    bm25_b: float = 0.75

    # --- encoders ------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- baseline switches (kept explicit so they cannot silently drift) -----
    metadata_filtering: bool = False
    query_expansion: bool = False
    include_eligibility_in_text: bool = False

    # --- io ------------------------------------------------------------------
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        total = self.bm25_weight + self.semantic_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Hybrid weights must sum to 1.0 (got {total:.3f}). "
                "Adjust bm25_weight / semantic_weight."
            )


# =============================================================================
# 2. MOCK DATA: VIRGINIA TECH RESEARCH OPPORTUNITY LISTINGS
# =============================================================================
# NOTE: these are SYNTHETIC listings written to resemble real VT research
# postings. They exist so the pipeline is executable today; they are meant to be
# swapped for a scrape/export of the real opportunity board.
#
# Schema (per spec):
#   id               -> str   unique opportunity id
#   title            -> str
#   description      -> str   research scope
#   required_skills  -> list[str]
#   eligibility      -> str   "Major(s), Year+"  (metadata only - not filtered)

OPPORTUNITIES: List[Dict[str, object]] = [
    {
        "id": "VT-001",
        "title": "Vision-Language Models for Autonomous Driving",
        "description": (
            "Build multimodal perception models that fuse camera imagery with "
            "natural-language scene descriptions to improve pedestrian and "
            "hazard detection for campus autonomous shuttles. Work spans "
            "dataset curation, transformer fine-tuning and closed-loop "
            "simulation evaluation."
        ),
        "required_skills": ["Python", "PyTorch", "Computer Vision", "Deep Learning"],
        "eligibility": "Computer Science, Electrical Engineering, Junior+",
    },
    {
        "id": "VT-002",
        "title": "Smart Grid Load Forecasting with Time-Series ML",
        "description": (
            "Develop forecasting models for campus and regional electricity "
            "demand using historical sensor streams. Projects include feature "
            "engineering for weather data, probabilistic forecasting, and "
            "deployment of dashboards for utility partners."
        ),
        "required_skills": ["Python", "Pandas", "Machine Learning", "Time Series"],
        "eligibility": "Electrical Engineering, Industrial Systems Engineering, Sophomore+",
    },
    {
        "id": "VT-003",
        "title": "Human-Robot Interaction for Assistive Robotics",
        "description": (
            "Design and evaluate collaborative robot behaviors that help older "
            "adults with daily tasks. Involves ROS-based control code, "
            "motion planning, and user studies measuring trust and workload."
        ),
        "required_skills": ["ROS", "C++", "Python", "Robotics"],
        "eligibility": "Mechanical Engineering, Computer Science, Junior+",
    },
    {
        "id": "VT-004",
        "title": "Genomic Data Pipelines for Precision Agriculture",
        "description": (
            "Analyze plant and livestock genomic datasets to identify markers "
            "linked to drought resilience. Tasks include variant calling, "
            "statistical association testing and reproducible pipeline "
            "engineering on high-performance computing clusters."
        ),
        "required_skills": ["R", "Bioinformatics", "Statistics", "Genomics"],
        "eligibility": "Biology, Biochemistry, Computer Science, Sophomore+",
    },
    {
        "id": "VT-005",
        "title": "Usable Security: Evaluating Password Manager UX",
        "description": (
            "Run mixed-methods human subjects studies on how non-experts adopt "
            "password managers. Involves survey and interview design, "
            "qualitative coding, usability testing and lightweight analysis "
            "scripts for behavioral logs."
        ),
        "required_skills": ["Human-Computer Interaction", "Survey Design", "Python", "Statistics"],
        "eligibility": "Computer Science, Psychology, Freshman+",
    },
    {
        "id": "VT-006",
        "title": "Reinforcement Learning for Drone Swarm Coordination",
        "description": (
            "Train multi-agent reinforcement learning policies that let aerial "
            "robot teams survey disaster zones cooperatively. Emphasis on "
            "reward shaping, high-fidelity simulation and sim-to-real transfer."
        ),
        "required_skills": ["Python", "Reinforcement Learning", "PyTorch", "Simulation"],
        "eligibility": "Computer Science, Aerospace Engineering, Junior+",
    },
    {
        "id": "VT-007",
        "title": "LLM-Based Tutoring Chatbot for CS1 Courses",
        "description": (
            "Build and evaluate a retrieval-augmented large language model "
            "tutor that gives programming hints without revealing solutions. "
            "Work covers prompt engineering, natural language processing "
            "evaluation, and a small web front end for students."
        ),
        "required_skills": ["NLP", "Transformers", "Python", "Web Development"],
        "eligibility": "Computer Science, Sophomore+",
    },
    {
        "id": "VT-008",
        "title": "Thermal Management for Electric Vehicle Battery Packs",
        "description": (
            "Model heat generation and cooling strategies for lithium-ion "
            "battery modules. Includes finite element thermal simulation, "
            "experimental validation on instrumented test benches and design "
            "optimization of cold plates."
        ),
        "required_skills": ["MATLAB", "Finite Element Analysis", "Thermodynamics", "Mechanical Design"],
        "eligibility": "Mechanical Engineering, Chemical Engineering, Junior+",
    },
]


# =============================================================================
# 3. MOCK DATA: SHORT, VAGUE UNDERGRADUATE PROFILES (QUERIES)
# =============================================================================
# Deliberately short and underspecified - this is the hard case we care about.
# Each is a 1-2 sentence self-description of skill level, major and interest.

TEST_PROFILES: List[Dict[str, str]] = [
    {
        "profile_id": "P1",
        "major": "Computer Science",
        "year": "Sophomore",
        "query": "I like machine learning and I know some Python. I want to work on AI stuff.",
    },
    {
        "profile_id": "P2",
        "major": "Biology",
        "year": "Junior",
        "query": "Biology major interested in data and health. I don't code very much.",
    },
    {
        "profile_id": "P3",
        "major": "Mechanical Engineering",
        "year": "Junior",
        "query": "I have built robots before, I like hardware and control systems.",
    },
    {
        "profile_id": "P4",
        "major": "Psychology",
        "year": "Sophomore",
        "query": "I am an HCI and design person, I like studying people and technology.",
    },
]


# =============================================================================
# 4. TEXT NORMALIZATION / TOKENIZATION
# =============================================================================
# The SAME tokenizer is applied to documents and queries. Lowercasing and
# stop-word removal are *normalization*, not expansion - they are applied
# symmetrically so the three methods still see identical inputs.

# Keeps "c++", "c#", "cs1", "3d" style tokens intact.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\+\+|#)?")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "i", "in",
    "into", "is", "it", "like", "not", "of", "on", "or", "some", "that", "the",
    "their", "to", "very", "want", "with", "work", "working", "we", "our",
}


def tokenize(text: str) -> List[str]:
    """
    Lowercase, regex-tokenize and drop stop-words.

    Symmetric by design: documents and queries pass through this identical
    function, so no method gains information the others lack.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def build_document(opportunity: Dict[str, object], cfg: Config) -> str:
    """
    Flatten one opportunity into the single searchable text blob used by all
    retrieval methods (BM25 and the encoder must consume identical text).

    `eligibility` is intentionally excluded - see `Config` baseline constraints.
    It stays in the dataframe as metadata for the (disabled) filtering stage.
    """
    parts = [
        str(opportunity["title"]),
        str(opportunity["description"]),
        "Skills: " + ", ".join(opportunity["required_skills"]),  # type: ignore[arg-type]
    ]
    if cfg.include_eligibility_in_text:  # baseline switch, default False
        parts.append("Eligibility: " + str(opportunity["eligibility"]))
    return " ".join(parts)


# =============================================================================
# 5. GENERIC RANKING HELPERS
# =============================================================================


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max scale a score vector into [0, 1].

    Degenerate case: if all scores are (near) identical, or the input contains
    NaN/inf, we return an all-zero vector rather than dividing by ~0. This is
    important because a query can legitimately produce all-zero BM25 scores
    (no lexicon overlap) - in that case the hybrid must fall back cleanly to the
    semantic signal instead of exploding.
    """
    s = np.asarray(scores, dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def rank_scores(
    opportunity_ids: Sequence[str],
    scores: Sequence[float],
    top_k: int,
) -> List[Tuple[str, int, float]]:
    """
    Deterministically sort candidates and return the Top-k as
    `(opportunity_id, rank, score)` with rank starting at 1.

    Ties are broken by `opportunity_id` so repeated runs are byte-identical
    (required for a reproducible evaluation harness).
    """
    order = sorted(
        zip(opportunity_ids, np.asarray(scores, dtype=float)),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [(opp_id, rank, float(score)) for rank, (opp_id, score) in enumerate(order[:top_k], start=1)]


# =============================================================================
# 6. METHOD 1 - KEYWORD SEARCH (BM25)
# =============================================================================


class NonNegativeBM25(BM25Okapi):
    """
    `rank_bm25.BM25Okapi` with negative IDFs floored at 0.

    Why: on a tiny corpus (n=8 here, n=10-50 in the pilot), a term appearing in
    most documents gets a *negative* IDF. Okapi then produces negative document
    scores, which (a) are hard to interpret and (b) break the [0,1] min-max
    assumption used by the hybrid. Flooring negative IDFs at 0 is the standard
    fix and leaves normal-collection behaviour unchanged.
    """

    def _calc_idf(self, nd) -> None:  # noqa: D102 - inherited behaviour
        super()._calc_idf(nd)
        self.idf = {term: max(0.0, value) for term, value in self.idf.items()}


class KeywordSearcher:
    """Method 1: BM25 keyword retrieval (lexical, exact-term driven)."""

    def __init__(self, documents: Sequence[str], cfg: Config) -> None:
        self.cfg = cfg
        self.tokenized_corpus = [tokenize(doc) for doc in documents]
        # Guard: BM25 assumes every document has at least one token.
        self.tokenized_corpus = [toks or ["<empty>"] for toks in self.tokenized_corpus]
        self.bm25 = NonNegativeBM25(self.tokenized_corpus, k1=cfg.bm25_k1, b=cfg.bm25_b)

    def score(self, query: str) -> np.ndarray:
        """Return a raw BM25 score per document for this query."""
        # `query_expansion=False` -> the query is tokenized as-is, verbatim.
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)


# =============================================================================
# 7. METHOD 2 - SEMANTIC SEARCH (all-MiniLM-L6-v2 + cosine similarity)
# =============================================================================


class HashedBoWEncoder:
    """
    Dependency-free fallback encoder (the "hashing trick").

    This exists ONLY so the harness still runs on machines where
    `sentence-transformers` / torch cannot be installed or downloaded (offline
    CI, restricted cluster login nodes, Python 3.13 wheels not yet published).

    It is a bag-of-words hash projection - it is NOT a semantic model and will
    score near-lexical. Any result produced with this fallback must be labelled
    as such in the write-up; it is never a substitute for MiniLM.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(
        self,
        texts: Sequence[str],
        normalize_embeddings: bool = True,   # accepted for API compatibility
        convert_to_numpy: bool = True,       # accepted for API compatibility
        show_progress_bar: bool = False,     # accepted for API compatibility
        **_: object,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
                vectors[row, int(digest, 16) % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


def load_encoder(model_name: str, allow_fallback: bool = True):
    """
    Load the sentence encoder, returning `(encoder, backend_name)`.

    `backend_name` is propagated into the report so every exported artefact
    records whether embeddings were real (MiniLM) or the offline fallback.
    """
    try:
        from sentence_transformers import SentenceTransformer  # imported lazily

        encoder = SentenceTransformer(model_name)
        return encoder, f"sentence-transformers::{model_name}"
    except Exception as exc:  # ImportError, download failure, OOM, ...
        if not allow_fallback:
            raise
        print(
            f"[warn] Could not load '{model_name}' ({type(exc).__name__}: {exc}).\n"
            "[warn] Falling back to the offline HashedBoW encoder - semantic scores\n"
            "[warn] in this run are lexical approximations, NOT MiniLM embeddings.\n"
        )
        return HashedBoWEncoder(), "offline-fallback::hashed-bow"


class SemanticSearcher:
    """Method 2: dense embedding retrieval with cosine similarity."""

    def __init__(self, documents: Sequence[str], encoder, cfg: Config) -> None:
        self.cfg = cfg
        self.encoder = encoder
        # L2-normalised document embeddings -> cosine == dot product.
        self.doc_embeddings = np.asarray(
            encoder.encode(
                list(documents),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )

    def score(self, query: str) -> np.ndarray:
        """Return cosine similarity in [-1, 1] per document for this query."""
        query_embedding = np.asarray(
            self.encoder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )[0]
        # Both sides are unit vectors, so the dot product IS cosine similarity.
        return self.doc_embeddings @ query_embedding


# =============================================================================
# 8. METHOD 3 - KEYWORD-PRIORITIZED HYBRID
# =============================================================================


def hybrid_scores(
    bm25_raw: np.ndarray,
    semantic_raw: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine lexical and semantic signals into a single ranking score.

    HOW RANKINGS ARE COMBINED (spec section 3)
    ------------------------------------------
    BM25 scores are unbounded and corpus-dependent (roughly 0..15 here), while
    cosine similarities live in a narrow band (roughly 0.1..0.7). Averaging the
    raw values would let BM25's scale dominate by accident rather than by
    design. So both signals are first put on a comparable scale:

        1. NORMALIZE (per query, across all candidate documents):
               bm25_norm     = (bm25_raw     - min) / (max - min)
               semantic_norm = (semantic_raw - min) / (max - min)
           Each series now spans exactly [0, 1]; the best document for a signal
           gets 1.0 and the worst gets 0.0.

        2. WEIGHTED AVERAGE (keyword priority, default 0.6 / 0.4):
               final = 0.6 * bm25_norm + 0.4 * semantic_norm

        3. RANK by `final` descending (ties broken by opportunity id).

    Consequences worth documenting in the write-up:
      * The hybrid is **rank-fusion by score scaling**, so it inherits BM25's
        brittleness to vocabulary mismatch: a purely semantic hit that BM25
        cannot see still receives bm25_norm = 0 and keeps only 40% of its
        signal. This is the intended "keyword-prioritized" behaviour, and the
        planned weight-sweep ablation (bm25_weight in 0.0..1.0) quantifies it.
      * Normalization is per query, so scores are comparable *within* a query
        but not across queries.

    Returns `(final, bm25_norm, semantic_norm)`.
    """
    bm25_norm = min_max_normalize(bm25_raw)
    semantic_norm = min_max_normalize(semantic_raw)
    final = cfg.bm25_weight * bm25_norm + cfg.semantic_weight * semantic_norm
    return final, bm25_norm, semantic_norm


# =============================================================================
# 9. RUN THE PIPELINE OVER EVERY PROFILE x EVERY METHOD
# =============================================================================

METHOD_KEYWORD = "Keyword (BM25)"


def method_labels(cfg: Config, backend_name: str) -> Tuple[str, str, str]:
    """
    Build the three method labels for this run.

    Labels are derived from the ACTUAL configuration so the exported CSVs and
    the report can never misdescribe the run:
      * the semantic label says so when embeddings came from the offline proxy
        instead of MiniLM;
      * the hybrid label prints the real weights after a `--bm25-weight` override.
    """
    if "hashed" in backend_name.lower():
        semantic_label = "Semantic (HashedBoW proxy)"
    else:
        semantic_label = f"Semantic ({cfg.embedding_model})"
    hybrid_label = f"Hybrid ({cfg.bm25_weight:.1f} KW / {cfg.semantic_weight:.1f} SEM)"
    return METHOD_KEYWORD, semantic_label, hybrid_label


def method_order(results: pd.DataFrame) -> List[str]:
    """Method labels in first-seen (i.e. Keyword, Semantic, Hybrid) order."""
    return list(dict.fromkeys(results["method"].tolist()))


def run_pipeline(cfg: Config, encoder, backend_name: str) -> pd.DataFrame:
    """
    Execute all three retrievers for all profiles and return the long results
    table with columns:
        [profile_id, method, opportunity_id, rank, score, title, description]
    """
    opportunity_ids = [str(o["id"]) for o in OPPORTUNITIES]
    documents = [build_document(o, cfg) for o in OPPORTUNITIES]

    keyword = KeywordSearcher(documents, cfg)
    semantic = SemanticSearcher(documents, encoder, cfg)
    label_keyword, label_semantic, label_hybrid = method_labels(cfg, backend_name)

    lookup = {str(o["id"]): o for o in OPPORTUNITIES}
    rows: List[Dict[str, object]] = []

    for profile in TEST_PROFILES:
        profile_id, query = profile["profile_id"], profile["query"]

        bm25_raw = keyword.score(query)
        semantic_raw = semantic.score(query)
        hybrid_raw, _, _ = hybrid_scores(bm25_raw, semantic_raw, cfg)

        for method, scores in (
            (label_keyword, bm25_raw),
            (label_semantic, semantic_raw),
            (label_hybrid, hybrid_raw),
        ):
            for opp_id, rank, score in rank_scores(opportunity_ids, scores, cfg.top_k):
                opportunity = lookup[opp_id]
                rows.append(
                    {
                        "profile_id": profile_id,
                        "method": method,
                        "opportunity_id": opp_id,
                        "rank": rank,
                        "score": round(score, 6),
                        "title": opportunity["title"],
                        "description": opportunity["description"],
                    }
                )

    results = pd.DataFrame(rows)
    results.attrs["encoder_backend"] = backend_name
    return results


# =============================================================================
# 10. MANUAL RELEVANCE LABELS (ground truth for the prototype)
# =============================================================================
# Grading rubric (per spec):
#     0 = Unsuitable      -> topic/skills clearly unrelated to the profile
#     1 = Partial Match   -> adjacent topic or transferable skills; would need
#                            ramp-up or co-mentoring
#     2 = Strong Match    -> profile's stated interests and skills line up with
#                            the listing's core work
#
# IMPORTANT: labels are assigned from listing CONTENT vs profile CONTENT only.
# `eligibility` is deliberately ignored (metadata filtering is OFF), so a
# Sophomore can legitimately be labelled 2 for a "Junior+" listing. This keeps
# the label set aligned with what the retrievers can actually observe.
#
# Labels are written as profile -> grades in the SAME ORDER as OPPORTUNITIES
# so they are easy to audit, then expanded into a tidy (profile, opportunity)
# mapping by `build_relevance_map()`.

LABEL_ORDER_NOTE = "grades follow the order of OPPORTUNITIES (VT-001 .. VT-008)"

RELEVANCE_LABELS: Dict[str, List[int]] = {
    # P1: ML + Python, wants AI.
    #     Strong on the ML/AI listings (001, 006, 007); partial where ML is a
    #     supporting tool (002, 005); unsuitable for biology/mechanical work.
    "P1": [2, 1, 0, 0, 1, 2, 2, 0],
    # P2: biology/data/health, minimal coding.
    #     Strong only on genomics (004); partial on applied data work (002) and
    #     the health-adjacent human study (005).
    "P2": [0, 1, 0, 2, 1, 0, 0, 0],
    # P3: robotics/hardware/control.
    #     Strong on HRI robotics (003); partial on autonomous driving (001),
    #     drone swarms (006) and the mechanical/thermal build (008).
    "P3": [1, 0, 2, 0, 0, 1, 0, 1],
    # P4: HCI/design, people + technology.
    #     Strong on usable-security UX (005); partial on human-robot
    #     interaction (003) and the student-facing tutor (007).
    "P4": [0, 0, 1, 0, 2, 0, 1, 0],
}


def build_relevance_map() -> Dict[Tuple[str, str], int]:
    """Expand the compact label matrix into {(profile_id, opportunity_id): grade}."""
    mapping: Dict[Tuple[str, str], int] = {}
    for profile_id, grades in RELEVANCE_LABELS.items():
        if len(grades) != len(OPPORTUNITIES):
            raise ValueError(
                f"Profile {profile_id} has {len(grades)} labels but there are "
                f"{len(OPPORTUNITIES)} opportunities ({LABEL_ORDER_NOTE})."
            )
        for opportunity, grade in zip(OPPORTUNITIES, grades):
            mapping[(profile_id, str(opportunity["id"]))] = int(grade)
    return mapping


def relevant_items(profile_id: str, threshold: int = 1) -> List[str]:
    """All opportunity ids whose grade >= `threshold` (default: Partial or better)."""
    return [
        str(opportunity["id"])
        for opportunity, grade in zip(OPPORTUNITIES, RELEVANCE_LABELS[profile_id])
        if grade >= threshold
    ]


# =============================================================================
# 11. RANKING METRICS
# =============================================================================


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """
    Discounted Cumulative Gain with the *graded* gain formulation

        DCG@k = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)

    Using 2^rel - 1 (rather than rel) rewards a Strong Match (2 -> 3.0) much
    more than a Partial Match (1 -> 1.0), which is what we want: a method that
    surfaces the single best-fitting lab should beat one that returns three
    lukewarm options.
    """
    gains = list(gains)[:k]
    return float(
        sum((2.0 ** grade - 1.0) / np.log2(position + 1.0) for position, grade in enumerate(gains, start=1))
    )


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Dict[str, int], k: int) -> float:
    """
    nDCG@k (PRIMARY METRIC).

    Normalises DCG@k by the DCG of the *ideal* ordering of the ground truth, so
    the score is comparable across profiles that have different numbers of
    relevant listings. Range [0, 1]; 1.0 means the retriever reproduced the
    perfect ranking.
    """
    ranked = list(ranked_ids)[:k]
    dcg = dcg_at_k([relevance.get(opp_id, 0) for opp_id in ranked], k)
    ideal = sorted(relevance.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    ranked_ids: Sequence[str],
    relevance: Dict[str, int],
    k: int,
    threshold: int = 1,
) -> float:
    """
    Precision@k with BINARY relevance (grade >= `threshold`, default Partial+).

    Answers: "of the k listings we showed the student, what fraction are worth
    clicking?" Denominator is always k (standard, and penalises short lists).
    Use `threshold=2` for a strict "only Strong Matches count" variant.
    """
    ranked = list(ranked_ids)[:k]
    if not ranked:
        return 0.0
    hits = sum(1 for opp_id in ranked if relevance.get(opp_id, 0) >= threshold)
    return hits / float(k)


def recall_at_k(
    ranked_ids: Sequence[str],
    relevance: Dict[str, int],
    k: int,
    threshold: int = 1,
) -> float:
    """
    Recall@k with BINARY relevance (grade >= `threshold`, default Partial+).

    Answers: "did we at least put every plausible listing in front of the
    student?" Denominator is the number of relevant listings in the ground
    truth, which is why the label matrix must cover the WHOLE corpus (not just
    the retrieved subset) - otherwise recall is silently inflated.
    """
    ranked = list(ranked_ids)[:k]
    n_relevant = sum(1 for grade in relevance.values() if grade >= threshold)
    if n_relevant == 0:
        return 0.0
    hits = len({opp_id for opp_id in ranked if relevance.get(opp_id, 0) >= threshold})
    return hits / float(n_relevant)


def evaluate(
    results: pd.DataFrame,
    relevance_map: Dict[Tuple[str, str], int],
    k: int,
    threshold: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score every (profile, method) pair in `results`.

    Returns
    -------
    per_profile : one row per profile x method with nDCG@k / P@k / R@k
    summary_df  : mean metrics per method (the headline comparison table)
    """
    per_profile_rows: List[Dict[str, object]] = []

    for (profile_id, method), group in results.groupby(["profile_id", "method"], sort=True):
        relevance = {
            opp_id: grade
            for (pid, opp_id), grade in relevance_map.items()
            if pid == profile_id
        }
        ranked_ids = list(group.sort_values("rank")["opportunity_id"])
        per_profile_rows.append(
            {
                "profile_id": profile_id,
                "method": method,
                "ndcg@k": round(ndcg_at_k(ranked_ids, relevance, k), 4),
                "precision@k": round(precision_at_k(ranked_ids, relevance, k, threshold), 4),
                "recall@k": round(recall_at_k(ranked_ids, relevance, k, threshold), 4),
                "n_relevant": sum(1 for g in relevance.values() if g >= threshold),
                "n_strong": sum(1 for g in relevance.values() if g == 2),
            }
        )

    per_profile = pd.DataFrame(per_profile_rows)

    summary_df = (
        per_profile.groupby("method", sort=False)[["ndcg@k", "precision@k", "recall@k"]]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"ndcg@k": "mean_ndcg@k", "precision@k": "mean_precision@k", "recall@k": "mean_recall@k"})
    )
    # Present methods in retrieval order (Keyword, Semantic, Hybrid) rather than
    # alphabetically. `sort=False` above already preserves first-seen order, so
    # this is just a defensive re-sort against future changes.
    order = method_order(results)
    summary_df["__order"] = summary_df["method"].map({m: i for i, m in enumerate(order)})
    summary_df = summary_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return per_profile, summary_df


# =============================================================================
# 12. REPORTING
# =============================================================================


def print_report(
    cfg: Config,
    results: pd.DataFrame,
    per_profile: pd.DataFrame,
    summary_df: pd.DataFrame,
    backend_name: str,
) -> None:
    """Human-readable console report comparing the three methods."""
    rule = "=" * 78

    print(f"\n{rule}\nVTURCS Team 1 - IRDM Prototype: Retrieval Method Comparison\n{rule}")
    print(f"Corpus          : {len(OPPORTUNITIES)} opportunity listings")
    print(f"Profiles        : {len(TEST_PROFILES)} short/vague student queries")
    print(f"Cut-off k       : {cfg.top_k}")
    print(f"Hybrid weights  : {cfg.bm25_weight:.2f} keyword / {cfg.semantic_weight:.2f} semantic")
    print(f"Encoder backend : {backend_name}")
    print(
        "Baselines OFF   : "
        f"metadata_filtering={cfg.metadata_filtering}, "
        f"query_expansion={cfg.query_expansion}, "
        f"eligibility_in_text={cfg.include_eligibility_in_text}"
    )

    # --- per-profile top-1, the most tangible view for a demo ---------------
    print(f"\n{'-' * 78}\nTop-1 recommendation per profile / method\n{'-' * 78}")
    order = method_order(results)
    top1 = results[results["rank"] == 1].pivot(index="profile_id", columns="method", values="title")
    print(top1.reindex(columns=order).fillna("-").to_string())

    # --- per-profile metrics -------------------------------------------------
    print(f"\n{'-' * 78}\nPer-profile metrics (graded nDCG; Precision/Recall count Partial+)\n{'-' * 78}")
    print(per_profile.to_string(index=False))

    # --- headline comparison -------------------------------------------------
    print(f"\n{'-' * 78}\nMEAN PERFORMANCE ACROSS PROFILES\n{'-' * 78}")
    print(summary_df.to_string(index=False))

    # --- winner summary ------------------------------------------------------
    print(f"\n{'-' * 78}\nInterpretation\n{'-' * 78}")
    best_by_ndcg = summary_df.loc[summary_df["mean_ndcg@k"].idxmax()]
    best_by_recall = summary_df.loc[summary_df["mean_recall@k"].idxmax()]
    print(f"Best mean nDCG@{cfg.top_k}  : {best_by_ndcg['method']} ({best_by_ndcg['mean_ndcg@k']:.4f})")
    print(f"Best mean Recall@{cfg.top_k}: {best_by_recall['method']} ({best_by_recall['mean_recall@k']:.4f})")

    pivoted = per_profile.pivot(index="profile_id", columns="method", values="ndcg@k")
    wins = pivoted.reindex(columns=order)
    print("\nPer-profile nDCG winner:")
    for profile_id in wins.index:
        row = wins.loc[profile_id].astype(float)
        print(f"  {profile_id}: {row.idxmax():<28} (nDCG@{cfg.top_k} = {row.max():.4f})")

    print(
        "\nNote: this prototype uses a synthetic corpus and hand-written labels.\n"
        "Absolute values are not the finding - the *relative ordering* of methods\n"
        "and the failure cases are what the pilot is meant to surface.\n"
    )


# =============================================================================
# 13. ENTRY POINT
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VTURCS Team 1 - IRDM matching prototype")
    parser.add_argument("--top-k", type=int, default=Config.top_k, help="cut-off k for retrieval and metrics")
    parser.add_argument("--bm25-weight", type=float, default=Config.bm25_weight, help="hybrid weight on BM25 (0..1)")
    parser.add_argument(
        "--offline-embeddings",
        action="store_true",
        help="never load sentence-transformers; use the hashed bag-of-words fallback",
    )
    parser.add_argument("--output-dir", type=str, default=Config.output_dir, help="directory for CSV exports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = Config(
        top_k=args.top_k,
        bm25_weight=args.bm25_weight,
        semantic_weight=round(1.0 - args.bm25_weight, 10),
        output_dir=args.output_dir,
    )

    # --- retrieve ------------------------------------------------------------
    if args.offline_embeddings:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        encoder, backend_name = HashedBoWEncoder(), "offline-forced::hashed-bow"
        print("[info] --offline-embeddings set: using HashedBoW encoder (lexical proxy).")
    else:
        encoder, backend_name = load_encoder(cfg.embedding_model, allow_fallback=True)

    results = run_pipeline(cfg, encoder, backend_name)

    # --- evaluate ------------------------------------------------------------
    relevance_map = build_relevance_map()
    per_profile, summary_df = evaluate(results, relevance_map, k=cfg.top_k, threshold=1)

    # --- export --------------------------------------------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "match_results.csv", index=False)
    per_profile.to_csv(out_dir / "evaluation_per_profile.csv", index=False)
    summary_df.to_csv(out_dir / "evaluation_summary.csv", index=False)

    # --- report --------------------------------------------------------------
    print_report(cfg, results, per_profile, summary_df, backend_name)
    print(f"Exports written to: {out_dir.resolve()}")
    print("  - match_results.csv")
    print("  - evaluation_per_profile.csv")
    print("  - evaluation_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
