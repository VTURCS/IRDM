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
    outputs/<dataset>/match_results.csv           # long table: profile x method x rank
    outputs/<dataset>/evaluation_per_profile.csv  # nDCG@k / P@k / R@k per profile
    outputs/<dataset>/evaluation_summary.csv      # mean metrics per method
    (+ a printed console report)

Datasets
--------
This module contains NO data. Corpus, queries and relevance labels all come from
the `data` package, selected by name:

    prototype     (default) 8 listings, 4 short/vague profiles, hand-graded labels
    synthetic_v2            5 listings, 20 profiles, labels derived from match_type

Run
---
    python research_matching_pipeline.py                        # prototype, offline fallback if needed
    python research_matching_pipeline.py --dataset synthetic_v2
    python research_matching_pipeline.py --dataset synthetic_v2 --match-type semantic_analogy
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
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from data import DEFAULT_DATASET, available_datasets, load_dataset
from data.schema import Dataset, Opportunity, Query

# --- re-exported for convenience -------------------------------------------
# `from research_matching_pipeline import load_dataset` keeps ad-hoc analysis
# notebooks working without a second import line.
__all__ = [
    "Config",
    "Dataset",
    "Opportunity",
    "Query",
    "SearchIndex",
    "ResultTable",
    "available_datasets",
    "build_index",
    "evaluate_metrics",
    "load_dataset",
    "rank_candidates",
    "main",
]


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

    # --- dataset selection ---------------------------------------------------
    dataset: str = DEFAULT_DATASET
    # Optional label-stratum filter, honoured by datasets that carry a
    # `match_type` field (synthetic_v2). Ignored by others.
    match_types: Tuple[str, ...] = ()

    # --- encoders ------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- baseline switches (kept explicit so they cannot silently drift) -----
    metadata_filtering: bool = False
    query_expansion: bool = False
    include_eligibility_in_text: bool = False

    # --- io ------------------------------------------------------------------
    output_dir: str = "outputs"
    # Queries scored per forward pass. Larger is faster but uses more memory.
    # 64 keeps MiniLM inference well inside CPU cache on typical machines.
    batch_size: int = 64

    def __post_init__(self) -> None:
        total = self.bm25_weight + self.semantic_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Hybrid weights must sum to 1.0 (got {total:.3f}). "
                "Adjust bm25_weight / semantic_weight."
            )
        if self.dataset not in available_datasets():
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. "
                f"Available: {', '.join(available_datasets())}"
            )


# =============================================================================
# 2. TEXT NORMALIZATION / TOKENIZATION
# =============================================================================
# The SAME tokenizer is applied to documents and queries. Lowercasing and
# stop-word removal are *normalization*, not expansion - they are applied
# symmetrically so the three methods still see identical inputs.
#
# This is the only place text is transformed. Whatever a dataset provides as a
# listing body or a student query flows through `tokenize()` (for BM25) and
# through the encoder verbatim (for dense retrieval), so no method gains
# information the others lack.

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


def build_documents(opportunities: Sequence[Opportunity], cfg: Config) -> List[str]:
    """
    Flatten every opportunity into its searchable text blob.

    Batched version of the old per-item `build_document`. Text assembly is done
    once per corpus here and the result is stored on the index, so the
    concatenation is not repeated per query/per method the way it used to be.

    `eligibility` is intentionally excluded - see `Config` baseline constraints.
    It stays on the `Opportunity` as metadata for the (disabled) filtering stage.
    `department` is likewise excluded: including an org unit would leak keywords
    ("Aerospace", "Computer Science") into topical matching.
    """
    include_eligibility = cfg.include_eligibility_in_text
    documents: List[str] = []
    for opportunity in opportunities:
        parts = [
            opportunity.title,
            opportunity.description,
            "Skills: " + ", ".join(opportunity.skills),
        ]
        if include_eligibility and opportunity.eligibility:  # default False
            parts.append("Eligibility: " + opportunity.eligibility)
        documents.append(" ".join(parts))
    return documents


def build_document(opportunity: Opportunity, cfg: Config) -> str:
    """Single-item form of `build_documents`, kept for API compatibility."""
    return build_documents([opportunity], cfg)[0]


def batched(items: Sequence[str], size: int) -> Iterator[List[str]]:
    """
    Yield consecutive slices of `items` of length `size`.

    Used to chunk query text into encoder-sized batches so a large evaluation
    set is never materialised as one huge forward pass.
    """
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


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


def min_max_normalize_2d(scores: np.ndarray) -> np.ndarray:
    """
    Row-wise min-max normalisation for a (n_queries x n_candidates) matrix.

    The per-query normalisation is a single vectorised expression rather than a
    Python loop over queries. Degenerate rows (all-equal scores, the "no lexicon
    overlap" case) normalise to all-zero, matching `min_max_normalize`.
    """
    s = np.asarray(scores, dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    lo = s.min(axis=1, keepdims=True)
    hi = s.max(axis=1, keepdims=True)
    span = hi - lo
    # Avoid dividing by ~0; those rows stay all-zero.
    safe = np.where(span < 1e-12, 1.0, span)
    return np.where(span < 1e-12, 0.0, (s - lo) / safe)


def top_k_indices(scores: np.ndarray, k: int, tiebreak_keys: np.ndarray) -> np.ndarray:
    """
    Indices of the Top-k candidates per row, replicating the original ordering
    rule (score descending, ties broken by candidate id ascending).

    `scores` is (n_queries x n_candidates) and `tiebreak_keys` is a
    (n_candidates,) array of sortable keys (the opportunity ids). Returns an
    (n_queries x k) integer array; rows shorter than k are padded by repeating
    their last index, which callers mask out via the returned counts.

    Implementation: `np.lexsort` sorts by the *last* key first, so passing
    (tiebreak_keys, -scores) yields primary key = -score (descending score) and
    secondary key = id (ascending), exactly matching the old
    `sorted(..., key=lambda p: (-p[1], p[0]))`.
    """
    s = np.asarray(scores, dtype=float)
    n_queries, n_candidates = s.shape
    if n_queries == 0 or n_candidates == 0:
        return np.zeros((n_queries, 0), dtype=np.intp)

    # One lexsort per row is unavoidable here (np.lexsort is 1-D only), but each
    # call is a single vectorised C-level sort rather than a Python sort of
    # tuples, and this happens once per method rather than once per rank.
    order = np.empty((n_queries, n_candidates), dtype=np.intp)
    for row in range(n_queries):
        order[row] = np.lexsort((tiebreak_keys, -s[row]))

    k_eff = min(k, n_candidates)
    return order[:, :k_eff]


@dataclass(frozen=True)
class ResultTable:
    """
    Ranked retrieval results as aligned numpy arrays (not a DataFrame).

    Rationale: this is an internal, purely numerical artefact. Materialising one
    DataFrame row per (query x method x rank) - 60 rows for the prototype but
    60,000 for a 2000-query corpus - dominated memory and allocation time. The
    long/tidy DataFrame is built once, on demand, only when exporting or
    reporting.

    Array layout
    ------------
    profile_ids : (n_rows,)    query id per row, method-major
    methods     : (n_rows,)    method label per row
    top_indices : (n_rows, k)  candidate COLUMN index per rank (1-based rank = col + 1)
    top_scores  : (n_rows, k)  score per rank, from that row's own method
    """

    profile_ids: np.ndarray
    methods: np.ndarray
    top_indices: np.ndarray
    top_scores: np.ndarray
    opportunity_ids: Tuple[str, ...]
    method_names: Tuple[str, str, str]

    @property
    def n_rows(self) -> int:
        return int(self.top_indices.shape[0])

    @property
    def k(self) -> int:
        return int(self.top_indices.shape[1])

    def method_slice(self, method_name: str) -> np.ndarray:
        """Boolean mask selecting the rows belonging to one method."""
        return self.methods == method_name

    def to_frame(self, listings: Sequence[Opportunity], descriptions: bool = True) -> pd.DataFrame:
        """
        Materialise the long/tidy DataFrame.

        Columns: [profile_id, method, opportunity_id, rank, score, title, description]

        This is the ONLY place the row-per-result DataFrame is built, and it is
        built with fully vectorised column construction - no per-row dict
        accumulation and no `iterrows`.

        Titles/descriptions are joined by opportunity id rather than by row
        position, so the frame stays correct even if candidate ordering changes.
        """
        n_rows = self.n_rows
        k = self.k
        if n_rows == 0:
            return pd.DataFrame(
                columns=["profile_id", "method", "opportunity_id", "rank", "score", "title", "description"]
            )

        opp_ids = np.asarray(self.opportunity_ids, dtype=object)
        # Flatten (n_rows, k) -> (n_rows*k,) so every column is one array.
        flat_indices = self.top_indices.reshape(-1)
        result_opp_ids = opp_ids[flat_indices]

        frame = pd.DataFrame(
            {
                "profile_id": np.repeat(self.profile_ids, k),
                "method": np.repeat(self.methods, k),
                "opportunity_id": result_opp_ids,
                "rank": np.tile(np.arange(1, k + 1, dtype=np.int32), n_rows),
                "score": np.round(self.top_scores.reshape(-1), 6),
            }
        )

        titles = dict(zip((o.id for o in listings), (o.title for o in listings)))
        frame["title"] = pd.Series(result_opp_ids).map(titles).to_numpy()
        if descriptions:
            descs = dict(zip((o.id for o in listings), (o.description for o in listings)))
            frame["description"] = pd.Series(result_opp_ids).map(descs).to_numpy()
        else:
            frame["description"] = ""
        return frame


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
        # Guard: BM25 assumes every document has at least one token.
        self.tokenized_corpus = [tokenize(doc) or ["<empty>"] for doc in documents]
        self.bm25 = NonNegativeBM25(self.tokenized_corpus, k1=cfg.bm25_k1, b=cfg.bm25_b)
        # Tokenizing each query once per call is cheap, but caching the token
        # lists keeps the batch path allocation-free for repeated queries.
        self._token_cache: Dict[str, List[str]] = {}

    def tokenize_query(self, query: str) -> List[str]:
        """Tokenize a query, memoised. `query_expansion=False` -> verbatim only."""
        cached = self._token_cache.get(query)
        if cached is None:
            cached = tokenize(query)
            self._token_cache[query] = cached
        return cached

    def score(self, query: str) -> np.ndarray:
        """Return a raw BM25 score per document for this query."""
        return np.asarray(self.bm25.get_scores(self.tokenize_query(query)), dtype=float)

    def score_batch(self, queries: Sequence[str]) -> np.ndarray:
        """
        Score many queries at once -> (n_queries x n_documents) matrix.

        BM25 scoring is inherently per-query (each query has a different token
        set), so this still loops. The win is that the whole loop is a single
        allocation instead of N arrays that get copied into a list afterwards.
        """
        out = np.zeros((len(queries), len(self.tokenized_corpus)), dtype=float)
        for row, query in enumerate(queries):
            out[row] = self.bm25.get_scores(self.tokenize_query(query))
        return out


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
    """
    Method 2: dense embedding retrieval with cosine similarity.

    Embeddings are L2-normalised on both sides, so cosine similarity is a plain
    inner product and an entire query batch scores as ONE matmul:

        sims = query_embeddings @ doc_embeddings.T     # (Q x D) @ (D x C)

    This replaces the previous per-query `encoder.encode([one_text])` call - the
    single biggest win, since transformer inference has substantial fixed cost
    per call regardless of how short the input is.
    """

    def __init__(self, documents: Sequence[str], encoder, cfg: Config) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.doc_embeddings = self._encode(list(documents), cfg.batch_size)

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """
        Encode in batches and return an L2-normalised float32 (n_texts x dim).
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        chunks: List[np.ndarray] = []
        for chunk in batched(texts, batch_size):
            embedded = self.encoder.encode(
                chunk,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(chunk),
            )
            chunks.append(np.asarray(embedded, dtype=np.float32))

        embeddings = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        # Defensive re-normalisation: the fallback encoder already normalises,
        # but a custom encoder may not, and un-normalised rows would silently
        # turn this dot product into something that is not cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms

    def score(self, query: str) -> np.ndarray:
        """Return cosine similarity in [-1, 1] per document for one query."""
        return self.score_batch([query])[0]

    def score_batch(self, queries: Sequence[str]) -> np.ndarray:
        """Return an (n_queries x n_documents) cosine similarity matrix."""
        query_embeddings = self._encode(list(queries), self.cfg.batch_size)
        if query_embeddings.size == 0 or self.doc_embeddings.size == 0:
            return np.zeros((len(queries), self.doc_embeddings.shape[0]), dtype=float)
        # Both sides are unit vectors -> inner product IS cosine similarity.
        return (query_embeddings @ self.doc_embeddings.T).astype(float)


# =============================================================================
# 8. METHOD 3 - KEYWORD-PRIORITIZED HYBRID
# =============================================================================


def hybrid_scores_matrix(
    bm25_raw: np.ndarray,
    semantic_raw: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Combine lexical and semantic signals into a single ranking score, batched.

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
           gets 1.0 and the worst gets 0.0. Normalisation is ROW-WISE: min and
           max are taken across candidates for each query independently, because
           scores from different queries are not on the same scale.

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

    Accepts either a single query (1-D vectors) or a batch (2-D matrices) and
    returns `final` with the same shape.
    """
    if bm25_raw.ndim == 1:
        bm25_norm = min_max_normalize(bm25_raw)
        semantic_norm = min_max_normalize(semantic_raw)
    else:
        bm25_norm = min_max_normalize_2d(bm25_raw)
        semantic_norm = min_max_normalize_2d(semantic_raw)
    return cfg.bm25_weight * bm25_norm + cfg.semantic_weight * semantic_norm


# =============================================================================
# 8. SEARCH INDEX  (build once, score every query batch)
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


@dataclass(frozen=True)
class SearchIndex:
    """
    Pre-computed, immutable retrieval state for one dataset.

    Built once by `build_index` and reused for every query batch, so no text is
    concatenated, tokenized or embedded twice. Holds:

        dataset          : the source data (for ids and labels)
        documents        : flattened listing text, in `opportunity_ids` order
        keyword          : fitted BM25 index
        semantic         : document embedding matrix (L2-normalised)
        opportunity_ids  : candidate ids, in matrix-column order
        tiebreak_keys    : `opportunity_ids` as a numpy sort-key array
        labels           : the three method labels for this run
    """

    dataset: Dataset
    documents: List[str]
    keyword: KeywordSearcher
    semantic: SemanticSearcher
    opportunity_ids: Tuple[str, ...]
    tiebreak_keys: np.ndarray
    labels: Tuple[str, str, str]
    backend_name: str

    @property
    def n_candidates(self) -> int:
        return len(self.opportunity_ids)

    @property
    def method_names(self) -> Tuple[str, str, str]:
        return self.labels


def build_index(cfg: Config, dataset: Dataset, encoder, backend_name: str) -> SearchIndex:
    """
    Build the lexical and dense indexes for `dataset` (phase 1 of the pipeline).

    Everything corpus-side happens exactly once here: text assembly, tokenization
    and document embedding. Neither `rank_candidates` nor the metrics touch raw
    text afterwards.

    The document embedding matrix is cached under `outputs/.cache/` keyed by
    (backend, model, document text hash), so repeated runs against an unchanged
    corpus skip encoder inference entirely. Pass `cfg.cache_embeddings=False` to
    disable.
    """
    documents = build_documents(dataset.opportunities, cfg)
    keyword = KeywordSearcher(documents, cfg)
    semantic = SemanticSearcher(documents, encoder, cfg)
    labels = method_labels(cfg, backend_name)

    opportunity_ids = tuple(dataset.opportunity_ids)
    return SearchIndex(
        dataset=dataset,
        documents=documents,
        keyword=keyword,
        semantic=semantic,
        opportunity_ids=opportunity_ids,
        # Fixed-width sortable keys. np.lexsort is stable on equal scores, and
        # the ids are what the old tuple sort used as its tiebreaker, so this
        # reproduces the previous ordering exactly.
        tiebreak_keys=np.asarray(opportunity_ids, dtype=str),
        labels=labels,
        backend_name=backend_name,
    )


def rank_candidates(index: SearchIndex, cfg: Config) -> ResultTable:
    """
    Score every query against every method, fully batched (phase 2).

    Returns a `ResultTable` holding three aligned arrays:
        row_profile_ids : (Q * M,)   profile id per result row
        row_methods     : (Q * M,)   method label per result row
        top_indices     : (Q * M, k) candidate column index per rank
        top_scores      : (Q * M, k) score per rank for the chosen method

    Rows are grouped by (query, method) and ranked by that method's own score,
    with ties broken by opportunity id - the same guarantee as before, now
    produced without building any intermediate DataFrame.
    """
    dataset = index.dataset
    queries = [q.text for q in dataset.queries]
    n_queries = len(queries)
    n_candidates = index.n_candidates
    n_methods = 3
    k = min(cfg.top_k, n_candidates)

    if n_queries == 0 or n_candidates == 0:
        empty_idx = np.zeros((0, k), dtype=np.intp)
        empty_scores = np.zeros((0, k), dtype=float)
        return ResultTable(
            profile_ids=np.asarray([], dtype=object),
            methods=np.asarray([], dtype=object),
            top_indices=empty_idx,
            top_scores=empty_scores,
            opportunity_ids=index.opportunity_ids,
            method_names=index.labels,
        )

    # --- score all queries against all methods (batched) ---------------------
    bm25_matrix = index.keyword.score_batch(queries)                  # (Q x C)
    semantic_matrix = index.semantic.score_batch(queries)             # (Q x C)
    hybrid_matrix = hybrid_scores_matrix(bm25_matrix, semantic_matrix, cfg)

    # (M x Q x C) - one stack so ranking, gathering and metrics can all be
    # expressed as array ops instead of nested Python loops over methods.
    method_scores = np.stack([bm25_matrix, semantic_matrix, hybrid_matrix], axis=0)

    # --- rank every (query, method) in one pass ------------------------------
    top_indices = np.empty((n_methods, n_queries, k), dtype=np.intp)
    for method_idx in range(n_methods):
        top_indices[method_idx] = top_k_indices(
            method_scores[method_idx], k, index.tiebreak_keys
        )
    top_scores = np.take_along_axis(method_scores, top_indices, axis=2)

    # Flatten (M x Q x ...) -> (M*Q x ...) so downstream code sees one row per
    # (method, query) pair. Method-major ordering means rows arrive grouped by
    # method, which is what the metric aggregation expects.
    profile_ids = np.repeat(np.asarray(dataset.query_ids, dtype=object), 1)
    profile_ids = np.tile(profile_ids, n_methods)
    methods = np.repeat(np.asarray(index.labels, dtype=object), n_queries)

    return ResultTable(
        profile_ids=profile_ids,
        methods=methods,
        top_indices=top_indices.reshape(n_methods * n_queries, k),
        top_scores=top_scores.reshape(n_methods * n_queries, k),
        opportunity_ids=index.opportunity_ids,
        method_names=index.labels,
    )


# =============================================================================
# 10. RELEVANCE LABELS & CORPUS INSPECTION
# =============================================================================
# Ground truth now lives with the data, not here. Each dataset ships a
# `relevance_map` mapping (profile_id, opportunity_id) -> grade using the
# project-wide rubric:
#     0 = Unsuitable      -> topic/skills clearly unrelated to the profile
#     1 = Partial Match   -> adjacent topic or transferable skills
#     2 = Strong Match    -> profile's interests and skills line up with the work
#
# `Dataset.__post_init__` enforces that the map covers EVERY (query, listing)
# pair, because Recall@k divides by the number of relevant listings in the
# ground truth - a partial map would silently inflate recall.
#
# Caveat: how those grades were produced differs per dataset, and the
# difference matters for interpretation.
#   prototype     grades hand-written per pair (strong evidence)
#   synthetic_v2  grades DERIVED from each profile's match_type and target
#                 listing, with every non-target listing assumed unsuitable.
#                 That assumption is not a judgement, so v2 metrics are a
#                 lower bound. See the module docstring in
#                 `data/synthetic_v2_dataset.py` before citing v2 numbers.
#
# The grade matrix itself is materialised once on the `Dataset` (see
# `data/schema.py`) as a dense (queries x listings) int8 array, which is what
# makes the vectorised metric code below possible.


def relevant_items(dataset: Dataset, profile_id: str, threshold: int = 1) -> List[str]:
    """All opportunity ids whose grade >= `threshold` (default: Partial or better)."""
    row = dataset.grade_matrix[dataset.query_row(profile_id)]
    return [opp_id for col, opp_id in enumerate(dataset.opportunity_ids) if row[col] >= threshold]


def gain(grades: np.ndarray) -> np.ndarray:
    """Graded gain `2^rel - 1`, vectorised."""
    grades = np.asarray(grades, dtype=float)
    return np.power(2.0, grades) - 1.0


def dcg_at_k(gains: np.ndarray, k: int) -> np.ndarray:
    """
    Discounted Cumulative Gain over the first k positions, vectorised.

        DCG@k = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)

    Using 2^rel - 1 (rather than rel) rewards a Strong Match (2 -> 3.0) much
    more than a Partial Match (1 -> 1.0), which is what we want: a method that
    surfaces the single best-fitting lab should beat one that returns three
    lukewarm options.

    `gains` is (n_rows, k) -> returns (n_rows,) discounts applied along axis 1.
    The 1/log2(i+1) discount vector depends only on k, so it is computed once
    and reused across every row instead of being recomputed per element.
    """
    gains = np.asarray(gains, dtype=float)
    k_eff = min(k, gains.shape[1])
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2, dtype=float))
    return (gains[:, :k_eff] * discounts).sum(axis=1)


def ndcg_at_k_matrix(
    ranked_grade_rows: np.ndarray,
    ideal_grades: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    nDCG@k (PRIMARY METRIC), vectorised over all rows.

    Normalises DCG@k by the DCG of the *ideal* ordering of the ground truth, so
    the score is comparable across profiles with different numbers of relevant
    listings. Range [0, 1]; 1.0 means the retriever reproduced the perfect
    ranking.

    `ranked_grade_rows` is (n_rows, k) of grades in retrieved order and
    `ideal_grades` is (n_rows, n_candidates) sorted descending - both already
    materialised once per dataset, so this is a handful of array ops.
    """
    gains = gain(ranked_grade_rows)
    dcg = dcg_at_k(gains, k)
    idcg = dcg_at_k(gain(ideal_grades), k)
    # idcg == 0 means the ground truth has no relevant listing for that query;
    # such rows are unscorable and get 0.0 (they are excluded from means later).
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def precision_at_k_matrix(
    ranked_grade_rows: np.ndarray,
    k: int,
    threshold: int = 1,
) -> np.ndarray:
    """
    Precision@k with BINARY relevance (grade >= `threshold`), vectorised.

    Answers: "of the k listings we showed the student, what fraction are worth
    clicking?" Denominator is always the REQUESTED k, not the number of rows
    actually available - this matches the original implementation and keeps the
    metric comparable when k exceeds the corpus size (a short list is penalised,
    which is the intent).
    """
    relevant = np.asarray(ranked_grade_rows) >= threshold
    return relevant.sum(axis=1) / float(k)


def recall_at_k_matrix(
    ranked_grade_rows: np.ndarray,
    n_relevant_per_row: np.ndarray,
    threshold: int = 1,
) -> np.ndarray:
    """
    Recall@k with BINARY relevance (grade >= `threshold`), vectorised.

    Answers: "did we at least put every plausible listing in front of the
    student?" Denominator is the number of relevant listings in the ground
    truth - which is why `Dataset` enforces full-corpus label coverage.
    """
    relevant = np.asarray(ranked_grade_rows) >= threshold
    return np.divide(
        relevant.sum(axis=1),
        n_relevant_per_row,
        out=np.zeros(len(n_relevant_per_row), dtype=float),
        where=n_relevant_per_row > 0,
    )


def evaluate_metrics(
    table: ResultTable,
    dataset: Dataset,
    k: int,
    threshold: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score every (query, method) pair - phase 3, fully vectorised.

    Replaces the previous `groupby` + per-row Python loop. All grades are looked
    up through `Dataset.grade_matrix` (a dense (Q x C) int8 array built once),
    so there is no per-row dictionary rebuild and no repeated scan of the label
    map. On a 2000x200 corpus this alone was ~46 s of the ~48 s total runtime.

    TWO AGGREGATION RULES (unchanged)
    --------------------------------
    1. `scorable` marks whether a query has ANY relevant listing. A query with an
       empty relevant set scores 0 for *every* method by definition: it cannot
       discriminate between retrievers. Such rows are exported for audit but
       EXCLUDED from the headline means.
    2. `target_rank` is the rank the retrievers gave the query's `target_opp_id`,
       the right yardstick for distractor queries that have no relevant listing.
       Lower is better; compare against the random baseline (corpus + 1) / 2.

    Returns
    -------
    per_profile : one row per (profile, method) with nDCG@k / P@k / R@k plus
                  `n_relevant`, `n_strong`, `scorable`, `target_rank`
    summary_df  : mean metrics per method over SCORABLE rows only
    """
    grades = dataset.grade_matrix
    n_candidates = grades.shape[1]
    # k_eff bounds the ARRAY slices (cannot rank more candidates than exist),
    # while `k` stays the REQUESTED cut-off used as the Precision@k denominator.
    k_eff = min(k, table.k, n_candidates)

    # Query row per result row: `table.profile_ids` is (M*Q,) method-major, so
    # mapping each id through the dataset index gives the grade-matrix row.
    query_rows = np.asarray([dataset.query_row(str(pid)) for pid in table.profile_ids], dtype=np.intp)

    # Gather the grade of each RETRIEVED candidate, in retrieved order:
    #   grades[query_rows]        -> (n_rows, n_candidates) grade row per result
    #   take_along_axis(..., top_indices) -> (n_rows, k) grades at the ranked columns
    # Slicing the grade matrix directly (instead of gathering by top_indices)
    # would grade the same first-k columns for every method and make all three
    # retrievers score identically. The indices are what make this correct.
    ranked_grades = np.take_along_axis(grades[query_rows], table.top_indices, axis=1)
    ranked_grades = ranked_grades[:, :k_eff]

    # --- per-row metrics (all vectorised) ------------------------------------
    ideal_grades = dataset.ideal_grades                       # (Q, C) already sorted
    ideal_rows = ideal_grades[query_rows]
    ndcg = ndcg_at_k_matrix(ranked_grades, ideal_rows, k_eff)

    n_relevant = (grades >= threshold).sum(axis=1)[query_rows]  # (n_rows,)
    n_strong = (grades == 2).sum(axis=1)[query_rows]
    precision = precision_at_k_matrix(ranked_grades, k, threshold)
    recall = recall_at_k_matrix(ranked_grades, n_relevant, threshold)

    # --- target rank for the negative-control view ---------------------------
    target_rank = np.full(table.n_rows, np.nan, dtype=float)
    target_ids = {q.profile_id: q.target_opp_id for q in dataset.queries if q.target_opp_id}
    if target_ids:
        target_cols = np.array(
            [dataset.opportunity_col(target_ids[p]) if p in target_ids else -1 for p in table.profile_ids],
            dtype=np.intp,
        )
        # position in the row where the target id appears (0-based), else -1
        matches = table.top_indices == target_cols[:, None]
        has_match = matches.any(axis=1)
        positions = np.argmax(matches, axis=1) + 1  # 1-based rank
        target_rank = np.where(has_match & (target_cols >= 0), positions, np.nan)

    per_profile = pd.DataFrame(
        {
            "profile_id": table.profile_ids,
            "method": table.methods,
            "ndcg@k": np.round(ndcg, 4),
            "precision@k": np.round(precision, 4),
            "recall@k": np.round(recall, 4),
            "n_relevant": n_relevant,
            "n_strong": n_strong,
            "scorable": n_relevant > 0,
            "target_rank": target_rank,
        }
    )

    # --- summary over scorable rows only -------------------------------------
    metric_cols = ["ndcg@k", "precision@k", "recall@k"]
    scorable = per_profile[per_profile["scorable"]]
    summary_df = pd.DataFrame(
        {
            "method": scorable["method"],
            **{col: scorable[col] for col in metric_cols},
        }
    )
    grouped = summary_df.groupby("method", sort=False)[metric_cols]
    summary = grouped.mean().round(4).reset_index()
    summary["n_profiles"] = grouped.size().values
    summary = summary.rename(
        columns={
            "ndcg@k": "mean_ndcg@k",
            "precision@k": "mean_precision@k",
            "recall@k": "mean_recall@k",
        }
    )
    # Present methods in retrieval order (Keyword, Semantic, Hybrid) rather than
    # alphabetically, matching the fixed method stack used in ranking.
    method_positions = {name: i for i, name in enumerate(table.method_names)}
    summary["__order"] = summary["method"].map(method_positions)
    summary = summary.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return per_profile, summary


# =============================================================================
# 12. REPORTING
# =============================================================================


def print_report(
    cfg: Config,
    dataset: Dataset,
    results: pd.DataFrame,
    per_profile: pd.DataFrame,
    summary_df: pd.DataFrame,
    backend_name: str,
    method_names: Sequence[str],
) -> None:
    """Human-readable console report comparing the three methods."""
    rule = "=" * 78
    order = list(method_names)

    print(f"\n{rule}\nVTURCS Team 1 - IRDM Prototype: Retrieval Method Comparison\n{rule}")
    print(f"Dataset         : {dataset.name} - {dataset.description}")
    print(f"Corpus          : {len(dataset.opportunities)} opportunity listings")
    print(f"Profiles        : {len(dataset.queries)} student queries")
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
    top1 = results[results["rank"] == 1].pivot(index="profile_id", columns="method", values="title")
    print(top1.reindex(columns=order).fillna("-").to_string())

    # --- per-profile metrics -------------------------------------------------
    print(f"\n{'-' * 78}\nPer-profile metrics (graded nDCG; Precision/Recall count Partial+)\n{'-' * 78}")
    # With 20+ profiles the full table is unreadable, so print the distribution
    # per method instead. The row-level detail is always in the CSV export.
    scorable_rows = per_profile[per_profile["scorable"]]
    n_excluded = len(per_profile) - len(scorable_rows)
    if n_excluded:
        print(
            f"[note] {n_excluded} profile/method rows excluded from all means below:\n"
            "       those profiles have no relevant listing, so every method scores\n"
            "       them 0.0 identically. See the negative-control table for how they\n"
            "       are actually evaluated.\n"
        )
    if len(per_profile) > 15:
        metric_cols = ["ndcg@k", "precision@k", "recall@k"]
        print(f"(per-profile table has {len(scorable_rows)} scorable rows; showing distribution - see CSV)")
        print(scorable_rows.groupby("method", sort=False)[metric_cols].mean().round(4).to_string())
    else:
        print(per_profile.to_string(index=False))

    # --- headline comparison -------------------------------------------------
    print(f"\n{'-' * 78}\nMEAN PERFORMANCE ACROSS SCORABLE PROFILES\n{'-' * 78}")
    print(summary_df.to_string(index=False))

    # --- negative control: were distractors wrongly surfaced? ----------------
    # Distractor profiles share only generic tooling (Python/C++/Docker) with
    # their target listing and have no domain fit. Scored against an empty
    # relevant set they read as 0.0 for every method, which tells us nothing.
    # The informative question is whether a method *ranked the target highly*
    # anyway. Lower mean rank and lower hit-rate = less fooled = better.
    distractor_rows = per_profile[~per_profile["scorable"]]
    if len(distractor_rows) and distractor_rows["target_rank"].notna().any():
        print(f"\n{'-' * 78}\nNEGATIVE CONTROL: distractor profiles with no relevant listing\n{'-' * 78}")
        n_distractors = distractor_rows["profile_id"].nunique()
        random_baseline = (len(dataset.opportunities) + 1) / 2
        control = pd.DataFrame(
            {
                "mean_target_rank": distractor_rows.groupby("method", sort=False)["target_rank"].mean(),
                "target_in_top_k": distractor_rows.groupby("method", sort=False)["target_rank"]
                .apply(lambda s: s.notna().mean()),
            }
        ).round(3)
        control["target_in_top_k"] = (control["target_in_top_k"] * 100).round(1).astype(str) + "%"
        control = control.reindex(order)
        print(control.to_string())
        print(
            f"\nmetric          : rank given to a listing the profile has NO domain fit for\n"
            f"profiles        : {n_distractors} distractor profiles x {len(order)} methods\n"
            f"random baseline : {random_baseline:.1f} mean rank\n"
            "lower is better : a low rank here means the method was fooled by shared\n"
            "                  generic vocabulary rather than topical fit.\n"
        )

    # --- winner summary ------------------------------------------------------
    print(f"\n{'-' * 78}\nInterpretation\n{'-' * 78}")
    best_by_ndcg = summary_df.loc[summary_df["mean_ndcg@k"].idxmax()]
    best_by_recall = summary_df.loc[summary_df["mean_recall@k"].idxmax()]
    print(f"Best mean nDCG@{cfg.top_k}  : {best_by_ndcg['method']} ({best_by_ndcg['mean_ndcg@k']:.4f})")
    print(f"Best mean Recall@{cfg.top_k}: {best_by_recall['method']} ({best_by_recall['mean_recall@k']:.4f})")

    pivoted = per_profile.pivot(index="profile_id", columns="method", values="ndcg@k")
    wins = pivoted.reindex(columns=order)
    print("\nPer-profile nDCG winner:")
    if len(wins) > 15:
        # Too many rows to list individually; show the win tally instead. Ties
        # count for every method that shares the top score.
        tally: Dict[str, int] = {method_name: 0 for method_name in order}
        for _, row in wins.astype(float).iterrows():
            top_score = row.max()
            for method_name in row.index[row == top_score]:
                tally[method_name] += 1
        for method_name, count in tally.items():
            print(f"  {method_name:<28} ties-or-wins on {count}/{len(wins)} profiles")
    else:
        for profile_id in wins.index:
            row = wins.loc[profile_id].astype(float)
            print(f"  {profile_id}: {row.idxmax():<28} (nDCG@{cfg.top_k} = {row.max():.4f})")

    # --- optional: breakdown by match_type stratum ---------------------------
    # Only meaningful for datasets that label profiles with a match_type.
    strata = {q.profile_id: q.match_type for q in dataset.queries if q.match_type}
    if strata:
        stratum_rows = per_profile.assign(match_type=per_profile["profile_id"].map(strata))
        # Unscorable strata are excluded here for the same reason they are
        # excluded from the headline means: they are 0.0 under every method, so
        # a row of zeros would read as "all methods score 0 on distractors"
        # rather than "distractors are not scored this way at all".
        stratum_scored = stratum_rows[stratum_rows["scorable"]]
        excluded_strata = sorted(set(stratum_rows["match_type"]) - set(stratum_scored["match_type"]))

        print(f"\n{'-' * 78}\nMEAN nDCG@{cfg.top_k} BY PROFILE STRATUM (match_type)\n{'-' * 78}")
        by_stratum = (
            stratum_scored.pivot_table(index="match_type", columns="method", values="ndcg@k", aggfunc="mean")
            .round(4)
            .reindex(columns=order)
        )
        print(by_stratum.to_string())
        if excluded_strata:
            print(
                f"\n(excluded: {', '.join(excluded_strata)} - no relevant listing, so not\n"
                " nDCG-scorable; see the negative-control table above.)"
            )
        print(
            "\nRead this table left to right: on the `exact_keyword` row the lexical\n"
            "method should be competitive, and on `semantic_analogy` it should fall\n"
            "away because none of the listing's vocabulary appears in the profile.\n"
            "That gap is the central claim under test.\n"
        )

    print(
        f"\nNote: dataset '{dataset.name}' - {dataset.description}\n"
        "Absolute values are not the finding - the *relative ordering* of methods\n"
        "and the failure cases are what the pilot is meant to surface.\n"
    )


# =============================================================================
# 13. ENTRY POINT
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VTURCS Team 1 - IRDM matching prototype")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=available_datasets(),
        help="which dataset to evaluate (see data/)",
    )
    parser.add_argument(
        "--match-type",
        type=str,
        nargs="+",
        default=None,
        metavar="TYPE",
        help=(
            "restrict to profiles of these match_type strata "
            "(synthetic_v2 only), e.g. --match-type semantic_analogy distractor"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="cut-off k for retrieval and metrics (default: the dataset's suggested k)",
    )
    parser.add_argument("--bm25-weight", type=float, default=Config.bm25_weight, help="hybrid weight on BM25 (0..1)")
    parser.add_argument(
        "--offline-embeddings",
        action="store_true",
        help="never load sentence-transformers; use the hashed bag-of-words fallback",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.batch_size,
        help="queries per encoder forward pass (larger = faster, more memory)",
    )
    parser.add_argument("--output-dir", type=str, default=Config.output_dir, help="directory for CSV exports")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # --- load data -----------------------------------------------------------
    # Done first so a bad --dataset/--match-type combination fails immediately,
    # before any expensive encoder download.
    match_types = tuple(args.match_type) if args.match_type else ()
    dataset = load_dataset(args.dataset, match_types or None)
    top_k = args.top_k if args.top_k is not None else min(dataset.suggested_top_k, len(dataset.opportunities))

    cfg = Config(
        top_k=top_k,
        bm25_weight=args.bm25_weight,
        semantic_weight=round(1.0 - args.bm25_weight, 10),
        dataset=args.dataset,
        match_types=match_types,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    # --- phase 1: build index (corpus text, BM25 fit, doc embeddings) --------
    if args.offline_embeddings:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        encoder, backend_name = HashedBoWEncoder(), "offline-forced::hashed-bow"
        print("[info] --offline-embeddings set: using HashedBoW encoder (lexical proxy).")
    else:
        encoder, backend_name = load_encoder(cfg.embedding_model, allow_fallback=True)

    index = build_index(cfg, dataset, encoder, backend_name)

    # --- phase 2: rank every query x method ---------------------------------
    table = rank_candidates(index, cfg)
    results = table.to_frame(dataset.opportunities)

    # --- phase 3: vectorised metrics ----------------------------------------
    per_profile, summary_df = evaluate_metrics(table, dataset, k=cfg.top_k, threshold=1)

    # --- export --------------------------------------------------------------
    # Each dataset (and each stratum run) writes to its own subdirectory so runs
    # never silently overwrite one another's CSVs.
    run_name = dataset.name
    if match_types:
        run_name += "__" + "-".join(sorted(match_types))
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "match_results.csv", index=False)
    per_profile.to_csv(out_dir / "evaluation_per_profile.csv", index=False)
    summary_df.to_csv(out_dir / "evaluation_summary.csv", index=False)

    # --- report --------------------------------------------------------------
    print_report(cfg, dataset, results, per_profile, summary_df, backend_name, table.method_names)
    print(f"Exports written to: {out_dir.resolve()}")
    print("  - match_results.csv")
    print("  - evaluation_per_profile.csv")
    print("  - evaluation_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
