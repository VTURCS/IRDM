"""
IRDM core: match student interest profiles to research listings three ways, and
score the rankings against graded relevance labels.

    Keyword   BM25 over each listing's title + description + skills
    Semantic  all-MiniLM-L6-v2 embeddings, cosine similarity
    Hybrid    w * minmax(BM25) + (1 - w) * minmax(cosine), per query (w = 0.6 by default)

All three see the same text: queries are used verbatim, and eligibility and
department are never searched or used to filter.

A dataset is a folder (in data/ or anywhere else) holding three CSV files:

    opportunities.csv  id, title, description   optional: skills (";"-separated), eligibility, department
    profiles.csv       id, query                optional: major, year, match_type, target_id
    relevance.csv      profile_id, opportunity_id, grade   (0 unsuitable, 1 partial, 2 strong)

relevance.csv must grade every (profile, listing) pair, zeros included, because
Recall@k divides by the number of relevant listings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET = "prototype"
MODEL_NAME = "all-MiniLM-L6-v2"
# Pinned Hugging Face commit, so an upstream model update cannot change results.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
KEYWORD = "Keyword (BM25)"
SEMANTIC = f"Semantic ({MODEL_NAME})"

REQUIRED_COLUMNS = {
    "opportunities.csv": ("id", "title", "description"),
    "profiles.csv": ("id", "query"),
    "relevance.csv": ("profile_id", "opportunity_id", "grade"),
}


def hybrid_label(bm25_weight: float) -> str:
    return f"Hybrid ({bm25_weight:g} KW / {round(1 - bm25_weight, 10):g} SEM)"


# --- data -------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    name: str
    opportunities: pd.DataFrame  # one row per listing
    profiles: pd.DataFrame  # one row per student profile
    grades: np.ndarray  # grades[i, j] = relevance of listing j to profile i


def list_datasets() -> list[str]:
    """Names of the dataset folders in data/."""
    return sorted(p.name for p in DATA_DIR.iterdir() if (p / "opportunities.csv").is_file())


def _read_csv(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.is_file():
        raise ValueError(f"{path}: file not found")
    try:
        # utf-8-sig accepts the byte-order mark Excel adds when saving as CSV.
        table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    missing = [column for column in REQUIRED_COLUMNS[filename] if column not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    if table.empty:
        raise ValueError(f"{path}: contains no data rows")
    table = table.apply(lambda column: column.str.strip())
    for column in REQUIRED_COLUMNS[filename]:
        blank = table.index[table[column] == ""]
        if len(blank):
            raise ValueError(f"{path}:{blank[0] + 2}: '{column}' must not be empty")  # line 1 is the header
    return table


def load_dataset(name_or_path: str | Path = DEFAULT_DATASET) -> Dataset:
    """Load a dataset by folder name in data/ or by path. Raises ValueError, naming the file, on bad input."""
    folder = DATA_DIR / name_or_path if (DATA_DIR / name_or_path).is_dir() else Path(name_or_path)
    if not folder.is_dir():
        raise ValueError(f"no dataset '{name_or_path}'; available: {', '.join(list_datasets())}")

    opportunities = _read_csv(folder, "opportunities.csv")
    profiles = _read_csv(folder, "profiles.csv")
    labels = _read_csv(folder, "relevance.csv")
    for table, filename in ((opportunities, "opportunities.csv"), (profiles, "profiles.csv")):
        duplicates = table["id"][table["id"].duplicated()]
        if len(duplicates):
            raise ValueError(f"{folder / filename}: duplicate id '{duplicates.iloc[0]}'")

    path = folder / "relevance.csv"
    checks = (
        (~labels["profile_id"].isin(profiles["id"]) | ~labels["opportunity_id"].isin(opportunities["id"]),
         "unknown profile or opportunity id"),
        (~labels["grade"].isin(["0", "1", "2"]), "grade must be 0, 1 or 2"),
        (labels.duplicated(["profile_id", "opportunity_id"]), "this pair is graded twice"),
    )
    for bad, message in checks:
        if bad.any():
            line = bad.idxmax()
            row = labels.loc[line]
            raise ValueError(f"{path}:{line + 2}: {message} (row: {row.profile_id},{row.opportunity_id},{row.grade})")

    grades = (
        labels.astype({"grade": int})
        .pivot(index="profile_id", columns="opportunity_id", values="grade")
        .reindex(index=profiles["id"], columns=opportunities["id"])
    )
    gaps = np.argwhere(grades.isna().to_numpy())
    if len(gaps):
        first = (profiles["id"].iloc[gaps[0][0]], opportunities["id"].iloc[gaps[0][1]])
        raise ValueError(
            f"{path}: {len(gaps)} (profile, opportunity) pairs have no grade, starting with {first}. "
            "Every pair needs a grade, zeros included."
        )
    return Dataset(folder.name, opportunities, profiles, grades.to_numpy(dtype=int))


# --- retrieval --------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:\+\+|#)?")  # keeps tokens like c++, c#, 3d
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "i", "in",
    "into", "is", "it", "like", "not", "of", "on", "or", "some", "that", "the",
    "their", "to", "very", "want", "with", "work", "working", "we", "our",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word tokens and drop stop-words. Applied to listings and queries alike."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


def document_text(listing: pd.Series) -> str:
    """The searchable text of a listing: title, description and skills."""
    skills = [skill.strip() for skill in listing.get("skills", "").split(";") if skill.strip()]
    return f"{listing['title']} {listing['description']} Skills: {', '.join(skills)}"


class _BM25(BM25Okapi):
    """
    BM25Okapi (k1=1.5, b=0.75) with negative IDFs floored at 0. In a small corpus a
    word that appears in most listings gets a negative IDF, which would make scores
    negative and push matching listings below non-matching ones.
    """

    def _calc_idf(self, nd):
        super()._calc_idf(nd)
        self.idf = {term: max(0.0, idf) for term, idf in self.idf.items()}


@lru_cache(maxsize=1)
def load_model():
    from sentence_transformers import SentenceTransformer  # slow import, so only when needed

    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)


@lru_cache(maxsize=16)
def _index(documents: tuple[str, ...]):
    """BM25 index and unit-length embeddings for a corpus, built once per distinct corpus."""
    return _BM25([tokenize(d) for d in documents]), load_model().encode(list(documents), normalize_embeddings=True)


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Rescale each row to [0, 1]. A row of equal scores (e.g. no keyword overlap at all) becomes all zeros."""
    low = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - low
    return np.divide(scores - low, span, out=np.zeros_like(scores), where=span > 1e-12)


def score(dataset: Dataset, queries, bm25_weight: float = 0.6) -> dict[str, np.ndarray]:
    """Score queries against every listing: {method: (n_queries x n_listings) scores, higher is better}."""
    bm25, embeddings = _index(tuple(document_text(row) for _, row in dataset.opportunities.iterrows()))
    keyword = np.array([bm25.get_scores(tokenize(q)) for q in queries], dtype=float)
    semantic = (load_model().encode(list(queries), normalize_embeddings=True) @ embeddings.T).astype(float)
    hybrid = bm25_weight * _minmax(keyword) + round(1 - bm25_weight, 10) * _minmax(semantic)
    return {KEYWORD: keyword, SEMANTIC: semantic, hybrid_label(bm25_weight): hybrid}


def top_k(scores: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k best listings per row: highest score first, ties broken by listing id."""
    return np.array([np.lexsort((ids, -row))[:k] for row in scores])


# --- evaluation -------------------------------------------------------------


def ndcg(ranked: np.ndarray, grades: np.ndarray) -> np.ndarray:
    """
    nDCG per row. `ranked` holds the grades of the retrieved listings in rank order
    (rows x k); `grades` holds every grade for the same rows, which gives the ideal
    order. Gain is 2^grade - 1, so a strong match counts three times a partial one.
    A row with nothing relevant scores 0.
    """
    discount = 1 / np.log2(np.arange(2, ranked.shape[1] + 2))
    dcg = ((2.0**ranked - 1) * discount).sum(axis=1)
    ideal = -np.sort(-grades, axis=1)[:, : ranked.shape[1]]
    idcg = ((2.0**ideal - 1) * discount).sum(axis=1)
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def _rank_of(ranked_ids: np.ndarray, target: str) -> float:
    hits = np.flatnonzero(ranked_ids == target)
    return float(hits[0] + 1) if target and len(hits) else np.nan


def evaluate(dataset: Dataset, k: int = 5, bm25_weight: float = 0.6):
    """
    Rank every profile with every method and score the rankings. Returns three tables:

      results      one row per (method, profile, rank): the listing and its score
      per_profile  nDCG@k, Precision@k and Recall@k per (profile, method)
      summary      mean metrics per method over scorable profiles

    Precision and Recall count grade >= 1 as relevant, and Precision always divides
    by k. A profile with no relevant listing scores 0 under every method, so it is
    marked unscorable and left out of the means. If profiles carry a `target_id`,
    `target_rank` records where each method ranked that listing (the negative
    control for distractor profiles).
    """
    listings, profiles, grades = dataset.opportunities, dataset.profiles, dataset.grades
    ids = listings["id"].to_numpy(dtype=str)
    n_relevant = (grades >= 1).sum(axis=1)
    targets = profiles["target_id"] if "target_id" in profiles else None
    results, per_profile = [], []
    for method, scores in score(dataset, profiles["query"], bm25_weight).items():
        top = top_k(scores, ids, min(k, len(ids)))
        ranked = np.take_along_axis(grades, top, axis=1)
        hits = (ranked >= 1).sum(axis=1)
        results.append(pd.DataFrame({
            "profile_id": np.repeat(profiles["id"].to_numpy(), top.shape[1]),
            "method": method,
            "opportunity_id": ids[top].ravel(),
            "rank": np.tile(np.arange(1, top.shape[1] + 1), len(top)),
            "score": np.take_along_axis(scores, top, axis=1).ravel().round(6),
            "title": listings["title"].to_numpy()[top].ravel(),
            "description": listings["description"].to_numpy()[top].ravel(),
        }))
        per_profile.append(pd.DataFrame({
            "profile_id": profiles["id"],
            "method": method,
            "ndcg@k": ndcg(ranked, grades).round(4),
            "precision@k": (hits / k).round(4),
            "recall@k": np.divide(hits, n_relevant, out=np.zeros(len(hits)), where=n_relevant > 0).round(4),
            "n_relevant": n_relevant,
            "n_strong": (grades == 2).sum(axis=1),
            "scorable": n_relevant > 0,
            "target_rank": np.nan if targets is None else [_rank_of(ids[row], t) for row, t in zip(top, targets)],
        }))

    per_profile = pd.concat(per_profile, ignore_index=True)
    scorable = per_profile[per_profile["scorable"]].groupby("method", sort=False)
    summary = scorable[["ndcg@k", "precision@k", "recall@k"]].mean().round(4).add_prefix("mean_")
    summary["n_profiles"] = scorable.size()
    return pd.concat(results, ignore_index=True), per_profile, summary.reset_index()


def by_match_type(dataset: Dataset, per_profile: pd.DataFrame) -> pd.DataFrame | None:
    """Mean nDCG@k per profile type (rows) and method (columns), when profiles have a match_type."""
    if "match_type" not in dataset.profiles:
        return None
    types = per_profile["profile_id"].map(dict(zip(dataset.profiles["id"], dataset.profiles["match_type"])))
    table = per_profile[per_profile["scorable"]].assign(match_type=types)
    return table.pivot_table(index="match_type", columns="method", values="ndcg@k").round(4)[
        list(dict.fromkeys(per_profile["method"]))
    ]


def search(dataset: Dataset, query: str, k: int = 5, bm25_weight: float = 0.6):
    """
    Rank the listings for one free-text query with each method. If the query is one
    of the dataset's profiles, listings carry their grade and methods their nDCG@k.
    Returns (matching profile id or None, [{method, ndcg, listings}]).
    """
    listings = dataset.opportunities
    ids = listings["id"].to_numpy(dtype=str)
    match = np.flatnonzero(dataset.profiles["query"].to_numpy() == query)[:1]
    grades = dataset.grades[match]  # 1 x listings for a known profile, else 0 x listings
    results = []
    for method, scores in score(dataset, [query], bm25_weight).items():
        top = top_k(scores, ids, min(k, len(ids)))[0]
        results.append({
            "method": method,
            "ndcg": float(ndcg(grades[:, top], grades)[0]) if len(match) else None,
            "listings": [
                {
                    "id": str(ids[j]),
                    "title": listings["title"].iat[j],
                    "score": float(scores[0, j]),
                    "grade": int(grades[0, j]) if len(match) else None,
                }
                for j in top
            ],
        })
    return (dataset.profiles["id"].iat[match[0]] if len(match) else None), results
