"""
Shared pytest fixtures and helpers.

Adds the repository root to `sys.path` so tests import the package the same way
the CLI does, without requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schema import Dataset, Opportunity, Query  # noqa: E402

# A small shared vocabulary. Random queries MUST draw from the corpus
# vocabulary, otherwise BM25 has zero lexical signal and every retrieval method
# degenerates to an arbitrary tie order - which produces confusing false
# failures in the metric-equivalence tests.
VOCAB = [f"skill{i}" for i in range(30)]


def make_random_dataset(n_queries: int, n_candidates: int, seed: int) -> Dataset:
    """Build a small random, fully-labelled dataset for property-based checks."""
    rng = np.random.default_rng(seed)

    opportunities = []
    for i in range(n_candidates):
        skills = [str(t) for t in rng.choice(VOCAB, size=int(rng.integers(3, 7)), replace=False)]
        opportunities.append(
            Opportunity(
                id=f"O{i:03d}",
                title=f"Listing {i} {skills[0]}",
                description=" ".join(str(t) for t in rng.choice(VOCAB, size=15)),
                skills=skills,
            )
        )

    queries = [
        Query(profile_id=f"Q{j:03d}", text=" ".join(str(t) for t in rng.choice(VOCAB, size=8)))
        for j in range(n_queries)
    ]

    relevance_map = {
        (q.profile_id, o.id): int(rng.choice([0, 0, 0, 1, 2]))
        for q in queries
        for o in opportunities
    }

    return Dataset(
        name="random_test",
        description="randomly generated dataset for property tests",
        opportunities=opportunities,
        queries=queries,
        relevance_map=relevance_map,
        suggested_top_k=min(5, n_candidates),
    )


class DeterministicEncoder:
    """
    Fake encoder returning deterministic unit vectors.

    Keeps semantic-similarity tests independent of any model download while
    still producing a non-trivial, reproducible embedding space.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts, **_kwargs) -> np.ndarray:  # noqa: ANN001 - matches ST API
        rows = len(texts)
        # Seed from the text content so identical text -> identical vector.
        vectors = np.vstack(
            [np.random.default_rng(abs(hash(t)) % (2**32)).random(self.dim) for t in texts]
        ) if rows else np.zeros((0, self.dim))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


@pytest.fixture
def encoder() -> DeterministicEncoder:
    """Deterministic stand-in for the sentence encoder."""
    return DeterministicEncoder()
