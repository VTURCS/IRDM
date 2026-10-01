"""Compare BM25, semantic search, and keyword-prioritized hybrid retrieval."""

import argparse
import csv
import math
import re
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KEYWORD_WEIGHT = 0.6
RRF_K = 60


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {', '.join(required)}")
        rows = list(reader)
    for line, row in enumerate(rows, 2):
        if any(not (row.get(key) or "").strip() for key in required):
            raise ValueError(f"{path}:{line}: required values must not be empty")
        for key in required:
            row[key] = row[key].strip()
    return rows


def unique_ids(rows, name):
    ids = [row["id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{name}: provide at least one row with unique IDs")
    return ids


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def rank(scores, ids):
    # IDs make ties deterministic across all methods.
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))


def hybrid_scores(keyword_order, semantic_order):
    scores = [0.0] * len(keyword_order)
    for weight, order in ((KEYWORD_WEIGHT, keyword_order),
                          (1 - KEYWORD_WEIGHT, semantic_order)):
        for position, index in enumerate(order, 1):
            scores[index] += weight / (RRF_K + position)
    return scores


def load_labels(path, profile_ids, opportunity_ids):
    if not path.exists():
        return None
    labels = {}
    for row in read_csv(path, ("profile_id", "opportunity_id", "grade")):
        pair = (row["profile_id"], row["opportunity_id"])
        if pair[0] not in profile_ids or pair[1] not in opportunity_ids:
            raise ValueError(f"Unknown IDs in relevance labels: {pair}")
        if pair in labels:
            raise ValueError(f"Duplicate relevance label: {pair}")
        if row["grade"] not in {"0", "1", "2"}:
            raise ValueError(f"{pair}: grade must be 0, 1, or 2")
        labels[pair] = int(row["grade"])
    return labels


def evaluate(profile_id, method, top_ids, all_ids, labels):
    complete_top = all((profile_id, oid) in labels for oid in top_ids)
    complete_corpus = all((profile_id, oid) in labels for oid in all_ids)
    precision = ""
    ndcg = ""
    status = "missing_top10_labels"
    if complete_top:
        grades = [labels[(profile_id, oid)] for oid in top_ids]
        precision = sum(grade > 0 for grade in grades) / len(grades)
        status = "missing_corpus_labels_for_ndcg"
        if complete_corpus:
            ideal = sorted((labels[(profile_id, oid)] for oid in all_ids),
                           reverse=True)[:len(top_ids)]
            dcg = lambda values: sum((2 ** grade - 1) / math.log2(i + 2)
                                     for i, grade in enumerate(values))
            denominator = dcg(ideal)
            ndcg = dcg(grades) / denominator if denominator else ""
            status = "complete" if denominator else "no_relevant_opportunities"
    return {"profile_id": profile_id, "method": method,
            "returned": len(top_ids), "precision_at_10": precision,
            "ndcg_at_10": ndcg, "status": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.data_dir
    try:
        opportunities = read_csv(root / "opportunities.csv", ("id", "title", "description"))
        profiles = read_csv(root / "profiles.csv", ("id", "query"))
        ids = unique_ids(opportunities, "opportunities")
        profile_ids = unique_ids(profiles, "profiles")
        labels = load_labels(root / "relevance.csv", profile_ids, ids)

        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        texts = [f"{row['title']} {row['description']}" for row in opportunities]
        tokens = [tokenize(text) for text in texts]
        if any(not words for words in tokens):
            raise ValueError("Each opportunity needs at least one searchable word")
        if any(not tokenize(row["query"]) for row in profiles):
            raise ValueError("Each query needs at least one searchable word")
        bm25 = BM25Okapi(tokens)
        model = SentenceTransformer(MODEL, device="cpu")
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        queries = model.encode([row["query"] for row in profiles],
                               normalize_embeddings=True, show_progress_bar=False)
        results, metrics = [], []
        for profile, query_vector in zip(profiles, queries):
            keyword = bm25.get_scores(tokenize(profile["query"]))
            semantic = np.asarray(vectors) @ query_vector
            hybrid = hybrid_scores(rank(keyword, ids), rank(semantic, ids))
            for method, scores in (("bm25", keyword), ("semantic", semantic),
                                   ("hybrid", hybrid)):
                top = rank(scores, ids)[:10]
                for position, i in enumerate(top, 1):
                    results.append({"profile_id": profile["id"], "query": profile["query"],
                                    "method": method, "opportunity_id": ids[i],
                                    "rank": position, "score": float(scores[i]),
                                    "title": opportunities[i]["title"],
                                    "description": opportunities[i]["description"]})
                if labels is not None:
                    metrics.append(evaluate(profile["id"], method,
                                            [ids[i] for i in top], ids, labels))
        write_csv(root / "results.csv", ("profile_id", "query", "method", "opportunity_id",
                                         "rank", "score", "title", "description"), results)
        print(f"Wrote {root / 'results.csv'}")
        # Always replace this file, so an earlier labeled run cannot leave stale metrics.
        write_csv(root / "metrics.csv", ("profile_id", "method", "returned", "precision_at_10",
                                         "ndcg_at_10", "status"), metrics)
        if labels is None:
            print("No relevance.csv: metrics.csv contains headers only.")
        else:
            print(f"Wrote {root / 'metrics.csv'}; inspect status for incomplete labels.")
            for method in ("bm25", "semantic", "hybrid"):
                for metric in ("precision_at_10", "ndcg_at_10"):
                    values = [row[metric] for row in metrics
                              if row["method"] == method and row[metric] != ""]
                    if values:
                        print(f"{method} {metric}: {sum(values)/len(values):.4f} "
                              f"({len(values)}/{len(profiles)} profiles)")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
