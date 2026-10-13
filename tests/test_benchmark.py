"""Hand-computed checks for the experiment's evaluation contract."""
import math
import tempfile
import unittest
from pathlib import Path

from benchmark import evaluate, hybrid_scores, load_labels, rank, read_csv, tokenize, unique_ids


class EvaluationTests(unittest.TestCase):
    def test_known_ranking(self):
        labels = {("p", "a"): 2, ("p", "b"): 1, ("p", "c"): 0}
        result = evaluate("p", "test", ["b", "c"], ["a", "b", "c"], labels)
        self.assertEqual(result["precision_at_10"], 0.5)
        self.assertEqual(result["recall_at_10"], 0.5)
        self.assertAlmostEqual(result["ndcg_at_10"], 1 / (3 + 1 / math.log2(3)))
        self.assertEqual(result["status"], "complete")

    def test_missing_unreturned_label_does_not_invent_recall(self):
        result = evaluate("p", "test", ["a"], ["a", "b"], {("p", "a"): 2})
        self.assertEqual(result["precision_at_10"], 1)
        self.assertEqual(result["recall_at_10"], "")
        self.assertEqual(result["ndcg_at_10"], "")
        self.assertEqual(result["status"], "missing_corpus_labels_for_ndcg_and_recall")

    def test_missing_returned_label_blanks_metrics(self):
        result = evaluate("p", "test", ["a", "b"], ["a", "b"], {("p", "a"): 2})
        for metric in ("precision_at_10", "recall_at_10", "ndcg_at_10"):
            self.assertEqual(result[metric], "")
        self.assertEqual(result["status"], "missing_top10_labels")

    def test_no_relevant_opportunities(self):
        result = evaluate("p", "test", ["a"], ["a"], {("p", "a"): 0})
        self.assertEqual(result["precision_at_10"], 0)
        self.assertEqual(result["recall_at_10"], "")
        self.assertEqual(result["ndcg_at_10"], "")
        self.assertEqual(result["status"], "no_relevant_opportunities")


class RetrievalTests(unittest.TestCase):
    def test_zero_keyword_signal_preserves_semantic_order(self):
        scores = hybrid_scores([0, 1, 2], [2, 1, 0], keyword_has_signal=False)
        self.assertEqual(rank(scores, ["a", "b", "c"]), [2, 1, 0])

    def test_weighted_fusion(self):
        scores = hybrid_scores([0, 1], [1, 0])
        self.assertAlmostEqual(scores[0], 0.6 / 61 + 0.4 / 62)
        self.assertGreater(scores[0], scores[1])

    def test_ties_are_deterministic(self):
        self.assertEqual(rank([0, 0], ["b", "a"]), [1, 0])

    def test_demo_has_five_cases_and_complete_labels(self):
        root = Path(__file__).resolve().parents[1] / "demo"
        profiles = read_csv(root / "profiles.csv", ("id", "query"))
        opportunities = read_csv(root / "opportunities.csv", ("id", "title", "description"))
        self.assertEqual(len(profiles), 5)
        labels = load_labels(root / "relevance.csv", unique_ids(profiles, "profiles"),
                             unique_ids(opportunities, "opportunities"))
        self.assertEqual(len(labels), len(profiles) * len(opportunities))
        vocabulary = set(tokenize(" ".join(row["title"] + " " + row["description"]
                                           for row in opportunities)))
        self.assertTrue(vocabulary.isdisjoint(tokenize(profiles[2]["query"])))


class LabelValidationTests(unittest.TestCase):
    def test_invalid_labels_rejected(self):
        for rows in ("p,a,3\n", "q,a,1\n", "p,b,1\n", "p,a,1\np,a,2\n"):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "relevance.csv"
                path.write_text("profile_id,opportunity_id,grade\n" + rows, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_labels(path, ["p"], ["a"])


if __name__ == "__main__":
    unittest.main()
