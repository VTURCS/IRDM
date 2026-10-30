"""
Checks that the retrieval pipeline, the CLI and the web UI work out of the box.

    docker compose run --rm irdm pytest     # in Docker
    pytest                                  # locally

The first two tests pin the numbers reported in docs/RESULTS.md, so any change to
the code, data or environment that moves a published result fails here.
"""

import re
import shutil

import numpy as np
import pandas as pd
import pytest

import app
import evaluate
import irdm

KW, SEM, HYB = irdm.KEYWORD, irdm.SEMANTIC, "Hybrid (0.6 KW / 0.4 SEM)"

# Mean (nDCG@5, Precision@5, Recall@5) per method.
PUBLISHED = {
    "vt_furf": {KW: (0.4852, 0.3083, 0.4608), SEM: (0.8079, 0.4417, 0.7192), HYB: (0.6228, 0.3917, 0.6326)},
    "prototype": {KW: (0.7046, 0.55, 0.7417), SEM: (0.8846, 0.65, 0.8875), HYB: (0.8364, 0.65, 0.8875)},
    "synthetic_v2": {KW: (0.9667, 0.20, 1.0), SEM: (0.9421, 0.20, 1.0), HYB: (0.9754, 0.20, 1.0)},
    "heldout": {KW: (0.5820, 0.2364, 0.5455), SEM: (0.8649, 0.3909, 0.8583), HYB: (0.8320, 0.3818, 0.8356)},
    "main_demo": {KW: (0.8454, 0.40, 0.9167), SEM: (1.0, 0.45, 1.0), HYB: (1.0, 0.45, 1.0)},
}

# Hybrid mean nDCG@5 on prototype by keyword weight. The endpoints equal the pure methods.
SWEEP = {0.0: 0.8846, 0.2: 0.9011, 0.3: 0.9004, 0.4: 0.9030, 0.6: 0.8364, 0.8: 0.8239, 1.0: 0.7046}


@pytest.mark.parametrize("name", PUBLISHED)
def test_reproduces_published_results(name):
    _, _, summary = irdm.evaluate(irdm.load_dataset(name))
    got = summary.set_index("method")[["mean_ndcg@k", "mean_precision@k", "mean_recall@k"]]
    assert list(got.index) == list(PUBLISHED[name])
    for method, expected in PUBLISHED[name].items():
        assert tuple(got.loc[method]) == pytest.approx(expected, abs=1e-4), method


def test_every_dataset_folder_loads():
    """
    Every folder in data/ must load: well-formed CSVs, known ids, every pair graded. This
    catches a slip while hand-editing relevance.csv (e.g. reviewing vt_furf's draft grades).
    """
    names = irdm.list_datasets()
    assert set(PUBLISHED) <= set(names)
    for name in names:
        dataset = irdm.load_dataset(name)  # raises ValueError naming the file and line
        assert (dataset.grades >= 1).any(), f"{name}: no relevant listing for any profile"


def test_reproduces_weight_sweep():
    dataset = irdm.load_dataset("prototype")
    for weight, expected in SWEEP.items():
        hybrid = irdm.evaluate(dataset, bm25_weight=weight)[2]["mean_ndcg@k"].iloc[2]
        assert hybrid == pytest.approx(expected, abs=1e-4), f"bm25_weight={weight}"


def test_ndcg_matches_hand_computation():
    grades = np.array([[2, 1, 0]])
    assert irdm.ndcg(np.array([[2, 1, 0]]), grades)[0] == pytest.approx(1.0)
    worst = (0 + 1 / np.log2(3) + 3 / 2) / (3 + 1 / np.log2(3))
    assert irdm.ndcg(np.array([[0, 1, 2]]), grades)[0] == pytest.approx(worst)
    assert irdm.ndcg(np.array([[0, 0]]), np.array([[0, 0, 0]]))[0] == 0  # nothing relevant: 0, no error


def test_no_keyword_overlap_ties_break_on_id_and_hybrid_follows_semantic():
    dataset = irdm.load_dataset("main_demo")
    _, results = irdm.search(dataset, "Quantum cryptography", k=12)
    ranked = {r["method"]: [listing["id"] for listing in r["listings"]] for r in results}
    assert ranked[KW] == sorted(ranked[KW])  # every BM25 score is 0, so listings come back in id order
    assert ranked[HYB] == ranked[SEM]


def test_search_grades_a_known_profile():
    dataset = irdm.load_dataset("prototype")
    profile_id, results = irdm.search(dataset, dataset.profiles["query"].iloc[0])
    assert profile_id == "P1"
    assert [r["method"] for r in results] == [KW, SEM, HYB]
    assert all(r["ndcg"] is not None and len(r["listings"]) == 5 for r in results)
    assert irdm.search(dataset, "something no profile says")[0] is None


@pytest.mark.parametrize("filename, old, new, message", [
    ("relevance.csv", "p01,o01,2", "p01,o01,3", "relevance.csv:2: grade must be 0, 1 or 2"),
    ("relevance.csv", "p01,o01,2\n", "", "1 (profile, opportunity) pairs have no grade"),
    ("relevance.csv", "p01,o01,2", "p01,o99,2", "relevance.csv:2: unknown profile or opportunity id"),
    ("profiles.csv", "id,query", "id,question", "missing required columns: query"),
    ("opportunities.csv", "o02,", "o01,", "duplicate id 'o01'"),
])
def test_bad_dataset_is_rejected_with_a_clear_message(tmp_path, filename, old, new, message):
    folder = shutil.copytree(irdm.DATA_DIR / "main_demo", tmp_path / "broken")
    path = folder / filename
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=re.escape(message)):
        irdm.load_dataset(folder)


def test_cli_prints_results_and_writes_three_csvs(tmp_path, capsys):
    assert evaluate.main(["prototype", "--output-dir", str(tmp_path)]) == 0
    out = tmp_path / "prototype"
    assert sorted(p.name for p in out.iterdir()) == [
        "evaluation_per_profile.csv", "evaluation_summary.csv", "match_results.csv"]
    assert len(pd.read_csv(out / "match_results.csv")) == 4 * 3 * 5  # profiles x methods x ranks
    assert "0.8846" in capsys.readouterr().out


def test_cli_rejects_bad_input():
    with pytest.raises(SystemExit) as exit_info:
        evaluate.main(["--bm25-weight", "1.5"])
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit, match="Error: no dataset 'nope'"):
        evaluate.main(["nope"])


@pytest.fixture
def client():
    return app.app.test_client()


def test_home_page_shows_the_evaluation(client):
    # Computed rather than pinned: the default dataset's grades are still a draft under review.
    _, _, summary = irdm.evaluate(irdm.load_dataset(irdm.DEFAULT_DATASET))
    page = client.get("/").get_data(as_text=True)
    assert f"How each method scores on {irdm.DEFAULT_DATASET}" in page
    for ndcg in summary["mean_ndcg@k"]:
        assert f"{ndcg:.4f}" in page


def test_query_shows_each_method_ranking(client):
    p1 = irdm.load_dataset("prototype").profiles["query"].iloc[0]
    page = client.get("/", query_string={"dataset": "prototype", "query": p1, "view": "compare"}).get_data(as_text=True)
    assert "LLM-Based Tutoring Chatbot for CS1 Courses" in page  # semantic's top pick for P1
    assert "nDCG@5" in page and "strong" in page  # graded, because the query is a dataset profile


def test_search_form_carries_dataset_k_and_weight(client):
    """
    Dataset, k and weight are set through the URL, so the form must resubmit them as
    hidden fields. Without this, pressing Match silently reverts to the defaults.
    """
    page = client.get("/", query_string={"dataset": "heldout", "k": 3, "weight": 0.4}).get_data(as_text=True)
    for field, value in (("dataset", "heldout"), ("k", "3"), ("weight", "0.4")):
        assert f'<input type="hidden" name="{field}" value="{value}">' in page


def test_bad_parameters_fall_back_to_defaults(client):
    response = client.get("/", query_string={"dataset": "nope", "k": "x", "weight": "7"})
    assert response.status_code == 200
    assert f"How each method scores on {irdm.DEFAULT_DATASET}" in response.get_data(as_text=True)


def test_broken_dataset_folder_shows_the_problem(client, tmp_path, monkeypatch):
    shutil.copytree(irdm.DATA_DIR / "main_demo", tmp_path / "mine")
    (tmp_path / "mine" / "relevance.csv").write_text("profile_id,opportunity_id,grade\n")
    monkeypatch.setattr(irdm, "DATA_DIR", tmp_path)
    page = client.get("/", query_string={"dataset": "mine"}).get_data(as_text=True)
    assert "contains no data rows" in page


def test_search_normalizes_full_corpus_before_truncation(monkeypatch):
    dataset = irdm.load_dataset("prototype")
    scores = {
        KW: np.array([[0, 10, 20, 30, 40, 50, 60, 100]], dtype=float),
        SEM: np.array([[-1, -.8, -.6, -.4, -.2, 0, .2, 1]], dtype=float),
        HYB: np.zeros((1, 8)),
    }
    monkeypatch.setattr(irdm, "score", lambda *args: scores)
    _, short = irdm.search(dataset, "custom", k=2)
    _, full = irdm.search(dataset, "custom", k=8)
    for result, complete in zip(short, full):
        assert result["listings"] == complete["listings"][:2]
    assert [r["normalized_score"] for r in short[0]["listings"]] == [1, .6]
    assert [r["score"] for r in short[0]["listings"]] == [100, 60]
    assert [r["normalized_score"] for r in short[1]["listings"]] == [1, .6]
    assert all(r["normalized_score"] == 0 for r in full[2]["listings"])


def test_single_list_deduplicates_filters_and_sorts_normalized_scores(client):
    from flask import template_rendered
    captured = []

    def record(sender, template, context, **extra):
        captured.append(context)

    with template_rendered.connected_to(record, app.app):
        response = client.get("/", query_string={"dataset": "vt_furf", "query": "I like biology"})
    assert response.status_code == 200
    context = captured[0]
    merged = context["combined_results"]
    assert context["view"] == "one"
    candidates = [l for r in context["results"] for l in r["listings"]]
    expected_ids = {l["id"] for l in candidates if l["normalized_score"] >= 0.5 and l["semantic_score"] >= 0.3}
    assert {l["id"] for l in merged} == expected_ids
    assert len(merged) == len(expected_ids)
    for l in merged:
        assert l["normalized_score"] == max(c["normalized_score"] for c in candidates if c["id"] == l["id"])
    scores = [l["normalized_score"] for l in merged]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= score <= 1 for score in scores)
    page = response.get_data(as_text=True)
    assert page.count('class="score-method"') == len(merged)
    assert page.count('class="combined-list"') == 1
    assert 'class="method"' not in page


def test_comparison_view_preserves_raw_scores(client):
    query = "I like biology"
    dataset = irdm.load_dataset("prototype")
    _, expected = irdm.search(dataset, query)
    page = client.get("/", query_string={"dataset": "prototype", "query": query, "view": "compare"}).get_data(as_text=True)
    assert page.count('class="method"') == 3
    assert 'class="combined-list"' not in page
    actual = [float(n) for n in re.findall(r'class="result-score">Score ([\d.-]+)', page)]
    assert actual == [round(l["score"], 3) for r in expected for l in r["listings"]]


def test_unknown_view_uses_single_list(client):
    page = client.get("/", query_string={"query": "biology", "view": "invalid"}).get_data(as_text=True)
    assert 'class="combined-list"' in page


def test_combined_threshold_best_method_ties_and_boundaries():
    def row(id, value):
        return {"id": id, "normalized_score": value, "semantic_score": .8, "score": value, "title": id, "grade": None}
    results = [
        {"method": KW, "listings": [row("a", .3), row("b", .5), row("c", 0)]},
        {"method": SEM, "listings": [row("a", .9), row("b", .5)]},
        {"method": HYB, "listings": [row("a", .8), row("d", 1)]},
    ]
    combined = app.combine_results(results)
    assert [r["id"] for r in combined] == ["d", "a", "b"]
    assert combined[1]["method"] == SEM
    assert set(combined[2]["methods"]) == {KW, SEM}
    assert [r["id"] for r in app.combine_results(results)] == ["d", "a", "b"]
    assert len(app.combine_results(results)) == 3
    assert app.combine_results([]) == []
    assert app.combine_results([{"method": KW, "listings": [row("a", .4999)]}]) == []


def test_cutoffs_are_fixed_and_not_user_controls(client):
    page = client.get("/", query_string={"query": "biology", "min_score": "0", "min_similarity": "0"}).get_data(as_text=True)
    assert "Minimum score" not in page
    assert "Minimum similarity" not in page
    assert 'name="min_score"' not in page
    assert 'name="min_similarity"' not in page
    assert "fixed score and semantic similarity cutoffs" in page


def test_combined_empty_state(client, monkeypatch):
    results = [{"method": KW, "listings": [
        {"id": "a", "normalized_score": .2, "semantic_score": .2, "score": .2, "title": "Example", "grade": None}
    ]}]
    monkeypatch.setattr(irdm, "search", lambda *args: (None, results))
    page = client.get("/", query_string={"query": "unrelated"}).get_data(as_text=True)
    assert 'No projects meet the matching cutoffs' in page
    assert 'class="combined-list"' not in page


def test_similarity_gate_rejects_high_rescaled_scores_from_every_method():
    def candidate(id, normalized, semantic):
        return {"id": id, "normalized_score": normalized, "semantic_score": semantic}
    results = [
        {"method": KW, "listings": [candidate("weak", 1, .2), candidate("pass", .8, .3)]},
        {"method": SEM, "listings": [candidate("weak", 1, .2)]},
        {"method": HYB, "listings": [candidate("weak", 1, .2)]},
    ]
    assert [r["id"] for r in app.combine_results(results)] == ["pass"]
    assert len(app.combine_results(results)) == 1
    results[0]["listings"][1]["semantic_score"] = .29
    assert app.combine_results(results) == []


def test_similarity_field_is_raw_semantic_score_for_every_method():
    dataset = irdm.load_dataset("prototype")
    _, results = irdm.search(dataset, "I like biology", k=8)
    semantic = {l["id"]: l["score"] for r in results if r["method"] == SEM for l in r["listings"]}
    for result in results:
        for listing in result["listings"]:
            assert listing["semantic_score"] == semantic[listing["id"]]
