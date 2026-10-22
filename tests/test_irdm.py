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
    page = client.get("/").get_data(as_text=True)
    assert "How each method scores on prototype" in page
    assert "0.8846" in page


def test_query_shows_each_method_ranking(client):
    p1 = irdm.load_dataset("prototype").profiles["query"].iloc[0]
    page = client.get("/", query_string={"dataset": "prototype", "query": p1}).get_data(as_text=True)
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
    assert "How each method scores on prototype" in response.get_data(as_text=True)


def test_broken_dataset_folder_shows_the_problem(client, tmp_path, monkeypatch):
    shutil.copytree(irdm.DATA_DIR / "main_demo", tmp_path / "mine")
    (tmp_path / "mine" / "relevance.csv").write_text("profile_id,opportunity_id,grade\n")
    monkeypatch.setattr(irdm, "DATA_DIR", tmp_path)
    page = client.get("/", query_string={"dataset": "mine"}).get_data(as_text=True)
    assert "contains no data rows" in page
