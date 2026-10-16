"""
CSV input format, `--data-dir`, and the pinned encoder revision.

The CSV folder format is how real VT listings will enter the pipeline, so bad
files must fail with a message naming the file and line, never a half-loaded
dataset.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import research_matching_pipeline as rmp
from data import load_csv_dataset, load_dataset

OPPORTUNITIES = "id,title,description,skills,eligibility\nA,Robots,Build robots,Python; ROS,Juniors\nB,Poems,Read poems,,\n"
PROFILES = "id,query,match_type\nQ1,robots please,exact\nQ2,poetry,paraphrase\n"
RELEVANCE = "profile_id,opportunity_id,grade\nQ1,A,2\nQ1,B,0\nQ2,A,0\nQ2,B,1\n"


def write_dataset(folder, opportunities=OPPORTUNITIES, profiles=PROFILES, relevance=RELEVANCE, encoding="utf-8"):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("opportunities", opportunities), ("profiles", profiles), ("relevance", relevance)):
        (folder / f"{name}.csv").write_text(text, encoding=encoding)
    return folder


def test_valid_folder_loads_all_fields(tmp_path) -> None:
    dataset = load_csv_dataset(write_dataset(tmp_path / "mine"))
    assert dataset.name == "mine"
    assert dataset.opportunity_ids == ["A", "B"]
    assert dataset.opportunities[0].skills == ["Python", "ROS"]
    assert dataset.opportunities[0].eligibility == "Juniors"
    assert dataset.opportunities[1].skills == []
    assert dataset.queries[1].match_type == "paraphrase"
    assert dataset.relevance_for("Q2") == {"A": 0, "B": 1}


def test_excel_byte_order_mark_is_accepted(tmp_path) -> None:
    dataset = load_csv_dataset(write_dataset(tmp_path / "bom", encoding="utf-8-sig"))
    assert dataset.opportunity_ids == ["A", "B"]


@pytest.mark.parametrize(
    "override, message",
    [
        ({"opportunities": "id,title\nA,Robots\n"}, "missing required columns: description"),
        ({"profiles": "id,query\nQ1,\nQ2,poetry\n"}, r"profiles.csv:2: 'query' must not be empty"),
        ({"relevance": RELEVANCE + "Q9,A,1\n"}, "unknown profile or opportunity id"),
        ({"relevance": RELEVANCE + "Q1,A,1\n"}, "duplicate label"),
        ({"relevance": RELEVANCE.replace("Q1,A,2", "Q1,A,3")}, "grade must be 0, 1 or 2"),
        ({"relevance": "profile_id,opportunity_id,grade\nQ1,A,2\n"}, "unlabelled"),
        ({"profiles": "id,query\n"}, "contains no data rows"),
    ],
)
def test_bad_files_are_rejected_with_a_clear_message(tmp_path, override, message) -> None:
    folder = write_dataset(tmp_path / "bad", **override)
    with pytest.raises(ValueError, match=message):
        load_csv_dataset(folder)


def test_missing_file_is_rejected(tmp_path) -> None:
    folder = write_dataset(tmp_path / "partial")
    (folder / "relevance.csv").unlink()
    with pytest.raises(ValueError, match="relevance.csv: file not found"):
        load_csv_dataset(folder)


def test_match_type_filter_subsets_queries_not_corpus(tmp_path) -> None:
    dataset = load_csv_dataset(write_dataset(tmp_path / "f"), match_types=("paraphrase",))
    assert dataset.query_ids == ["Q2"]
    assert dataset.opportunity_ids == ["A", "B"]


def test_unknown_match_type_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown match_type"):
        load_csv_dataset(write_dataset(tmp_path / "f"), match_types=("nope",))


def test_match_type_filter_needs_the_column(tmp_path) -> None:
    folder = write_dataset(tmp_path / "f", profiles="id,query\nQ1,robots\nQ2,poetry\n")
    with pytest.raises(ValueError, match="no match_type column"):
        load_csv_dataset(folder, match_types=("exact",))


def test_main_demo_matches_the_main_branch_files() -> None:
    """main's demo set, carried over verbatim: 12 listings, 5 profiles, full labels."""
    dataset = load_dataset("main_demo")
    assert (len(dataset.opportunities), len(dataset.queries)) == (12, 5)
    assert dataset.relevance_for("p01")["o01"] == 2
    assert sum(dataset.relevance_for("p05").values()) == 0  # the no-match case


def test_heldout_carries_skills_and_query_types() -> None:
    dataset = load_dataset("heldout")
    assert (len(dataset.opportunities), len(dataset.queries)) == (30, 24)
    assert dataset.opportunities[0].skills == ["C++", "ROS", "Control Systems"]
    assert {q.match_type for q in dataset.queries} == {
        "exact", "paraphrase", "vague", "token", "long", "nomatch"
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_data_dir_run_writes_to_a_folder_named_after_the_dataset(tmp_path) -> None:
    folder = write_dataset(tmp_path / "vt_listings")
    out = tmp_path / "out"
    exit_code = rmp.main(["--data-dir", str(folder), "--offline-embeddings", "--output-dir", str(out)])
    assert exit_code == 0
    results = pd.read_csv(out / "vt_listings" / "match_results.csv")
    assert len(results) == 2 * 3 * 2  # 2 profiles x 3 methods x k=2


def test_bad_data_dir_exits_with_a_one_line_error(tmp_path) -> None:
    folder = write_dataset(tmp_path / "bad", relevance="profile_id,opportunity_id,grade\nQ1,A,2\n")
    with pytest.raises(SystemExit) as excinfo:
        rmp.main(["--data-dir", str(folder), "--offline-embeddings", "--output-dir", str(tmp_path)])
    assert str(excinfo.value.code).startswith("Error:")
    assert "unlabelled" in str(excinfo.value.code)


def test_dataset_and_data_dir_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        rmp.main(["--dataset", "prototype", "--data-dir", str(tmp_path), "--offline-embeddings"])
    assert excinfo.value.code == 2  # argparse usage error


def test_config_accepts_unregistered_name_only_with_data_dir() -> None:
    assert rmp.Config(dataset="vt_listings", data_dir="somewhere").dataset == "vt_listings"
    with pytest.raises(ValueError, match="Unknown dataset"):
        rmp.Config(dataset="vt_listings")


# ---------------------------------------------------------------------------
# Pinned encoder revision
# ---------------------------------------------------------------------------


def test_encoder_is_loaded_at_the_pinned_revision(monkeypatch) -> None:
    """An upstream model update must not silently change results."""
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    _, backend = rmp.load_encoder("all-MiniLM-L6-v2", allow_fallback=False)
    assert calls == [("all-MiniLM-L6-v2", {"revision": rmp.EMBEDDING_REVISION})]
    assert backend == f"sentence-transformers::all-MiniLM-L6-v2@{rmp.EMBEDDING_REVISION[:7]}"
    assert rmp.Config().embedding_revision == rmp.EMBEDDING_REVISION
