"""
Web interface: type a student's interests and compare each method's ranked
listings side by side, along with each method's scores on the whole dataset.

    python app.py        then open http://localhost:8000
"""

from __future__ import annotations

from flask import Flask, render_template, request

import irdm

app = Flask(__name__)
irdm.load_model()  # load once at startup so the first page is fast

GRADE_NAMES = {0: "unsuitable", 1: "partial", 2: "strong"}


@app.get("/")
def index():
    datasets = irdm.list_datasets()
    name = request.args.get("dataset", irdm.DEFAULT_DATASET)
    if name not in datasets:
        name = irdm.DEFAULT_DATASET
    query = request.args.get("query", "").strip()
    weight = request.args.get("weight", 0.6, type=float)
    weight = weight if 0 <= weight <= 1 else 0.6
    page = dict(datasets=datasets, name=name, query=query, weight=weight, k=5, grade_names=GRADE_NAMES)

    try:
        dataset = irdm.load_dataset(name)
    except ValueError as exc:  # a malformed dataset folder: show what is wrong instead of failing
        return render_template("index.html", error=str(exc), **page)

    n_profiles, n_listings = dataset.grades.shape
    k = min(max(request.args.get("k", 5, type=int), 1), n_listings)
    profile_id, results = irdm.search(dataset, query, k, weight) if query else (None, None)
    _, per_profile, summary = irdm.evaluate(dataset, k, weight)
    by_type = irdm.by_match_type(dataset, per_profile)
    page.update(
        k=k,
        n_listings=n_listings,
        n_profiles=n_profiles,
        profiles=dataset.profiles.to_dict("records"),
        profile_id=profile_id,
        results=results,
        summary=summary.to_dict("records"),
        best_ndcg=summary["mean_ndcg@k"].max(),
        n_scorable=int((dataset.grades >= 1).any(axis=1).sum()),
        by_type=None if by_type is None else by_type.reset_index().values.tolist(),
        methods=list(summary["method"]),
    )
    return render_template("index.html", **page)


if __name__ == "__main__":
    app.run(port=8000)
