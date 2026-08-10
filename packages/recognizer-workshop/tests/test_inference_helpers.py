"""Regression tests for inference CSV helpers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from recognizer_workshop.app.agile_modeling_helpers import make_inference_subset
from recognizer_workshop.app.services.inference import inference as inference_service


def test_make_inference_subset_uses_inference_score_column(tmp_path: Path):
    inference_csv = tmp_path / "inference.csv"
    subset_csv = tmp_path / "inference.3.csv"

    rows = [
        {"idx": index, "label": label, "score": score}
        for label in ("alpha", "beta")
        for index, score in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0), start=1)
    ]
    pd.DataFrame(rows).to_csv(inference_csv, index=False)

    make_inference_subset(inference_csv, subset_csv, limit=3)

    subset = pd.read_csv(subset_csv)
    assert list(subset.columns) == ["idx", "label", "score"]
    assert len(subset) <= 6
    assert subset.groupby("label").size().le(3).all()


def test_run_inference_writes_blank_site_fields_when_metadata_is_missing(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(
        inference_service.brutalism,
        "get_brute_search_ids",
        lambda db, subset, exclude_labeled=False: np.array([1]),
    )
    monkeypatch.setattr(inference_service, "make_site_map", lambda db, baw_api: {})
    monkeypatch.setattr(
        inference_service.baw_utils,
        "extract_arid_and_domain",
        lambda source_id: ("https://example.org", 123),
    )
    monkeypatch.setattr(
        inference_service.baw_utils,
        "make_baw_listen_link_from_arid",
        lambda *args: "https://example.org/listen",
    )

    def fake_write_inference_csv(*args, row_func, **kwargs):
        captured["row"] = row_func([1, "dataset", "source", 2.0, "alpha", 0.5])

    monkeypatch.setattr(inference_service.classifier_mod, "write_inference_csv", fake_write_inference_csv)

    inference_service.run_inference(
        classifier=SimpleNamespace(),
        db=SimpleNamespace(),
        output_filepath=tmp_path / "inference.csv",
        baw_api=SimpleNamespace(),
        subset=1.0,
    )

    assert captured["row"] == [7.0, 123, "https://example.org/listen", "", ""]
