"""Regression tests for inference CSV helpers."""

from pathlib import Path
from types import SimpleNamespace
import csv

import numpy as np
import pandas as pd

from recognizer_workshop.app.agile_modeling_helpers import make_inference_subset
from recognizer_workshop.app.services.inference import inference as inference_service
from recognizer_workshop.app.services.inference.inference import create_inference_subset
from recognizer_workshop.app.inference.inference_result import InferenceResultItem
from recognizer_workshop.perch_hoplite_custom_modifications import classifier as classifier_mod


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


def test_create_inference_subset_max_selects_highest_scores_per_label(tmp_path: Path):
    inference_csv = tmp_path / "inference.csv"
    subset_csv = tmp_path / "inference.max.csv"
    frame = pd.DataFrame(
        [
            {"idx": 1, "label": "alpha", "score": 0.1},
            {"idx": 2, "label": "alpha", "score": 0.9},
            {"idx": 3, "label": "alpha", "score": 0.5},
            {"idx": 4, "label": "beta", "score": 0.2},
            {"idx": 5, "label": "beta", "score": 0.8},
        ]
    )
    frame.to_csv(inference_csv, index=False)

    row_count = create_inference_subset(
        inference_csv,
        subset_csv,
        limit=2,
        mode="max",
    )

    subset = pd.read_csv(subset_csv)
    assert row_count == 4
    assert subset.groupby("label").size().to_dict() == {"alpha": 2, "beta": 2}
    assert subset[subset["label"] == "alpha"]["idx"].tolist() == [2, 3]
    assert subset[subset["label"] == "beta"]["idx"].tolist() == [5, 4]


def test_empty_inference_result_reports_no_detections(tmp_path: Path):
    inference_csv = tmp_path / "empty.csv"
    inference_csv.write_text("idx,label,score\n", encoding="utf-8")
    item = InferenceResultItem(
        csv_path=inference_csv,
        inference_date=pd.Timestamp.now().to_pydatetime(),
        recognizer_name="test",
        recognizer_display_name="test",
        metadata={},
    )

    info = item.get_info()

    assert info["total_row_count"] == 0
    assert info["preview_df"].empty
    assert info["unique_labels"] == []
    assert "error" not in info


def test_inference_csv_writer_quotes_commas_in_fields(tmp_path: Path):
    output_path = tmp_path / "inference.csv"
    source = SimpleNamespace(
        dataset_name="search_set",
        source_id="https://example.org/audio/1",
        offsets=(2.0,),
    )
    db = SimpleNamespace(get_embedding_source=lambda _idx: source)
    state = SimpleNamespace(
        csv_filepath=output_path,
        labels=("alpha",),
        threshold=0.0,
        row_func=lambda _row: ["Albany Creek Reserve, Wendon Way"],
        get_thread_db=lambda: db,
    )

    initializer = classifier_mod.csv_worker_initializer_factory(
        lambda: ["site_name"]
    )
    initializer(state)
    classifier_mod.csv_worker_fn(
        np.array([1]),
        np.array([[0.5]], dtype=np.float32),
        state,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == [
        "idx", "dataset_name", "source_id", "offset", "label", "score", "site_name"
    ]
    assert rows[1][-1] == "Albany Creek Reserve, Wendon Way"
    assert len(rows[1]) == len(rows[0])


def test_run_inference_continues_without_site_metadata(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(
        inference_service.brutalism,
        "get_brute_search_ids",
        lambda db, subset, exclude_labeled=False: np.array([1]),
    )
    monkeypatch.setattr(
        inference_service,
        "make_site_map",
        lambda db, baw_api: (_ for _ in ()).throw(RuntimeError("unauthorized")),
    )
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
        captured["header"] = row_func()
        captured["row"] = row_func([1, "dataset", "source", 2.0, "alpha", 0.5])

    monkeypatch.setattr(inference_service.classifier_mod, "write_inference_csv", fake_write_inference_csv)

    warnings = []
    inference_service.run_inference(
        classifier=SimpleNamespace(),
        db=SimpleNamespace(),
        output_filepath=tmp_path / "inference.csv",
        baw_api=SimpleNamespace(),
        subset=1.0,
        warning_callback=warnings.append,
    )

    assert captured["header"] == ["end_offset", "audio_recording_id", "link"]
    assert captured["row"] == [7.0, 123, "https://example.org/listen"]
    assert warnings == ["Site information was not retrieved: unauthorized"]


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
