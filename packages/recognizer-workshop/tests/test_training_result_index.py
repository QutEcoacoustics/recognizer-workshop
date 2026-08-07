"""Tests for SavedTrainingResultsIndex and related helpers."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from ml_collections import config_dict

from perch_hoplite.agile.classifier import LinearClassifier

from recognizer_workshop.app.services.training_result_index import (
    SavedTrainingResultsIndex,
    SavedTrainingResultsItem,
    load_training_result_from_classifier_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_classifier(embedding_dim: int = 8, classes: tuple = ("a",)) -> LinearClassifier:
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((embedding_dim, len(classes))).astype(np.float32)
    beta_bias = rng.standard_normal((len(classes),)).astype(np.float32)
    model_config = config_dict.ConfigDict({"model_name": "test"})
    return LinearClassifier(beta=beta, beta_bias=beta_bias, classes=classes, embedding_model_config=model_config)


def _write_version(models_root: Path, classifier_name: str, version_name: str, *, metrics: dict | None = None) -> Path:
    """Create a minimal versioned classifier directory on disk."""
    version_dir = models_root / classifier_name / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    recognizer_path = version_dir / "recognizer.json"
    _make_linear_classifier().save(recognizer_path)
    if metrics is not None:
        (version_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return version_dir


# ---------------------------------------------------------------------------
# SavedTrainingResultsItem.display_name
# ---------------------------------------------------------------------------

class TestDisplayName:

    def test_date_format_returns_human_readable(self, tmp_path):
        version_dir = tmp_path / "owl" / "20240315_143022"
        version_dir.mkdir(parents=True)
        item = SavedTrainingResultsItem(
            classifier_name="owl",
            version_name="20240315_143022",
            version_dir=version_dir,
            modified_time=datetime.now(),
        )
        assert item.display_name == "2024-03-15 14:30:22"

    def test_non_date_name_returns_raw(self, tmp_path):
        version_dir = tmp_path / "owl" / "my_custom_version"
        version_dir.mkdir(parents=True)
        item = SavedTrainingResultsItem(
            classifier_name="owl",
            version_name="my_custom_version",
            version_dir=version_dir,
            modified_time=datetime.now(),
        )
        assert item.display_name == "my_custom_version"

    def test_partial_date_falls_back_to_raw(self, tmp_path):
        version_dir = tmp_path / "owl" / "20240315"
        version_dir.mkdir(parents=True)
        item = SavedTrainingResultsItem(
            classifier_name="owl",
            version_name="20240315",
            version_dir=version_dir,
            modified_time=datetime.now(),
        )
        assert item.display_name == "20240315"


# ---------------------------------------------------------------------------
# SavedTrainingResultsIndex — scanning
# ---------------------------------------------------------------------------

class TestIndexScan:

    def test_empty_models_folder_returns_empty(self, tmp_path):
        index = SavedTrainingResultsIndex(tmp_path / "models")
        index.initialize_index()
        assert index.entries == {}

    def test_nonexistent_folder_returns_empty(self, tmp_path):
        index = SavedTrainingResultsIndex(tmp_path / "does_not_exist")
        index.initialize_index()
        assert index.entries == {}

    def test_none_folder_returns_empty(self):
        index = SavedTrainingResultsIndex(None)
        index.initialize_index()
        assert index.entries == {}

    def test_finds_classifier_and_version(self, tmp_path):
        _write_version(tmp_path, "owl", "20240101_120000")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        assert "owl" in index.entries
        assert "20240101_120000" in index.entries["owl"]

    def test_ignores_version_dir_without_recognizer_json(self, tmp_path):
        (tmp_path / "owl" / "empty_version").mkdir(parents=True)
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        assert index.entries == {}

    def test_ignores_top_level_files(self, tmp_path):
        (tmp_path / "some_file.json").write_text("{}")
        _write_version(tmp_path, "owl", "20240101_120000")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        assert list(index.entries.keys()) == ["owl"]

    def test_multiple_classifiers(self, tmp_path):
        _write_version(tmp_path, "owl", "20240101_120000")
        _write_version(tmp_path, "frog", "20240102_120000")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        assert set(index.entries.keys()) == {"owl", "frog"}

    def test_multiple_versions_same_classifier(self, tmp_path):
        _write_version(tmp_path, "owl", "20240101_120000")
        _write_version(tmp_path, "owl", "20240102_120000")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        assert set(index.entries["owl"].keys()) == {"20240101_120000", "20240102_120000"}


# ---------------------------------------------------------------------------
# SavedTrainingResultsIndex — ordering
# ---------------------------------------------------------------------------

class TestIndexOrdering:

    def test_get_versions_newest_first(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        time.sleep(0.02)
        _write_version(tmp_path, "owl", "v2")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        versions = index.get_versions_for_classifier("owl")
        assert versions[0].version_name == "v2"
        assert versions[1].version_name == "v1"

    def test_get_classifier_list_newest_first(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        time.sleep(0.02)
        _write_version(tmp_path, "frog", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        names = index.get_classifier_list()
        assert names[0] == "frog"
        assert names[1] == "owl"

    def test_classifier_ordered_by_newest_version_not_oldest(self, tmp_path):
        # owl has an old v1 and a very new v3; frog has only v2 (between them)
        _write_version(tmp_path, "owl", "v1")
        time.sleep(0.02)
        _write_version(tmp_path, "frog", "v2")
        time.sleep(0.02)
        _write_version(tmp_path, "owl", "v3")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        assert index.get_classifier_list()[0] == "owl"


# ---------------------------------------------------------------------------
# SavedTrainingResultsIndex — active selection
# ---------------------------------------------------------------------------

class TestActiveSelection:

    def test_initialize_selects_globally_newest(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        time.sleep(0.02)
        _write_version(tmp_path, "frog", "v2")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        active = index.get_active_classifier_entry()
        assert active.classifier_name == "frog"
        assert active.version_name == "v2"

    def test_empty_index_returns_none_active(self, tmp_path):
        index = SavedTrainingResultsIndex(tmp_path / "empty")
        index.initialize_index()
        assert index.get_active_classifier_entry() is None

    def test_set_active_classifier_updates_selection(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        _write_version(tmp_path, "owl", "v2")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        index.set_active_classifier("owl", "v1")
        assert index.get_active_classifier_entry().version_name == "v1"

    def test_set_active_classifier_raises_for_unknown(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        with pytest.raises(ValueError, match="No saved classifier"):
            index.set_active_classifier("owl", "nonexistent")

    def test_set_active_classifier_raises_for_unknown_name(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        with pytest.raises(ValueError):
            index.set_active_classifier("does_not_exist", "v1")


# ---------------------------------------------------------------------------
# SavedTrainingResultsIndex — delete
# ---------------------------------------------------------------------------

class TestDeleteVersion:

    def test_delete_removes_directory(self, tmp_path):
        version_dir = _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        index.delete_classifier_version("owl", "v1")
        assert not version_dir.exists()

    def test_delete_rescans_and_reselects(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        time.sleep(0.02)
        _write_version(tmp_path, "owl", "v2")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        index.delete_classifier_version("owl", "v2")
        active = index.get_active_classifier_entry()
        assert active.version_name == "v1"

    def test_delete_last_version_leaves_empty_index(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        index.delete_classifier_version("owl", "v1")
        assert index.is_empty
        assert index.get_active_classifier_entry() is None

    def test_delete_nonexistent_raises(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        with pytest.raises(ValueError):
            index.delete_classifier_version("owl", "nonexistent")


# ---------------------------------------------------------------------------
# SavedTrainingResultsIndex.is_empty
# ---------------------------------------------------------------------------

class TestIsEmpty:

    def test_empty_when_no_classifiers(self, tmp_path):
        index = SavedTrainingResultsIndex(tmp_path / "models")
        index.initialize_index()
        assert index.is_empty

    def test_not_empty_when_has_classifiers(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        assert not index.is_empty


# ---------------------------------------------------------------------------
# load_training_result_from_classifier_entry
# ---------------------------------------------------------------------------

class TestLoadTrainingResultFromEntry:

    def test_loads_classifier_model(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        entry = index.get_active_classifier_entry()
        result = load_training_result_from_classifier_entry(entry)
        assert result.classifier_model is not None

    def test_no_metrics_file_gives_none_scores(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")  # no metrics kwarg
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        entry = index.get_active_classifier_entry()
        result = load_training_result_from_classifier_entry(entry)
        assert result.eval_scores is None
        assert result.training_info is None

    def test_with_metrics_file_loads_scores_and_info(self, tmp_path):
        metrics = {"metrics": {"roc_auc": 0.95}, "info": {"balance": True}}
        _write_version(tmp_path, "owl", "v1", metrics=metrics)
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        entry = index.get_active_classifier_entry()
        result = load_training_result_from_classifier_entry(entry)
        assert result.eval_scores == {"roc_auc": 0.95}
        assert result.training_info == {"balance": True}

    def test_sets_classifier_name_and_version_on_result(self, tmp_path):
        _write_version(tmp_path, "owl", "v1")
        index = SavedTrainingResultsIndex(tmp_path)
        index.initialize_index()
        entry = index.get_active_classifier_entry()
        result = load_training_result_from_classifier_entry(entry)
        assert result.classifier_name == "owl"
        assert result.version_name == "v1"
