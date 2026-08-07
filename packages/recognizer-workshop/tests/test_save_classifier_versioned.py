"""Tests for save_classifier_versioned, sanitize_classifier_name, and the
engine-level index management methods (refresh, set_active, delete_active)."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from ml_collections import config_dict

from perch_hoplite.agile.classifier import LinearClassifier

from recognizer_workshop.app.agile_config import AgileConfig
from recognizer_workshop.app.agile_engine import agile_state, sanitize_classifier_name
from recognizer_workshop.app.services.training import TrainingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_classifier(embedding_dim: int = 8, classes: tuple = ("a", "b")) -> LinearClassifier:
    rng = np.random.default_rng(42)
    beta = rng.standard_normal((embedding_dim, len(classes))).astype(np.float32)
    beta_bias = rng.standard_normal((len(classes),)).astype(np.float32)
    model_config = config_dict.ConfigDict({"model_name": "test_model"})
    return LinearClassifier(beta=beta, beta_bias=beta_bias, classes=classes, embedding_model_config=model_config)


@pytest.fixture
def engine(tmp_path):
    config = AgileConfig(
        db_path=tmp_path / "db",
        models_folder=tmp_path / "models",
    )
    eng = agile_state(config)
    eng.train_result = TrainingResult(
        classifier_model=_make_linear_classifier(),
        eval_scores={"roc_auc": 0.9},
        training_info={"resolved_weak_negatives_batch_size": 16},
    )
    return eng, tmp_path


def _make_train_result_json() -> dict:
    return {"metrics": {"roc_auc": 0.9}, "info": {"balance": True}}


# ---------------------------------------------------------------------------
# sanitize_classifier_name
# ---------------------------------------------------------------------------

class TestSanitizeClassifierName:

    def test_lowercases(self):
        assert sanitize_classifier_name("MyBird") == "mybird"

    def test_replaces_spaces_with_underscore(self):
        assert sanitize_classifier_name("my bird") == "my_bird"

    def test_replaces_hyphens_with_underscore(self):
        assert sanitize_classifier_name("my-bird") == "my_bird"

    def test_removes_special_characters(self):
        assert sanitize_classifier_name("bird!@#$%") == "bird"

    def test_allows_dots_and_digits(self):
        assert sanitize_classifier_name("owl_v1.2") == "owl_v1.2"

    def test_strips_leading_trailing_dots_underscores(self):
        # strip('._') strips both dots and underscores from both ends
        assert sanitize_classifier_name("._bird_.") == "bird"

    def test_empty_string_returns_classifier(self):
        assert sanitize_classifier_name("") == "classifier"

    def test_only_special_chars_returns_classifier(self):
        assert sanitize_classifier_name("!@#$") == "classifier"

    def test_multiple_spaces_become_single_underscore(self):
        # runs of whitespace/hyphens collapse to a single underscore
        assert sanitize_classifier_name("my  bird") == "my_bird"

    def test_mixed_hyphens_and_spaces(self):
        # the whole run " - " is one match of [\s\-]+, so becomes a single underscore
        assert sanitize_classifier_name("my - bird") == "my_bird"


# ---------------------------------------------------------------------------
# save_classifier_versioned — directory structure
# ---------------------------------------------------------------------------

class TestSaveClassifierVersioned:

    def test_creates_recognizer_json(self, engine):
        eng, tmp_path = engine
        result_json = _make_train_result_json()
        version_dir = eng.save_classifier_versioned(result_json, "owl")
        assert (version_dir / "recognizer.json").exists()

    def test_creates_metrics_json(self, engine):
        eng, tmp_path = engine
        result_json = _make_train_result_json()
        version_dir = eng.save_classifier_versioned(result_json, "owl")
        assert (version_dir / "metrics.json").exists()

    def test_metrics_json_content_matches_input(self, engine):
        eng, tmp_path = engine
        result_json = _make_train_result_json()
        version_dir = eng.save_classifier_versioned(result_json, "owl")
        saved = json.loads((version_dir / "metrics.json").read_text())
        assert saved["metrics"]["roc_auc"] == 0.9

    def test_directory_structure_is_models_name_version(self, engine):
        eng, tmp_path = engine
        result_json = _make_train_result_json()
        version_dir = eng.save_classifier_versioned(result_json, "little_owl")
        assert version_dir.parent.name == "little_owl"
        assert version_dir.parent.parent == (tmp_path / "models").resolve()

    def test_returns_path_object(self, engine):
        eng, tmp_path = engine
        version_dir = eng.save_classifier_versioned(_make_train_result_json(), "owl")
        assert isinstance(version_dir, Path)

    def test_sanitizes_classifier_name(self, engine):
        eng, tmp_path = engine
        version_dir = eng.save_classifier_versioned(_make_train_result_json(), "My Owl!")
        assert version_dir.parent.name == "my_owl"

    def test_injects_classifier_name_into_recognizer_json(self, engine):
        eng, tmp_path = engine
        version_dir = eng.save_classifier_versioned(_make_train_result_json(), "owl")
        rec = json.loads((version_dir / "recognizer.json").read_text())
        assert rec["classifier_name"] == "owl"

    def test_path_traversal_via_name_defused_by_sanitizer(self, engine):
        # sanitize_classifier_name strips slashes and dots, so "../../evil" → "evil";
        # the traversal guard never needs to fire because the name is already safe.
        eng, tmp_path = engine
        version_dir = eng.save_classifier_versioned(_make_train_result_json(), "../../evil")
        assert version_dir.parent.name == "evil"
        assert str(version_dir).startswith(str((tmp_path / "models").resolve()))


# ---------------------------------------------------------------------------
# Engine index management methods
# ---------------------------------------------------------------------------

class TestEngineIndexManagement:

    def _write_version(self, models_root: Path, name: str, version: str) -> Path:
        """Write a minimal versioned classifier without using the engine."""
        version_dir = models_root / name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        _make_linear_classifier().save(version_dir / "recognizer.json")
        (version_dir / "metrics.json").write_text(
            json.dumps({"metrics": {"roc_auc": 0.8}, "info": {}}), encoding="utf-8"
        )
        return version_dir

    def test_refresh_index_discovers_existing_versions(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1")
        eng.refresh_saved_training_results_index()
        index = eng.saved_training_results_index
        assert "owl" in index.entries

    def test_refresh_index_loads_active_train_result(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1")
        eng.refresh_saved_training_results_index()
        assert eng.train_result is not None
        assert eng.train_result.classifier_model is not None

    def test_refresh_index_sets_train_result_none_when_empty(self, engine):
        eng, tmp_path = engine
        # models folder exists but is empty
        (tmp_path / "models").mkdir(exist_ok=True)
        eng.refresh_saved_training_results_index()
        assert eng.train_result is None

    def test_set_active_saved_classifier_changes_train_result(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1")
        time.sleep(0.02)
        self._write_version(tmp_path / "models", "owl", "v2")
        eng.refresh_saved_training_results_index()
        # index auto-selects v2 (newest); manually switch to v1
        eng.set_active_saved_classifier("owl", "v1")
        assert eng.saved_training_results_index.selected_classifier_version == "v1"
        assert eng.train_result.version_name == "v1"

    def test_set_active_raises_when_index_not_initialized(self, engine):
        eng, _ = engine
        eng.saved_training_results_index = None
        with pytest.raises(ValueError, match="not initialized"):
            eng.set_active_saved_classifier("owl", "v1")

    def test_delete_active_removes_version(self, engine):
        eng, tmp_path = engine
        version_dir = self._write_version(tmp_path / "models", "owl", "v1")
        eng.refresh_saved_training_results_index()
        eng.delete_active_saved_classifier_version()
        assert not version_dir.exists()

    def test_delete_active_reselects_remaining_version(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1")
        time.sleep(0.02)
        self._write_version(tmp_path / "models", "owl", "v2")
        eng.refresh_saved_training_results_index()
        # v2 is active (newest); delete it → v1 should become active
        eng.delete_active_saved_classifier_version()
        assert eng.saved_training_results_index.selected_classifier_version == "v1"

    def test_delete_active_when_no_active_raises(self, engine):
        eng, tmp_path = engine
        (tmp_path / "models").mkdir(exist_ok=True)
        eng.refresh_saved_training_results_index()
        with pytest.raises(ValueError, match="No active"):
            eng.delete_active_saved_classifier_version()

    def test_delete_active_raises_when_index_not_initialized(self, engine):
        eng, _ = engine
        eng.saved_training_results_index = None
        with pytest.raises(ValueError, match="not initialized"):
            eng.delete_active_saved_classifier_version()


class TestCorruptClassifierHandling:

    def _write_version(self, models_root: Path, name: str, version: str, *, corrupt: bool = False) -> Path:
        version_dir = models_root / name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        if corrupt:
            (version_dir / "recognizer.json").write_text("not valid json", encoding="utf-8")
        else:
            _make_linear_classifier().save(version_dir / "recognizer.json")
        return version_dir

    def test_corrupt_recognizer_does_not_crash_refresh(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1", corrupt=True)
        # must not raise
        eng.refresh_saved_training_results_index()

    def test_corrupt_recognizer_sets_train_result_none(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1", corrupt=True)
        eng.refresh_saved_training_results_index()
        assert eng.train_result is None

    def test_corrupt_recognizer_sets_load_error_message(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1", corrupt=True)
        eng.refresh_saved_training_results_index()
        assert eng.train_result_load_error is not None
        assert isinstance(eng.train_result_load_error, str)

    def test_valid_load_clears_error(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1", corrupt=True)
        eng.refresh_saved_training_results_index()
        assert eng.train_result_load_error is not None
        # now write a valid version and refresh
        self._write_version(tmp_path / "models", "owl", "v2")
        eng.refresh_saved_training_results_index()
        assert eng.train_result_load_error is None
        assert eng.train_result is not None

    def test_corrupt_recognizer_index_still_has_entry(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1", corrupt=True)
        eng.refresh_saved_training_results_index()
        # index still knows about the entry so user can see/delete it
        assert eng.saved_training_results_index.get_active_classifier_entry() is not None

    def test_set_active_corrupt_sets_error(self, engine):
        eng, tmp_path = engine
        self._write_version(tmp_path / "models", "owl", "v1")
        self._write_version(tmp_path / "models", "owl", "v2", corrupt=True)
        eng.refresh_saved_training_results_index()
        eng.set_active_saved_classifier("owl", "v2")
        assert eng.train_result is None
        assert eng.train_result_load_error is not None
