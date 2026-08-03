"""Tests for agile_state.save_classifier."""

import json

import numpy as np
import pytest
from ml_collections import config_dict

from perch_hoplite.agile.classifier import LinearClassifier
from recognizer_workshop.app.services.training import TrainingResult

from recognizer_workshop.app.agile_config import AgileConfig
from recognizer_workshop.app.agile_engine import agile_state


def _make_linear_classifier(embedding_dim: int = 8, classes: tuple = ("a", "b")) -> LinearClassifier:
    rng = np.random.default_rng(42)
    beta = rng.standard_normal((embedding_dim, len(classes))).astype(np.float32)
    beta_bias = rng.standard_normal((len(classes),)).astype(np.float32)
    model_config = config_dict.ConfigDict({"model_name": "test_model"})
    return LinearClassifier(beta=beta, beta_bias=beta_bias, classes=classes, embedding_model_config=model_config)


@pytest.fixture
def engine_with_classifier(tmp_path):
    config = AgileConfig(
        db_path=tmp_path / "db",
        models_folder=tmp_path / "models",
    )
    eng = agile_state(config)
    eng.train_result = TrainingResult(
        classifier_model=_make_linear_classifier(),
        eval_scores={},
        training_info={},
    )
    return eng, tmp_path


class TestSaveClassifier:

    def test_creates_file(self, engine_with_classifier):
        eng, tmp_path = engine_with_classifier
        out = tmp_path / "models" / "classifier.json"
        eng.save_classifier(out)
        assert out.exists()

    def test_default_path_uses_models_folder(self, engine_with_classifier):
        eng, tmp_path = engine_with_classifier
        eng.save_classifier()
        assert (tmp_path / "models" / "classifier.json").exists()

    def test_output_is_valid_json(self, engine_with_classifier):
        eng, tmp_path = engine_with_classifier
        out = tmp_path / "classifier.json"
        eng.save_classifier(out)
        with out.open("r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_output_is_pretty_printed(self, engine_with_classifier):
        """File should contain newlines and indentation, not be a single line."""
        eng, tmp_path = engine_with_classifier
        out = tmp_path / "classifier.json"
        eng.save_classifier(out)
        text = out.read_text(encoding="utf-8")
        assert "\n" in text, "Expected newlines (pretty-printed JSON)"
        assert "  " in text, "Expected indentation (pretty-printed JSON)"

    def test_ends_with_newline(self, engine_with_classifier):
        eng, tmp_path = engine_with_classifier
        out = tmp_path / "classifier.json"
        eng.save_classifier(out)
        assert out.read_text(encoding="utf-8").endswith("\n")

    def test_roundtrip_loads_correctly(self, engine_with_classifier):
        """Saved file must be loadable by LinearClassifier.load with equal weights."""
        eng, tmp_path = engine_with_classifier        
        original = eng.get_trained_classifier()
        out = tmp_path / "classifier.json"
        eng.save_classifier(out)
        loaded = LinearClassifier.load(str(out))
        np.testing.assert_allclose(loaded.beta, original.beta)
        np.testing.assert_allclose(loaded.beta_bias, original.beta_bias)
        # LinearClassifier.load deserialises classes as a list; compare values not type
        assert list(loaded.classes) == list(original.classes)

    def test_creates_parent_directories(self, engine_with_classifier):
        eng, tmp_path = engine_with_classifier
        out = tmp_path / "deep" / "nested" / "classifier.json"
        eng.save_classifier(out)
        assert out.exists()
