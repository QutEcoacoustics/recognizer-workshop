"""Tests for the None-handling and NaN/Inf normalisation added to
training_result_to_json in this branch."""

import math

import numpy as np
import pytest

from recognizer_workshop.app.services.training import TrainingResult, training_result_to_json
from ml_collections import config_dict
from perch_hoplite.agile.classifier import LinearClassifier


def _make_result(eval_scores=None, training_info=None) -> TrainingResult:
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((4, 1)).astype(np.float32)
    beta_bias = rng.standard_normal((1,)).astype(np.float32)
    model_config = config_dict.ConfigDict({"model_name": "test"})
    clf = LinearClassifier(beta=beta, beta_bias=beta_bias, classes=("a",), embedding_model_config=model_config)
    return TrainingResult(classifier_model=clf, eval_scores=eval_scores, training_info=training_info)


# ---------------------------------------------------------------------------
# None eval_scores / training_info
# ---------------------------------------------------------------------------

class TestNoneHandling:

    def test_none_eval_scores_returns_metrics_none(self):
        result = _make_result(eval_scores=None, training_info=None)
        out = training_result_to_json(result)
        assert out["metrics"] is None

    def test_none_training_info_returns_info_none(self):
        result = _make_result(eval_scores=None, training_info=None)
        out = training_result_to_json(result)
        assert out["info"] is None

    def test_both_none_returns_both_none(self):
        result = _make_result(eval_scores=None, training_info=None)
        out = training_result_to_json(result)
        assert out == {"metrics": None, "info": None}

    def test_non_none_scores_still_serialised(self):
        result = _make_result(eval_scores={"roc_auc": 0.95}, training_info={"balance": True})
        out = training_result_to_json(result)
        assert out["metrics"]["roc_auc"] == pytest.approx(0.95)
        assert out["info"]["balance"] is True


# ---------------------------------------------------------------------------
# NaN / Inf normalisation
# ---------------------------------------------------------------------------

class TestNonFiniteNormalisation:

    def test_nan_float_in_scores_becomes_none(self):
        result = _make_result(eval_scores={"roc_auc": float("nan")})
        out = training_result_to_json(result)
        assert out["metrics"]["roc_auc"] is None

    def test_inf_float_in_scores_becomes_none(self):
        result = _make_result(eval_scores={"value": float("inf")})
        out = training_result_to_json(result)
        assert out["metrics"]["value"] is None

    def test_neg_inf_float_in_scores_becomes_none(self):
        result = _make_result(eval_scores={"value": float("-inf")})
        out = training_result_to_json(result)
        assert out["metrics"]["value"] is None

    def test_finite_float_unchanged(self):
        result = _make_result(eval_scores={"cmap": 0.75})
        out = training_result_to_json(result)
        assert out["metrics"]["cmap"] == pytest.approx(0.75)

    def test_nan_in_ndarray_becomes_none_element(self):
        arr = np.array([0.9, float("nan"), 0.8], dtype=np.float32)
        result = _make_result(eval_scores={"roc_auc_individual": arr})
        out = training_result_to_json(result)
        values = out["metrics"]["roc_auc_individual"]
        assert values[0] == pytest.approx(0.9, abs=1e-4)
        assert values[1] is None
        assert values[2] == pytest.approx(0.8, abs=1e-4)

    def test_inf_in_ndarray_becomes_none_element(self):
        arr = np.array([float("inf"), 0.5], dtype=np.float32)
        result = _make_result(eval_scores={"values": arr})
        out = training_result_to_json(result)
        assert out["metrics"]["values"][0] is None
        assert out["metrics"]["values"][1] == pytest.approx(0.5, abs=1e-4)

    def test_nan_numpy_floating_becomes_none(self):
        result = _make_result(eval_scores={"x": np.float32("nan")})
        out = training_result_to_json(result)
        assert out["metrics"]["x"] is None

    def test_eval_preds_stripped_from_output(self):
        result = _make_result(eval_scores={
            "roc_auc": 0.9,
            "eval_preds": np.zeros((5, 1), dtype=np.float32),
            "eval_labels": np.zeros((5, 1), dtype=np.float32),
        })
        out = training_result_to_json(result)
        assert "eval_preds" not in out["metrics"]
        assert "eval_labels" not in out["metrics"]

    def test_result_is_json_serialisable(self):
        """Output must be serialisable with strict json.dumps (no NaN/Inf)."""
        import json
        arr = np.array([0.9, float("nan"), float("inf")], dtype=np.float32)
        result = _make_result(
            eval_scores={"roc_auc_individual": arr, "top1_acc": float("nan")},
            training_info={"balance": True},
        )
        out = training_result_to_json(result)
        # Should not raise
        serialised = json.dumps(out)
        assert serialised is not None
