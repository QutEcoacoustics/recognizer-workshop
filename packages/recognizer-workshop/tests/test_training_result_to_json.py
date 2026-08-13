"""Tests for the None-handling and NaN/Inf normalisation added to
training_result_to_json in this branch."""

import math
import json
from dataclasses import asdict

import numpy as np
import pytest

from recognizer_workshop.app.services.training import (
    TrainClassifierParams,
    TrainingResult,
    training_result_to_json,
)
from recognizer_workshop.app.services.json_utils import to_json_safe
from ml_collections import config_dict
from perch_hoplite.agile.classifier import LinearClassifier


def _make_result(eval_scores=None, training_info=None) -> TrainingResult:
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((4, 1)).astype(np.float32)
    beta_bias = rng.standard_normal((1,)).astype(np.float32)
    model_config = config_dict.ConfigDict({"model_name": "test"})
    clf = LinearClassifier(beta=beta, beta_bias=beta_bias, classes=("a",), embedding_model_config=model_config)
    return TrainingResult(
        classifier_model=clf,
        info={'metrics': eval_scores, 'info': training_info},
    )


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


def test_to_json_safe_handles_nested_numpy_values_and_non_finite_numbers():
    value = {
        'array': np.array([0.5, np.nan]),
        'numpy_float': np.float64(np.inf),
        'tuple': (np.int64(3), float('-inf')),
    }

    result = to_json_safe(value)

    assert result == {
        'array': [0.5, None],
        'numpy_float': None,
        'tuple': [3, None],
    }
    json.dumps(result, allow_nan=False)


def test_training_params_are_serialized_when_present():
    result = _make_result()
    result = TrainingResult(
        classifier_model=result.classifier_model,
        info={'params': asdict(TrainClassifierParams(train_ratio=1.0, k_folds=1))},
    )

    output = training_result_to_json(result)

    assert output['params']['train_ratio'] == 1.0
    assert output['params']['k_folds'] == 1


def test_structured_training_info_serializes_evaluation_and_final_model():
    result = _make_result()
    metrics = {
        'top1_acc': 0.9,
        'train_ratio': 0.9,
    }
    result = TrainingResult(
        classifier_model=result.classifier_model,
        info={
            'evaluation': {
            'average_metrics': metrics,
            'average_data_counts': {'train': {}, 'test': {}},
            'folds': [],
            'params': {'train_ratio': 0.9},
            },
            'final_model': {
                'data_info': {
                    'data_counts': {'train': {}, 'test': {}},
                    'balancing_debug': {'balance_dataset_enabled': True},
                },
                'params': asdict(TrainClassifierParams(train_ratio=1.0)),
            },
        },
    )

    output = training_result_to_json(result)

    assert output['evaluation']['average_metrics'] == metrics
    assert output['evaluation']['average_data_counts'] == {'train': {}, 'test': {}}
    assert output['final_model']['data_info']['balancing_debug']['balance_dataset_enabled'] is True
    assert output['final_model']['params']['train_ratio'] == 1.0
