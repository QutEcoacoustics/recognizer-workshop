"""Optuna-based stateful regularization tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import optuna

from recognizer_workshop.app.services import training
from recognizer_workshop.app.services.json_utils import to_json_safe

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegularizationTuningResult:
    best_params: training.TrainClassifierParams
    best_cross_validation_result: training.CrossValidationResult
    trial_history: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            'best_params': asdict(self.best_params),
            'best_cross_validation_result': self.best_cross_validation_result.to_dict(),
            'trial_history': list(self.trial_history),
        }


def state_key(
    classifier_name: str | None,
    target_labels,
    params: training.TrainClassifierParams,
) -> str:
    return json.dumps(
        {
            "classifier_name": classifier_name or "",
            "target_labels": list(map(str, target_labels)),
            "training_recipe": {
                "learning_rate": params.learning_rate,
                "weak_neg_weight": params.weak_neg_weight,
                "num_steps": params.num_steps,
                "train_ratio": params.train_ratio,
                "batch_size": params.batch_size,
                "weak_negatives_batch_size": params.weak_negatives_batch_size,
                "seed": params.seed,
                "regularize_bias": params.regularize_bias,
                "balance_dataset": bool(params.balance_dataset),
            },
        },
        sort_keys=True,
    )


def _load_state(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def delete_tuning_state(
    path: Path | str | None,
    classifier_name: str | None,
    target_labels,
    params: training.TrainClassifierParams,
) -> bool:
    """Delete only the tuning entry matching one stable training recipe."""
    state_path = Path(path) if path else None
    if state_path is None:
        return False

    state = _load_state(state_path)
    key = state_key(classifier_name, target_labels, params)
    if key not in state:
        return False

    del state[key]
    _save_state(state_path, state)
    return True


def tune_regularization(
    db,
    target_labels,
    params: training.TrainClassifierParams,
    fold_splits: Sequence[training.PreparedDataSplit],
    classifier_name: str | None = None,
    state_path: Path | str | None = None,
) -> RegularizationTuningResult:
    """
    Tune L1/L2 against one fixed collection of validation splits.

    Args:
        db: The database to use for training.
        target_labels: The labels to train on.
        params: The training parameters to use, except for L1/L2.
        fold_splits: The pre-split training/validation data to use.
        classifier_name: Optional name of the classifier, used for state persistence.
        state_path: Optional path to a JSON file to persist tuning state.

    Returns:
        A RegularizationTuningResult containing the best parameters, metrics, and trial history.

    Uses optuna for hyperparameter optimization. Persists tuning state to a JSON file
    so that future active learning runs can continue tuning from previous results. 
    The state is keyed by the classifier name, target labels, and stable training recipe.
    
    """
    state_file = Path(state_path) if state_path else None
    state = _load_state(state_file)
    key = state_key(classifier_name, target_labels, params)
    previous = state.get(key, {})
    base_params = replace(params, auto_tune_regularization=False)
    trial_history = []
    trial_results = {}

    def record_trial(trial_number, trial_params, cross_validation):
        metrics = cross_validation.average_metrics
        trial_history.append({
            'trial': trial_number,
            'loss': float(metrics.validation_loss),
            'params': asdict(trial_params),
            'metrics': metrics.to_dict(),
            'data_counts': cross_validation.average_data_counts,
        })
        if trial_number != 'baseline':
            trial_results[trial_number] = (
                trial_params,
                cross_validation,
            )
        return float(metrics.validation_loss)

    def _bounds(previous_value):
        if previous_value is None or previous_value <= 0:
            return 1e-6, 1.0
        return max(1e-6, previous_value / 10.0), min(1.0, previous_value * 10.0)

    previous_params = previous.get('best_params', {})
    previous_l1 = previous.get('l1_strength', previous_params.get('l1_strength'))
    previous_l2 = previous.get('l2_strength', previous_params.get('l2_strength'))
    l1_low, l1_high = _bounds(previous_l1)
    l2_low, l2_high = _bounds(previous_l2)
    sampler = optuna.samplers.TPESampler(seed=params.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    if (
        previous_l1 is not None
        and previous_l2 is not None
    ):
        study.enqueue_trial({
            "l1_strength": max(float(previous_l1), l1_low),
            "l2_strength": max(float(previous_l2), l2_low),
        })

    def objective(trial):
        l1 = trial.suggest_float("l1_strength", l1_low, l1_high, log=True)
        l2 = trial.suggest_float("l2_strength", l2_low, l2_high, log=True)
        trial_params = replace(base_params, l1_strength=l1, l2_strength=l2)
        cv = training.cross_validate_classifier(
            db=db,
            target_labels=target_labels,
            params=trial_params,
            fold_splits=fold_splits,
        )
        return record_trial(trial.number, trial_params, cv)

    # Always compare against no regularization, including when the previous
    # search found positive strengths.
    baseline_params = replace(base_params, l1_strength=0.0, l2_strength=0.0)
    baseline_cv = training.cross_validate_classifier(
        db=db,
        target_labels=target_labels,
        params=baseline_params,
        fold_splits=fold_splits,
    )
    baseline_loss = record_trial('baseline', baseline_params, baseline_cv)
    requested_trials = max(1, int(params.tuning_trials))
    # only use half the number of tuning trials if we have a saved state, to save time. 
    trial_count = max(1, requested_trials // 2) if previous else requested_trials
    study.optimize(objective, n_trials=trial_count)

    best_trial = study.best_trial
    best_params, best_cross_validation_result = trial_results[best_trial.number]
    best_metrics = best_cross_validation_result.average_metrics
    log.info(
        "Regularization baseline validation loss: %.6f (l1=0, l2=0)",
        baseline_loss,
    )
    log.info(
        "Regularization best Optuna trial %s: validation loss %.6f (l1=%.8g, l2=%.8g)",
        best_trial.number,
        best_metrics.validation_loss,
        best_params.l1_strength,
        best_params.l2_strength,
    )
    if baseline_loss <= best_metrics.validation_loss:
        log.info(
            "Regularization selection: baseline wins or ties (%.6f <= %.6f); using l1=0, l2=0",
            baseline_loss,
            best_metrics.validation_loss,
        )
        best_params = baseline_params
        best_cross_validation_result = baseline_cv
    else:
        log.info(
            "Regularization selection: Optuna trial %s wins (%.6f < %.6f); using l1=%.8g, l2=%.8g",
            best_trial.number,
            best_metrics.validation_loss,
            baseline_loss,
            best_params.l1_strength,
            best_params.l2_strength,
        )

    result = RegularizationTuningResult(
        best_params=best_params,
        best_cross_validation_result=best_cross_validation_result,
        trial_history=tuple(trial_history),
    )
    state[key] = {
        'l1_strength': best_params.l1_strength,
        'l2_strength': best_params.l2_strength,
        'best_params': to_json_safe(asdict(best_params)),
        'best_metrics': to_json_safe(best_metrics.to_dict()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(state_file, state)
    return result
