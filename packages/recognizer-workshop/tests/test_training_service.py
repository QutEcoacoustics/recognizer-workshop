from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from perch_hoplite.db import interface
from perch_hoplite.db.tests import test_utils as db_test_utils
from recognizer_workshop.app.services import training
from recognizer_workshop.app.services import regularization_tuning


def _make_labeled_db(tmp_path: Path, *, positive_count: int, negative_count: int):
    rng = np.random.default_rng(1234)
    db = db_test_utils.make_db(
        path=str(tmp_path / 'training-db'),
        db_type='in_mem',
        num_embeddings=positive_count + negative_count,
        rng=rng,
        embedding_dim=8,
    )
    ids = list(db.get_embedding_ids())
    label = 'alpha'

    for embedding_id in ids[:positive_count]:
        db.insert_label(
            interface.Label(
                embedding_id=int(embedding_id),
                label=label,
                type=interface.LabelType.POSITIVE,
                provenance='test',
            )
        )

    for embedding_id in ids[positive_count:positive_count + negative_count]:
        db.insert_label(
            interface.Label(
                embedding_id=int(embedding_id),
                label=label,
                type=interface.LabelType.NEGATIVE,
                provenance='test',
            )
        )

    db.commit()
    return db


def test_calculate_balanced_weak_negatives_batch_size_matches_ratio_formula(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=10, negative_count=2)

    _resolved_batch, _resolved_weight, _message = training.calculate_balanced_weak_negatives_batch_size(
        db,
        target_labels=('alpha',),
        weak_neg_weight=0.2,
        batch_size=10,
    )

    # we need to add 8 negatives 'worth' @ 0.2 weight, so 40 weak negatives total, across all batches
    # for 100% balancing. Default balancing factor is 0.6, so adjust that to 0.6*40 = 24 weak negatives across all batches. 
    # with a batch size of 10, we are doing 12/10 batches, which means we need 10/12 * 24 = 20 weak negatives per 
    # batch to achieve the desired balancing.

    assert _resolved_batch == 20


def test_calculate_balanced_weak_negatives_batch_size_returns_half_batch_size_when_already_balanced(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=5, negative_count=5)

    _resolved_batch, _resolved_weight, _message = training.calculate_balanced_weak_negatives_batch_size(
        db,
        target_labels=('alpha',),
        weak_neg_weight=0.2,
        batch_size=10,
    )

    assert _resolved_batch == 5


def test_train_classifier_uses_explicit_weak_negative_batch_size(monkeypatch, tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=10, negative_count=2)
    params = training.TrainClassifierParams(
        batch_size=10,
        weak_neg_weight=0.2,
        weak_negatives_batch_size=7,
        train_ratio=1.0,
        num_steps=3,
    )
    captured = {}

    class DummyClassifier:
        pass

    def fake_train_linear_classifier(data_manager, learning_rate, weak_neg_weight, num_train_steps):
        captured['weak_negatives_batch_size'] = data_manager.weak_negatives_batch_size
        captured['learning_rate'] = learning_rate
        captured['weak_neg_weight'] = weak_neg_weight
        captured['num_train_steps'] = num_train_steps
        return DummyClassifier(), {
            'top1_acc': 1.0,
            'roc_auc': 1.0,
            'roc_auc_individual': np.array([1.0]),
            'cmap': 1.0,
            'cmap_individual': np.array([1.0]),
            'micro_precision': 1.0,
            'macro_precision': 1.0,
            'micro_recall': 1.0,
            'macro_recall': 1.0,
            'micro_f1': 1.0,
            'macro_f1': 1.0,
            'validation_loss': 0.5,
            'per_class_confusion': [],
            'eval_preds': np.zeros((1, 1), dtype=np.float32),
            'eval_labels': np.zeros((1, 1), dtype=np.float32),
        }

    monkeypatch.setattr(training.classifier, 'train_linear_classifier', fake_train_linear_classifier)

    result = training.train_classifier(db, ('alpha',), params)

    assert captured['weak_negatives_batch_size'] == 7
    assert captured['learning_rate'] == params.learning_rate
    assert captured['weak_neg_weight'] == params.weak_neg_weight
    assert captured['num_train_steps'] == params.num_steps
    assert result.info['final_model']['data_info']['balancing_debug']['oversampled_train_ids_used_count'] >= 0


# ── oversample factor tests ──────────────────────────────────────────────────

def test_calculate_oversample_factors_neg_deficit(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=1000, negative_count=100)
    params = training.TrainClassifierParams(batch_size=32, weak_neg_weight=0.1, weak_negatives_batch_size=200)
    train_ids = np.array(list(db.get_embedding_ids()))
    
    factors = training.calculate_oversample_factors(db, ('alpha',), params, train_ids)
    
    # expected_pos_signal = 1000 * (32 - 20) = 12000
    # expected_neg_signal = 100 * (32 + 20) = 5200
    # neg_factor = 12000 / 5200 = 2.307
    assert factors['alpha']['pos_factor'] == 1.0
    assert abs(factors['alpha']['neg_factor'] - 2.307) < 0.01


def test_calculate_oversample_factors_pos_deficit(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=250, negative_count=100)
    params = training.TrainClassifierParams(batch_size=32, weak_neg_weight=0.1, weak_negatives_batch_size=200)
    train_ids = np.array(list(db.get_embedding_ids()))
    
    factors = training.calculate_oversample_factors(db, ('alpha',), params, train_ids)
    
    # expected_pos_signal = 250 * (32 - 20) = 3000
    # expected_neg_signal = 100 * (32 + 20) = 5200
    # pos_factor = 5200 / 3000 = 1.733
    assert factors['alpha']['neg_factor'] == 1.0
    assert abs(factors['alpha']['pos_factor'] - 1.733) < 0.01


def test_calculate_oversample_factors_already_balanced(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=100, negative_count=100)
    params = training.TrainClassifierParams(batch_size=32, weak_neg_weight=0.1, weak_negatives_batch_size=0)
    train_ids = np.array(list(db.get_embedding_ids()))
    
    factors = training.calculate_oversample_factors(db, ('alpha',), params, train_ids)
    
    assert factors['alpha'] == {'pos_factor': 1.0, 'neg_factor': 1.0}


def test_calculate_oversample_factors_weak_negs_saturate(tmp_path):
    # When Weak Negatives consume the entire batch capacity (B <= W), the module skips balancing.
    db = _make_labeled_db(tmp_path, positive_count=100, negative_count=10)
    params = training.TrainClassifierParams(batch_size=10, weak_neg_weight=1.0, weak_negatives_batch_size=10)
    train_ids = np.array(list(db.get_embedding_ids()))
    
    factors = training.calculate_oversample_factors(db, ('alpha',), params, train_ids)
    
    assert factors['alpha'] == {'pos_factor': 1.0, 'neg_factor': 1.0}


def test_build_oversampled_train_ids_repeats_minority(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=10, negative_count=2)
    factors = {'alpha': {'pos_factor': 1.0, 'neg_factor': 3.0}}
    train_ids = np.array(list(db.get_embedding_ids()))
    
    oversampled_ids = training.build_oversampled_train_ids(db, ('alpha',), factors, train_ids)
    
    # 10 positives + (2 negatives * 3 repeats) = 16
    assert len(oversampled_ids) == 16


def test_build_oversampled_train_ids_no_oversampling(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=5, negative_count=5)
    factors = {'alpha': {'pos_factor': 1.0, 'neg_factor': 1.0}}
    train_ids = np.array(list(db.get_embedding_ids()))
    
    oversampled_ids = training.build_oversampled_train_ids(db, ('alpha',), factors, train_ids)
    
    assert len(oversampled_ids) == 10


def test_build_oversampled_train_ids_deduplicates_overlapping_labels(tmp_path):
    db = db_test_utils.make_db(
        path=str(tmp_path / 'multilabel-training-db'),
        db_type='in_mem',
        num_embeddings=3,
        rng=np.random.default_rng(1234),
        embedding_dim=8,
    )
    embedding_ids = list(db.get_embedding_ids())
    for label, embedding_id in (
        ('alpha', embedding_ids[0]),
        ('alpha', embedding_ids[1]),
        ('beta', embedding_ids[1]),
        ('beta', embedding_ids[2]),
    ):
        db.insert_label(
            interface.Label(
                embedding_id=int(embedding_id),
                label=label,
                type=interface.LabelType.POSITIVE,
                provenance='test',
            )
        )
    db.commit()

    factors = {
        'alpha': {'pos_factor': 1.0, 'neg_factor': 1.0},
        'beta': {'pos_factor': 1.0, 'neg_factor': 1.0},
    }
    oversampled_ids = training.build_oversampled_train_ids(
        db, ('alpha', 'beta'), factors, np.asarray(embedding_ids)
    )

    np.testing.assert_array_equal(oversampled_ids, np.asarray(embedding_ids))


def test_train_classifier_balance_flow_uses_isolated_splits(monkeypatch, tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=10, negative_count=2)
    params = training.TrainClassifierParams(
        batch_size=4, weak_neg_weight=0.1, weak_negatives_batch_size=0, seed=42
    )

    captured = {}

    class DummyClassifier:
        pass

    def fake_train_linear_classifier(data_manager, learning_rate, weak_neg_weight, num_train_steps):
        train_ids, eval_ids = data_manager.get_train_test_split()
        captured['final_train_ids'] = train_ids
        captured['final_eval_ids'] = eval_ids
        return DummyClassifier(), {
            'top1_acc': 1.0, 'roc_auc': 1.0, 'cmap': 1.0,
            'roc_auc_individual': np.array([1.0]),
            'cmap_individual': np.array([1.0]),
            'micro_precision': 1.0, 'macro_precision': 1.0,
            'micro_recall': 1.0, 'macro_recall': 1.0,
            'micro_f1': 1.0, 'macro_f1': 1.0,
            'validation_loss': 0.5, 'per_class_confusion': [],
            'eval_preds': np.zeros((1, 1), dtype=np.float32),
            'eval_labels': np.zeros((1, 1), dtype=np.float32),
        }

    # Inject an arbitrary array to verify the data manager overwrite functions as intended
    fake_oversampled = np.array([999, 999], dtype=np.int64)
    monkeypatch.setattr(training, 'calculate_oversample_factors', lambda *a, **k: {'alpha': {'pos_factor': 1.0, 'neg_factor': 1.0}})
    monkeypatch.setattr(training, 'build_oversampled_train_ids', lambda *a, **k: fake_oversampled)
    monkeypatch.setattr(training.classifier, 'train_linear_classifier', fake_train_linear_classifier)
    
    training.train_classifier(db, ('alpha',), params)

    # Verify the training pipeline fed the simulated oversampled array directly to the classifier
    np.testing.assert_array_equal(captured['final_train_ids'], fake_oversampled)


def test_cross_validation_trains_final_model_on_all_labeled_ids(monkeypatch, tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=4, negative_count=3)
    params = training.TrainClassifierParams(
        batch_size=3,
        train_ratio=0.75,
        num_steps=1,
        seed=42,
        k_folds=2,
    )
    captured_splits = []

    class DummyClassifier:
        pass

    def fake_train_linear_classifier(data_manager, learning_rate, weak_neg_weight, num_train_steps):
        train_ids, eval_ids = data_manager.get_train_test_split()
        captured_splits.append((np.asarray(train_ids), np.asarray(eval_ids)))
        return DummyClassifier(), {
            'top1_acc': 1.0,
            'roc_auc': 1.0,
                'roc_auc_individual': np.array([1.0]),
            'cmap': 1.0,
                'cmap_individual': np.array([1.0]),
            'micro_precision': 1.0,
            'macro_precision': 1.0,
            'micro_recall': 1.0,
            'macro_recall': 1.0,
            'micro_f1': 1.0,
            'macro_f1': 1.0,
            'validation_loss': 0.5,
                'per_class_confusion': [],
            'eval_preds': np.zeros((0, 1), dtype=np.float32),
            'eval_labels': np.zeros((0, 1), dtype=np.float32),
        }

    monkeypatch.setattr(training.classifier, 'train_linear_classifier', fake_train_linear_classifier)
    monkeypatch.setattr(training, 'calculate_confusion_and_prf_metrics', lambda *args, **kwargs: {})

    params = replace(params, balance_dataset=False)
    result = training.train_classifier(db, ('alpha',), params)

    assert len(captured_splits) == 3
    assert all(len(eval_ids) > 0 for _, eval_ids in captured_splits[:2])
    np.testing.assert_array_equal(captured_splits[2][0], np.asarray(list(db.get_embedding_ids())))
    assert len(captured_splits[2][1]) == 0
    assert result.info['evaluation']['average_metrics']['validation_loss'] == 0.5
    assert result.info['evaluation']['average_metrics']['train_ratio'] == params.train_ratio
    assert result.info['final_model']['params']['train_ratio'] == 1.0
    assert result.info['final_model']['params']['k_folds'] == 1


def test_train_ratio_one_prepares_all_labeled_ids_for_training(tmp_path):
    db = _make_labeled_db(tmp_path, positive_count=4, negative_count=3)
    params = training.TrainClassifierParams(train_ratio=1.0)

    params = training.TrainClassifierParams(train_ratio=1.0, balance_dataset=False)
    split = training.get_oversampled_data_split(db, ('alpha',), params)

    np.testing.assert_array_equal(split.train_ids, np.asarray(list(db.get_embedding_ids())))
    assert len(split.eval_ids) == 0
    assert split.data_info['data_counts']['train']['alpha']['pos'] == {
        'unique': 4,
        'oversampled': 4,
    }
    assert split.data_info['data_counts']['train']['alpha']['neg'] == {
        'unique': 3,
        'oversampled': 3,
    }
    assert split.data_info['data_counts']['test']['alpha'] == {'pos': 0, 'neg': 0}


def test_tune_regularization_persists_state_and_reduces_follow_up_trials(monkeypatch, tmp_path):
    class FakeDB:
        def get_embeddings_by_label(self, label, label_type, _unused=None):
            return [1, 2] if label_type == interface.LabelType.POSITIVE else [3]

    calls = []
    seen_splits = []

    def fake_cross_validate_classifier(db, target_labels, params, fold_splits):
        calls.append((params.l1_strength, params.l2_strength))
        seen_splits.append(fold_splits)
        loss = 1.0 if params.l1_strength == 0.0 and params.l2_strength == 0.0 else 0.1
        metrics = training.EvalMetrics(
            train_ratio=0.9,
            top1_acc=0.5,
            roc_auc=0.5,
            roc_auc_individual=np.array([0.5]),
            cmap=0.5,
            cmap_individual=np.array([0.5]),
            micro_precision=0.5,
            macro_precision=0.5,
            micro_recall=0.5,
            macro_recall=0.5,
            micro_f1=0.5,
            macro_f1=0.5,
            validation_loss=loss,
            per_class_confusion=[],
        )
        data_info = {
            'data_counts': {'train': {}, 'test': {}},
            'balancing_debug': {'balance_dataset_enabled': False},
        }
        return training.CrossValidationResult(
            average_metrics=metrics,
            average_data_counts=data_info['data_counts'],
            folds=({
                'metrics': metrics.to_dict(),
                'data_info': data_info,
            },),
        )

    monkeypatch.setattr(regularization_tuning.training, 'cross_validate_classifier', fake_cross_validate_classifier)
    params = training.TrainClassifierParams(tuning_trials=2, tuning_folds=2, seed=7)
    state_path = tmp_path / 'regularization.json'
    fold_splits = (
            training.PreparedDataSplit(
            train_ids=np.array([1]),
            eval_ids=np.array([2]),
            data_info={
                'data_counts': {'train': {}, 'test': {}},
                'balancing_debug': {'balance_dataset_enabled': False},
            },
        ),
            training.PreparedDataSplit(
            train_ids=np.array([2]),
            eval_ids=np.array([1]),
            data_info={
                'data_counts': {'train': {}, 'test': {}},
                'balancing_debug': {'balance_dataset_enabled': False},
            },
        ),
    )

    first = regularization_tuning.tune_regularization(
        FakeDB(), ('alpha',), params,
        fold_splits=fold_splits,
        classifier_name='demo', state_path=state_path,
    )
    first_call_count = len(calls)
    second = regularization_tuning.tune_regularization(
        FakeDB(), ('alpha',), params,
        fold_splits=fold_splits,
        classifier_name='demo', state_path=state_path,
    )

    assert first.best_params.l1_strength > 0
    assert first.best_params.l2_strength > 0
    assert first.best_cross_validation_result.average_metrics.validation_loss == 0.1
    assert len(first.trial_history) == 3
    assert state_path.exists()
    assert second.best_cross_validation_result.average_metrics.validation_loss == 0.1
    assert len(second.trial_history) == 2
    assert len(calls) - first_call_count == 2
    assert all(candidate is fold_splits for candidate in seen_splits)

    changed_data_splits = (
        training.PreparedDataSplit(
            train_ids=np.array([10]),
            eval_ids=np.array([20]),
            data_info={
                'data_counts': {'train': {}, 'test': {}},
                'balancing_debug': {'balance_dataset_enabled': False},
            },
        ),
        training.PreparedDataSplit(
            train_ids=np.array([20]),
            eval_ids=np.array([10]),
            data_info={
                'data_counts': {'train': {}, 'test': {}},
                'balancing_debug': {'balance_dataset_enabled': False},
            },
        ),
    )
    third = regularization_tuning.tune_regularization(
        FakeDB(), ('alpha',), params,
        fold_splits=changed_data_splits,
        classifier_name='demo', state_path=state_path,
    )
    assert third.best_cross_validation_result.average_metrics.validation_loss == 0.1
    assert len(third.trial_history) == 2


def test_delete_tuning_state_removes_only_matching_entry(tmp_path):
    params = training.TrainClassifierParams()
    params = training.TrainClassifierParams(balance_dataset=False)
    reporting_params = training.TrainClassifierParams(
        f1_threshold=0.5,
        balance_dataset=False,
    )
    matching_key = regularization_tuning.state_key('demo', ('alpha',), params)
    assert matching_key == regularization_tuning.state_key('demo', ('alpha',), reporting_params)
    other_key = regularization_tuning.state_key('other', ('alpha',), params)
    state_path = tmp_path / 'regularization.json'
    state_path.write_text(json.dumps({matching_key: {'value': 1}, other_key: {'value': 2}}))

    removed = regularization_tuning.delete_tuning_state(state_path, 'demo', ('alpha',), params)

    assert removed is True
    assert json.loads(state_path.read_text()) == {other_key: {'value': 2}}


def test_eval_metrics_round_trip_and_string_format():
    values = {
        'train_ratio': 0.9,
        'top1_acc': 0.91,
        'roc_auc': 0.82,
        'roc_auc_individual': np.array([0.8]),
        'cmap': 0.73,
        'cmap_individual': np.array([0.7]),
        'micro_precision': 0.64,
        'macro_precision': 0.63,
        'micro_recall': 0.62,
        'macro_recall': 0.61,
        'micro_f1': 0.60,
        'macro_f1': 0.59,
        'validation_loss': 0.5,
        'per_class_confusion': [],
    }

    metrics = training.EvalMetrics.from_dict(values)

    round_trip = metrics.to_dict()
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(round_trip[key], value)
        else:
            assert round_trip[key] == value
    assert 'top-1    0.910' in str(metrics)
    assert 'val_loss 0.500' in str(metrics)
