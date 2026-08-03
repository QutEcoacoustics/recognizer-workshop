from pathlib import Path

import numpy as np

from perch_hoplite.db import interface
from perch_hoplite.db.tests import test_utils as db_test_utils
from recognizer_workshop.app.services import training


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
            'cmap': 1.0,
            'eval_preds': np.zeros((1, 1), dtype=np.float32),
            'eval_labels': np.zeros((1, 1), dtype=np.float32),
        }

    monkeypatch.setattr(training.classifier, 'train_linear_classifier', fake_train_linear_classifier)

    result = training.train_classifier(db, ('alpha',), params)

    assert captured['weak_negatives_batch_size'] == 7
    assert captured['learning_rate'] == params.learning_rate
    assert captured['weak_neg_weight'] == params.weak_neg_weight
    assert captured['num_train_steps'] == params.num_steps
    assert result.training_info['resolved_weak_negatives_batch_size'] == 7


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
            'eval_preds': np.zeros((1, 1), dtype=np.float32),
            'eval_labels': np.zeros((1, 1), dtype=np.float32),
        }

    # Inject an arbitrary array to verify the data manager overwrite functions as intended
    fake_oversampled = np.array([999, 999], dtype=np.int64)
    monkeypatch.setattr(training, 'calculate_oversample_factors', lambda *a, **k: {'alpha': {'pos_factor': 1.0, 'neg_factor': 1.0}})
    monkeypatch.setattr(training, 'build_oversampled_train_ids', lambda *a, **k: fake_oversampled)
    monkeypatch.setattr(training.classifier, 'train_linear_classifier', fake_train_linear_classifier)
    
    training.train_classifier(db, ('alpha',), params, balance_dataset=True)

    # Verify the training pipeline fed the simulated oversampled array directly to the classifier
    np.testing.assert_array_equal(captured['final_train_ids'], fake_oversampled)
