from dataclasses import asdict, dataclass, field, replace
import math
import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from recognizer_workshop.perch_hoplite_custom_modifications import classifier
from perch_hoplite.agile import classifier_data
from perch_hoplite.db import interface as hoplite_interface
from recognizer_workshop.app.services.json_utils import to_json_safe

if TYPE_CHECKING:
    from recognizer_workshop.app.services.regularization_tuning import RegularizationTuningResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainClassifierParams:
    learning_rate: float = 1e-3
    weak_neg_weight: float = 0.05
    num_steps: int = 128
    train_ratio: float = 0.9
    batch_size: int = 128
    weak_negatives_batch_size: int = 128
    seed: int = 5
    f1_threshold: float = 0.0
    l1_strength: float = 0.0
    l2_strength: float = 0.0
    regularize_bias: bool = False
    k_folds: int = 4
    auto_tune_regularization: bool = False
    tuning_trials: int = 10
    tuning_folds: int = 4
    balance_dataset: bool = True


@dataclass(frozen=True)
class EvalMetrics:
    train_ratio: float
    top1_acc: float
    roc_auc: float
    roc_auc_individual: Any
    cmap: float
    cmap_individual: Any
    micro_precision: float
    macro_precision: float
    micro_recall: float
    macro_recall: float
    micro_f1: float
    macro_f1: float
    validation_loss: float
    per_class_confusion: Any

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> 'EvalMetrics':
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def average(cls, metrics: Sequence['EvalMetrics']) -> 'EvalMetrics':
        if not metrics:
            raise ValueError('Cannot average an empty metrics sequence.')

        def average_values(values):
            array_values = np.asarray(values, dtype=float)
            finite = np.isfinite(array_values)
            if not finite.any():
                return float('nan')
            return float(np.nanmean(array_values))

        def average_arrays(values):
            stacked = np.stack(values)
            with np.errstate(invalid='ignore'):
                return np.nanmean(stacked, axis=0)

        def average_confusions(values):
            if not all(value is not None for value in values):
                return None
            if not all(len(value) == len(values[0]) for value in values):
                return None
            averaged = []
            for class_index in range(len(values[0])):
                first = values[0][class_index]
                result = {
                    'class_index': first['class_index'],
                    'class_label': first['class_label'],
                }
                for key, value in first.items():
                    if key in result:
                        continue
                    result[key] = average_arrays(
                        [np.asarray(metrics[class_index][key], dtype=float) for metrics in values]
                    ).tolist()
                    if np.asarray(result[key]).ndim == 0:
                        result[key] = float(result[key])
                averaged.append(result)
            return averaged

        first = metrics[0]
        return cls(
            train_ratio=average_values([metric.train_ratio for metric in metrics]),
            top1_acc=average_values([metric.top1_acc for metric in metrics]),
            roc_auc=average_values([metric.roc_auc for metric in metrics]),
            roc_auc_individual=average_arrays([metric.roc_auc_individual for metric in metrics]),
            cmap=average_values([metric.cmap for metric in metrics]),
            cmap_individual=average_arrays([metric.cmap_individual for metric in metrics]),
            micro_precision=average_values([metric.micro_precision for metric in metrics]),
            macro_precision=average_values([metric.macro_precision for metric in metrics]),
            micro_recall=average_values([metric.micro_recall for metric in metrics]),
            macro_recall=average_values([metric.macro_recall for metric in metrics]),
            micro_f1=average_values([metric.micro_f1 for metric in metrics]),
            macro_f1=average_values([metric.macro_f1 for metric in metrics]),
            validation_loss=average_values([metric.validation_loss for metric in metrics]),
            per_class_confusion=average_confusions(
                [metric.per_class_confusion for metric in metrics]
            ) if first.per_class_confusion is not None else None,
        )

    def __str__(self) -> str:
        labels = (
            ('top1_acc', 'top-1'),
            ('roc_auc', 'roc_auc'),
            ('cmap', 'cmap'),
            ('micro_precision', 'micro_p'),
            ('macro_precision', 'macro_p'),
            ('micro_recall', 'micro_r'),
            ('macro_recall', 'macro_r'),
            ('micro_f1', 'micro_f1'),
            ('macro_f1', 'macro_f1'),
        )
        lines = [f'{label:<8} {getattr(self, key):.3f}' for key, label in labels]
        lines.append(f'{"val_loss":<8} {self.validation_loss:.3f}')
        lines.append(f'{"train_ratio":<8} {self.train_ratio:.3f}')
        return '\n'.join(lines)


@dataclass(frozen=True)
class CrossValidationResult:
    average_metrics: EvalMetrics
    average_data_counts: dict[str, Any] = field(default_factory=dict)
    folds: tuple[dict[str, Any], ...] = ()
    params: TrainClassifierParams | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'average_metrics': self.average_metrics.to_dict(),
            'average_data_counts': self.average_data_counts,
            'folds': list(self.folds),
            'params': asdict(self.params) if self.params is not None else None,
        }


def _average_data_counts(data_counts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric leaves in the nested train/test count structure."""
    if not data_counts:
        return {}

    keys = set().union(*(counts.keys() for counts in data_counts))
    averaged = {}
    for key in keys:
        values = [counts[key] for counts in data_counts if key in counts]
        if all(isinstance(value, dict) for value in values):
            averaged[key] = _average_data_counts(values)
        else:
            averaged[key] = float(np.mean([float(value) for value in values]))
    return averaged


@dataclass(frozen=True)
class TrainingResult:
    classifier_model: Any
    info: dict[str, Any]


@dataclass(frozen=True)
class PreparedDataSplit:
    train_ids: np.ndarray
    eval_ids: np.ndarray
    data_info: dict[str, Any]

    @property
    def training_info(self) -> dict[str, Any]:
        """Compatibility view of the split's balancing debug information."""
        return self.data_info.get('balancing_debug', {})


def calculate_balanced_weak_negatives_batch_size(
    db,
    target_labels: Sequence[str],
    weak_neg_weight: float = 0.05,
    batch_size: int = 32,
    max_weak_batch_size: int = 512,
    balancing_factor: float = 0.6,
    min_weak_neg_weight: float = 0.05,
) -> tuple[int, float, str]:
    """
    Estimate weak-negative batch size and adjusted weight using global DB counts.
    arg balancing_factor: float in [0,1]. How much of the negative deficit to fill with weak negatives. 

    For each of the target labels, this calculates the deficit of negative examples compared to positives, and the ratio of that deficit to the total labeled examples for that label. It averages those ratios across labels to get an average per-label deficit ratio, and also calculates a global deficit ratio based on total positive and negative counts. It then blends those two ratios to get a final blended deficit ratio. The number of weak negatives to add per batch is then calculated as the blended deficit ratio multiplied by the batch size, divided by the weak negative weight. The function also applies a maximum cap to the weak negative batch size and ensures that the weak negative weight does not drop below a baseline.
    """
    if batch_size <= 0:
        return 0, 0.0, "Invalid batch_size parameter"
    
    max_weak_batch_size = max(int(max_weak_batch_size), 1)
    
    balancing_factor = max(0.0, min(float(balancing_factor), 1.0))
    if balancing_factor <= 0.0:
        return 0, 0.0, "Weak-negative balancing disabled (balancing_factor=0)."

    target_labels_set = set(target_labels)
    if not target_labels_set:
        return 0, 0.0, "No target labels provided"

    pos_counts = db.get_class_counts(hoplite_interface.LabelType.POSITIVE)
    neg_counts = db.get_class_counts(hoplite_interface.LabelType.NEGATIVE)

    total_pos_count = 0
    total_neg_count = 0
    per_label_deficit_ratios = []

    # Calculate global sums and per-class ratios for the target labels
    # for each class, we calculate how many negative labelled items are missing compared to positives. 
    # then we calculate the ratio of that deficit to the total number of items for the class. 
    # for each batch of size N, if we added n*that ratio negative items, it would balance that class. 
    # e.g p=75, n=25, deficit=50, ratio=0.5, batch_size=32 (24 pos +8 neg) ->  0.5*32 = 16 = 24 pos and 24 neg 
    for label in target_labels_set:
        pos_count = int(pos_counts.get(label, 0))
        neg_count = int(neg_counts.get(label, 0))
        total_pos_count += pos_count
        total_neg_count += neg_count

        label_total_count = pos_count + neg_count
        if label_total_count > 0:
            neg_deficit_for_label = max(pos_count - neg_count, 0)
            per_label_deficit_ratios.append(neg_deficit_for_label / label_total_count)

    # Calculate the Average Per-Class Ratio (Gives disproportionate weight to smaller classes)
    average_per_label_deficit_ratio = (
        sum(per_label_deficit_ratios) / len(per_label_deficit_ratios)
        if per_label_deficit_ratios
        else 0.0
    )
    

    # this is the global deficit ratio all classes pooled. It gives weight proportionate to class size. 
    global_labeled_total_count = total_pos_count + total_neg_count
    global_neg_deficit = (
        max(total_pos_count - total_neg_count, 0)
        if global_labeled_total_count > 0
        else 0
    )
    global_deficit_ratio = (
        global_neg_deficit / global_labeled_total_count
        if global_labeled_total_count > 0
        else 0.0
    )

    # Both have merits, so we take the average. 
    blended_deficit_ratio = (global_deficit_ratio + average_per_label_deficit_ratio) / 2.0

    # Scale by balancing_factor so users can dial weak-negative balancing up/down.    
    adjusted_deficit_ratio = blended_deficit_ratio * balancing_factor

    # Number of fully-weighted negatives missing from an average batch.
    missing_negative_signal_per_batch = adjusted_deficit_ratio * batch_size

    # We never want the weight to drop below a baseline (0.05), even if the user provided something tiny.
    baseline_weak_negative_weight = max(weak_neg_weight, min_weak_neg_weight)
    # We always want at least some weak negatives, even if the dataset is perfectly balanced.
    minimum_weak_negative_batch_size = batch_size / 2.0

    # Calculate required weak batch size before applying size limits
    uncapped_weak_negative_batch_size = missing_negative_signal_per_batch / baseline_weak_negative_weight

    # constraining the batch size to within a max and minimum
    final_weak_negative_batch_size = int(math.ceil(
        min(max(uncapped_weak_negative_batch_size, minimum_weak_negative_batch_size), max_weak_batch_size)
    ))
    
    # recalculate the weak_neg_weight so that we get the desired effective weak negative impact
    final_weak_negative_weight = max(
        missing_negative_signal_per_batch / final_weak_negative_batch_size,
        baseline_weak_negative_weight,
    )

    # Format message based on class count and balancing factor
    if len(target_labels_set) > 1:
        if balancing_factor < 1.0:
            message = (
                f"Deficit ratios (pool/avg/blend/adj): "
                f"{global_deficit_ratio:.2f}/"
                f"{average_per_label_deficit_ratio:.2f}/"
                f"{blended_deficit_ratio:.2f}/"
                f"{adjusted_deficit_ratio:.2f}."
            )
        else:
            message = (
                f"Deficit ratios (pool/avg/blend): "
                f"{global_deficit_ratio:.2f}/"
                f"{average_per_label_deficit_ratio:.2f}/"
                f"{blended_deficit_ratio:.2f}."
            )
    else:
        if balancing_factor < 1.0:
            message = f"Deficit ratio: {blended_deficit_ratio:.2f} (adjusted: {adjusted_deficit_ratio:.2f})."
        else:
            message = f"Deficit ratio: {blended_deficit_ratio:.2f}."

    return final_weak_negative_batch_size, final_weak_negative_weight, message


def calculate_oversample_factors(
    db,
    target_labels: Sequence[str],
    params: TrainClassifierParams,
    train_ids: np.ndarray,
) -> dict[str, dict[str, float]]:
    """For each label, compute how much to oversample the minority labeled side
    using strictly the training split and the algebraic pool dilution formula.
    """
    weak_neg_weight = params.weak_neg_weight
    B = float(params.batch_size)
    W = float(params.weak_negatives_batch_size) * weak_neg_weight

    train_set = set(train_ids.tolist())
    oversample_factors_by_label = {}

    for label in target_labels:
        pos_embedding_ids = db.get_embeddings_by_label(
            label, hoplite_interface.LabelType.POSITIVE, None
        )
        neg_embedding_ids = db.get_embeddings_by_label(
            label, hoplite_interface.LabelType.NEGATIVE, None
        )

        # Count occurrences strictly within the training split
        P = sum(1 for x in pos_embedding_ids if x in train_set)
        N = sum(1 for x in neg_embedding_ids if x in train_set)

        # If batch is entirely consumed by weak negatives, we cannot balance.
        if P + N <= 0 or B <= W:
            oversample_factors_by_label[label] = {'pos_factor': 1.0, 'neg_factor': 1.0}
            continue

        # Expected signal calculations accounting for denominator expansion
        pos_signal = P * (B - W)
        neg_signal = N * (B + W)

        if pos_signal > neg_signal and N > 0:
            neg_factor = pos_signal / neg_signal
            oversample_factors_by_label[label] = {
                'pos_factor': 1.0,
                'neg_factor': max(neg_factor, 1.0),
            }
        elif neg_signal > pos_signal and P > 0:
            pos_factor = neg_signal / pos_signal
            oversample_factors_by_label[label] = {
                'pos_factor': max(pos_factor, 1.0),
                'neg_factor': 1.0,
            }
        else:
            oversample_factors_by_label[label] = {'pos_factor': 1.0, 'neg_factor': 1.0}

    return oversample_factors_by_label


def build_oversampled_train_ids(
    db,
    target_labels: Sequence[str],
    oversample_factors: dict[str, dict[str, float]],
    train_ids: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Build a training ID array with repetition applied per embedding ID."""
    rng = np.random.default_rng(seed)
    unique_train_ids = np.unique(np.asarray(train_ids, dtype=np.int64))
    train_set = set(unique_train_ids.tolist())
    repeat_factors_by_id = {int(embedding_id): 1.0 for embedding_id in unique_train_ids}

    def _repeat_ids_with_fraction(ids: np.ndarray, repeat_factor: float) -> np.ndarray:
        if len(ids) == 0:
            return np.array([], dtype=np.int64)

        factor = max(float(repeat_factor), 1.0)
        full_repeats = int(math.floor(factor))
        fractional_part = factor - full_repeats

        chunks: list[np.ndarray] = []
        if full_repeats > 0:
            chunks.append(np.tile(ids, full_repeats))

        if fractional_part > 0:
            fractional_target = fractional_part * len(ids)
            extra_count = int(math.floor(fractional_target))
            if rng.random() < (fractional_target - extra_count):
                extra_count += 1
            if extra_count > 0:
                chunks.append(rng.permutation(ids)[:extra_count])

        if not chunks:
            return np.array([], dtype=np.int64)
        return np.concatenate(chunks)

    for label in target_labels:
        label_factors = oversample_factors.get(label, {'pos_factor': 1.0, 'neg_factor': 1.0})

        for label_type, factor_key in (
            (hoplite_interface.LabelType.POSITIVE, 'pos_factor'),
            (hoplite_interface.LabelType.NEGATIVE, 'neg_factor'),
        ):
            repeat_factor = max(float(label_factors[factor_key]), 1.0)
            labeled_ids = db.get_embeddings_by_label(label, label_type, None)
            for embedding_id in labeled_ids:
                embedding_id = int(embedding_id)
                if embedding_id in train_set:
                    repeat_factors_by_id[embedding_id] = max(
                        repeat_factors_by_id[embedding_id], repeat_factor
                    )

    ids_by_repeat_factor: dict[float, list[int]] = {}
    for embedding_id in unique_train_ids:
        repeat_factor = repeat_factors_by_id[int(embedding_id)]
        ids_by_repeat_factor.setdefault(repeat_factor, []).append(int(embedding_id))

    oversampled_embedding_id_chunks = [
        _repeat_ids_with_fraction(np.asarray(ids, dtype=np.int64), repeat_factor)
        for repeat_factor, ids in ids_by_repeat_factor.items()
    ]
    if not oversampled_embedding_id_chunks:
        return np.array([], dtype=np.int64)

    return np.concatenate(oversampled_embedding_id_chunks)


def _count_labeled_ids(
    db,
    target_labels: Sequence[str],
    train_ids: np.ndarray,
    unique_train_ids: np.ndarray,
    eval_ids: np.ndarray,
) -> dict[str, Any]:
    """Count labeled train occurrences and unique train/eval examples."""
    train_id_set = set(np.asarray(unique_train_ids).tolist())
    eval_id_set = set(np.asarray(eval_ids).tolist())
    counts = {'train': {}, 'test': {}}

    for label in target_labels:
        label_counts = {'train': {}, 'test': {}}
        for key, label_type in (
            ('pos', hoplite_interface.LabelType.POSITIVE),
            ('neg', hoplite_interface.LabelType.NEGATIVE),
        ):
            labeled_ids = set(
                db.get_embeddings_by_label(label, label_type, None).tolist()
            )
            label_counts['train'][key] = {
                'oversampled': int(sum(embedding_id in labeled_ids for embedding_id in train_ids)),
                'unique': int(sum(embedding_id in labeled_ids for embedding_id in train_id_set)),
            }
            label_counts['test'][key] = int(
                sum(embedding_id in labeled_ids for embedding_id in eval_id_set)
            )
        counts['train'][label] = label_counts['train']
        counts['test'][label] = label_counts['test']

    return counts


def get_oversampled_data_split(
    db,
    target_labels: Sequence[str],
    params: TrainClassifierParams,
) -> PreparedDataSplit:
    """Prepare one train/eval split, including optional oversampling."""
    data_manager = classifier_data.AgileDataManager(
        target_labels=tuple(target_labels),
        db=db,
        train_ratio=params.train_ratio,
        min_eval_examples=1,
        batch_size=params.batch_size,
        weak_negatives_batch_size=params.weak_negatives_batch_size,
        rng=np.random.default_rng(seed=params.seed),
    )
    train_ids, eval_ids = data_manager.get_train_test_split()

    if params.train_ratio >= 1.0:
        original_train_ids = np.unique(np.concatenate([train_ids, eval_ids]))
        original_eval_ids = np.array([], dtype=np.int64)
    else:
        original_train_ids = np.asarray(train_ids).copy()
        original_eval_ids = np.asarray(eval_ids).copy()

    oversampling_applied = False
    oversampled_used_count = 0
    auto_oversample_factors = None
    prepared_train_ids = original_train_ids

    if params.balance_dataset:
        auto_oversample_factors = calculate_oversample_factors(
            db=db,
            target_labels=target_labels,
            params=params,
            train_ids=original_train_ids,
        )
        oversampled_train_ids = build_oversampled_train_ids(
            db=db,
            target_labels=target_labels,
            oversample_factors=auto_oversample_factors,
            train_ids=original_train_ids,
            seed=params.seed,
        )
        if len(oversampled_train_ids) > 0:
            prepared_train_ids = oversampled_train_ids
            oversampling_applied = True
            oversampled_used_count = int(len(oversampled_train_ids))

    data_counts = _count_labeled_ids(
        db=db,
        target_labels=target_labels,
        train_ids=prepared_train_ids,
        unique_train_ids=original_train_ids,
        eval_ids=original_eval_ids,
    )
    balancing_debug = {
        'balance_dataset_enabled': bool(params.balance_dataset),
        'oversampling_applied': oversampling_applied,
        'oversampled_train_ids_requested_count': oversampled_used_count,
        'oversampled_train_ids_used_count': oversampled_used_count,
        'excluded_eval_from_oversampled_count': 0,
        'original_train_ids_count': int(len(original_train_ids)),
        'eval_ids_count': int(len(original_eval_ids)),
    }
    if auto_oversample_factors is not None:
        balancing_debug['oversample_factors'] = auto_oversample_factors
        train_set = set(original_train_ids.tolist())
        balancing_debug['per_label_oversample_breakdown'] = {
            label: {
                'train_pos_count': sum(
                    1 for embedding_id in db.get_embeddings_by_label(
                        label, hoplite_interface.LabelType.POSITIVE, None
                    ) if embedding_id in train_set
                ),
                'train_neg_count': sum(
                    1 for embedding_id in db.get_embeddings_by_label(
                        label, hoplite_interface.LabelType.NEGATIVE, None
                    ) if embedding_id in train_set
                ),
                'pos_factor': auto_oversample_factors[label]['pos_factor'],
                'neg_factor': auto_oversample_factors[label]['neg_factor'],
            }
            for label in target_labels
        }

    return PreparedDataSplit(
        train_ids=prepared_train_ids,
        eval_ids=original_eval_ids,
        data_info={
            'data_counts': data_counts,
            'balancing_debug': balancing_debug,
        },
    )


def _train_single_model(
    db,
    target_labels: Sequence[str],
    params: TrainClassifierParams,
    split: PreparedDataSplit,
    evaluate: bool,
) -> tuple[Any, EvalMetrics | None]:
    """Train once using an already prepared split."""
    if evaluate and len(split.eval_ids) == 0:
        raise ValueError('Cannot evaluate a training split with no evaluation IDs.')

    data_manager = classifier_data.AgileDataManager(
        target_labels=tuple(target_labels),
        db=db,
        train_ratio=1.0,
        min_eval_examples=1,
        batch_size=params.batch_size,
        weak_negatives_batch_size=params.weak_negatives_batch_size,
        rng=np.random.default_rng(seed=params.seed),
    )

    def _provided_split():
        return split.train_ids, split.eval_ids

    data_manager.get_train_test_split = _provided_split

    train_kwargs = {
        'data_manager': data_manager,
        'learning_rate': params.learning_rate,
        'weak_neg_weight': params.weak_neg_weight,
        'num_train_steps': params.num_steps,
    }
    if params.l1_strength or params.l2_strength or params.regularize_bias:
        train_kwargs.update({
            'l1_strength': params.l1_strength,
            'l2_strength': params.l2_strength,
            'regularize_bias': params.regularize_bias,
        })

    linear_classifier, eval_scores = classifier.train_linear_classifier(
        **train_kwargs,
    )

    output_scores = dict(eval_scores)
    if evaluate:
        confusion_metrics = calculate_confusion_and_prf_metrics(
            eval_scores.get('eval_preds'),
            eval_scores.get('eval_labels'),
            class_names=target_labels,
            threshold=params.f1_threshold,
        )
        output_scores.update(confusion_metrics)
        for key in ('eval_ids', 'eval_preds', 'eval_labels'):
            output_scores.pop(key, None)
        output_scores['train_ratio'] = params.train_ratio

    log.info('%s', '\n' + '-' * 80)
    if evaluate:
        metrics = EvalMetrics.from_dict(output_scores)
        log.info('%s', metrics)
    else:
        metrics = None
    log.info(
        'training params: l1=%.8g l2=%.8g train_ratio=%.3f weak_neg_batch=%s',
        params.l1_strength,
        params.l2_strength,
        params.train_ratio,
        params.weak_negatives_batch_size,
    )

    return linear_classifier, metrics


def train_classifier(
    db,
    target_labels: Sequence[str],
    params: TrainClassifierParams,
    classifier_name: str | None = None,
    regularization_state_path=None,
) -> TrainingResult:
    """Coordinate tuning, validation, and final one-pass model training."""
    if params.auto_tune_regularization:
        fold_count = params.tuning_folds
    else:
        fold_count = max(params.k_folds, 1)

    fold_splits = tuple(
        get_oversampled_data_split(
            db=db,
            target_labels=target_labels,
            params=replace(params, seed=params.seed + fold_index),
        )
        for fold_index in range(fold_count)
    )

    if params.auto_tune_regularization:
        from recognizer_workshop.app.services.regularization_tuning import tune_regularization

        tuning: RegularizationTuningResult = tune_regularization(
            db=db,
            target_labels=target_labels,
            params=params,
            fold_splits=fold_splits,
            classifier_name=classifier_name,
            state_path=regularization_state_path,
        )
        selected_params = tuning.best_params
        best_cross_validation = tuning.best_cross_validation_result
        evaluation_info = best_cross_validation.to_dict()
        evaluation_info['regularization_tuning'] = tuning.to_dict()
    else:
        cross_validation = cross_validate_classifier(
            db=db,
            target_labels=target_labels,
            params=params,
            fold_splits=fold_splits,
        )
        selected_params = params
        evaluation_info = cross_validation.to_dict()

    final_params = replace(
        selected_params,
        auto_tune_regularization=False,
        k_folds=1,
        train_ratio=1.0,
    )
    final_split = get_oversampled_data_split(
        db=db,
        target_labels=target_labels,
        params=final_params,
    )
    final_classifier, _ = _train_single_model(
        db=db,
        target_labels=target_labels,
        params=final_params,
        split=final_split,
        evaluate=False,
    )
    info = {
        'evaluation': evaluation_info,
        'final_model': {
            'data_info': final_split.data_info,
            'params': asdict(final_params),
        },
    }
    return TrainingResult(
        classifier_model=final_classifier,
        info=info,
    )


def get_all_labeled_ids(db, target_labels: Sequence[str]) -> np.ndarray:
    labeled_ids = []
    for label in target_labels:
        labeled_ids.extend(db.get_embeddings_by_label(label, hoplite_interface.LabelType.POSITIVE, None))
        labeled_ids.extend(db.get_embeddings_by_label(label, hoplite_interface.LabelType.NEGATIVE, None))
    if not labeled_ids:
        return np.array([], dtype=np.int64)
    return np.unique(np.asarray(labeled_ids, dtype=np.int64))


def cross_validate_classifier(
    db,
    target_labels: Sequence[str],
    params: TrainClassifierParams,
    fold_splits: Sequence[PreparedDataSplit],
) -> CrossValidationResult:
    """Evaluate one parameter set against one fixed collection of splits."""
    if not fold_splits:
        raise ValueError('At least one fold split is required.')

    fold_records = []
    for split in fold_splits:
        _, metrics = _train_single_model(
            db=db,
            target_labels=target_labels,
            params=replace(params, k_folds=1, auto_tune_regularization=False),
            split=split,
            evaluate=True,
        )
        fold_records.append({
            'metrics': metrics.to_dict() if metrics is not None else {},
            'data_info': split.data_info,
        })

    return CrossValidationResult(
        average_metrics=EvalMetrics.average([
            EvalMetrics.from_dict(fold['metrics']) for fold in fold_records
        ]),
        average_data_counts=_average_data_counts([
            fold['data_info']['data_counts'] for fold in fold_records
        ]),
        folds=tuple(fold_records),
        params=params,
    )


def training_result_to_json(result: TrainingResult) -> dict[str, Any]:
    """Create a JSON-safe persisted result from TrainingResult."""
    info = result.info
    if 'evaluation' not in info:
        legacy_metrics = info.get('metrics')
        if isinstance(legacy_metrics, dict):
            legacy_metrics = dict(legacy_metrics)
            legacy_metrics.pop('eval_preds', None)
            legacy_metrics.pop('eval_labels', None)
            info = {**info, 'metrics': legacy_metrics}
    return to_json_safe(info)


def _require_metric(scores: dict[str, Any], key: str) -> float:
    """Return a required metric as float, raising if missing."""
    if key not in scores:
        raise KeyError(f"Missing required training metric: {key}")
    return float(scores[key])


def calculate_confusion_and_prf_metrics(
    logits,
    labels,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Calculate per-class 2x2 confusion data and aggregated PR/F1 metrics."""

    def _safe_div(num: float, denom: float) -> float:
        return float(num / denom) if denom > 0 else float('nan')

    missing_metrics = {
            'per_class_confusion': [],
            'micro_precision': float('nan'),
            'macro_precision': float('nan'),
            'micro_recall': float('nan'),
            'macro_recall': float('nan'),
            'micro_f1': float('nan'),
            'macro_f1': float('nan'),
        }

    if logits is None or labels is None:
        return missing_metrics

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.size == 0 or labels.size == 0:
        return missing_metrics

    if logits.ndim != 2 or labels.ndim != 2:
        raise ValueError('Expected eval_preds and eval_labels to be 2D arrays [n_examples, n_classes].')

    if logits.shape != labels.shape:
        raise ValueError(
            f'Expected eval_preds and eval_labels to have matching shape, got {logits.shape} and {labels.shape}.'
        )

    y_pred = logits > threshold
    y_true = labels > 0.5

    per_class_confusion = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    class_count = y_true.shape[1]
    class_names_list = list(class_names) if class_names is not None else []

    if class_names_list and len(class_names_list) != class_count:
        raise ValueError(
            f'Expected {class_count} class names, got {len(class_names_list)}.'
        )

    for c in range(y_true.shape[1]):
        true_c = y_true[:, c]
        pred_c = y_pred[:, c]

        tp_c = np.logical_and(pred_c, true_c).sum()
        fp_c = np.logical_and(pred_c, np.logical_not(true_c)).sum()
        fn_c = np.logical_and(np.logical_not(pred_c), true_c).sum()
        tn_c = np.logical_and(np.logical_not(pred_c), np.logical_not(true_c)).sum()

        precision_c = _safe_div(tp_c, tp_c + fp_c)
        recall_c = _safe_div(tp_c, tp_c + fn_c)
        f1_c = _safe_div(2 * tp_c, 2 * tp_c + fp_c + fn_c)

        per_class_precision.append(precision_c)
        per_class_recall.append(recall_c)
        per_class_f1.append(f1_c)

        total_tp += int(tp_c)
        total_fp += int(fp_c)
        total_fn += int(fn_c)

        class_label = class_names_list[c] if class_names_list else str(c)
        per_class_confusion.append({
            'class_index': c,
            'class_label': class_label,
            'tp': int(tp_c),
            'fp': int(fp_c),
            'fn': int(fn_c),
            'tn': int(tn_c),
            # Matrix layout: [[TN, FP], [FN, TP]]
            'matrix_2x2': [[int(tn_c), int(fp_c)], [int(fn_c), int(tp_c)]],
            'precision': precision_c,
            'recall': recall_c,
            'f1': f1_c,
            'support': int(true_c.sum()),
        })

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn)

    macro_precision = float(np.nanmean(per_class_precision)) if per_class_precision else float('nan')
    macro_recall = float(np.nanmean(per_class_recall)) if per_class_recall else float('nan')
    macro_f1 = float(np.nanmean(per_class_f1)) if per_class_f1 else float('nan')

    return {
        'per_class_confusion': per_class_confusion,
        'micro_precision': float(micro_precision),
        'macro_precision': float(macro_precision),
        'micro_recall': float(micro_recall),
        'macro_recall': float(macro_recall),
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
    }
