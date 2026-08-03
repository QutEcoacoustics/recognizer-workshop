from dataclasses import dataclass
import math
import logging
from typing import Any, Sequence

import numpy as np
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
from perch_hoplite.db import interface as hoplite_interface

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


@dataclass(frozen=True)
class TrainingResult:
    classifier_model: Any
    eval_scores: dict[str, Any]
    training_info: dict[str, Any]


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
    """Build a training ID array oversampling strictly from the train split."""
    oversampled_embedding_id_chunks: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    train_set = set(train_ids.tolist())

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
        
        all_pos_ids = db.get_embeddings_by_label(label, hoplite_interface.LabelType.POSITIVE, None)
        all_neg_ids = db.get_embeddings_by_label(label, hoplite_interface.LabelType.NEGATIVE, None)

        # Filter down to the train split before any duplication
        train_pos_ids = np.array([x for x in all_pos_ids if x in train_set], dtype=np.int64)
        train_neg_ids = np.array([x for x in all_neg_ids if x in train_set], dtype=np.int64)

        pos_repeat_factor = max(label_factors['pos_factor'], 1.0)
        neg_repeat_factor = max(label_factors['neg_factor'], 1.0)

        if len(train_pos_ids) > 0:
            oversampled_embedding_id_chunks.append(
                _repeat_ids_with_fraction(train_pos_ids, pos_repeat_factor)
            )
        if len(train_neg_ids) > 0:
            oversampled_embedding_id_chunks.append(
                _repeat_ids_with_fraction(train_neg_ids, neg_repeat_factor)
            )

    if not oversampled_embedding_id_chunks:
        return np.array([], dtype=np.int64)

    return np.concatenate(oversampled_embedding_id_chunks)



def train_classifier(
    db,
    target_labels: Sequence[str],
    params: TrainClassifierParams,
    balance_dataset: bool = True,
) -> TrainingResult:
    """Train a linear classifier and return model + metrics."""
    # 1. Initialize data manager to generate the definitive train/test split first
    data_manager = classifier_data.AgileDataManager(
        target_labels=tuple(target_labels),
        db=db,
        train_ratio=params.train_ratio,
        min_eval_examples=1,
        batch_size=params.batch_size,
        weak_negatives_batch_size=params.weak_negatives_batch_size,
        rng=np.random.default_rng(seed=params.seed),
    )

    original_train_ids, original_eval_ids = data_manager.get_train_test_split()
    
    oversampling_applied = False
    oversampled_used_count = 0
    auto_oversample_factors = None

    # 2. Oversample specifically on the training split
    if balance_dataset:
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
            # Overwrite the split function to serve the directly oversampled array
            def _oversampled_split():
                return oversampled_train_ids, original_eval_ids
            
            data_manager.get_train_test_split = _oversampled_split
            oversampling_applied = True
            oversampled_used_count = int(len(oversampled_train_ids))

    linear_classifier, eval_scores = classifier.train_linear_classifier(
        data_manager=data_manager,
        learning_rate=params.learning_rate,
        weak_neg_weight=params.weak_neg_weight,
        num_train_steps=params.num_steps,
    )

    confusion_metrics = calculate_confusion_and_prf_metrics(
        eval_scores.get('eval_preds'),
        eval_scores.get('eval_labels'),
        class_names=target_labels,
        threshold=params.f1_threshold,
    )
    output_scores = dict(eval_scores)
    output_scores.update(confusion_metrics)

    top1 = _require_metric(output_scores, 'top1_acc')
    rocauc = _require_metric(output_scores, 'roc_auc')
    cmap = _require_metric(output_scores, 'cmap')
    micro_precision = _require_metric(output_scores, 'micro_precision')
    macro_precision = _require_metric(output_scores, 'macro_precision')
    micro_recall = _require_metric(output_scores, 'micro_recall')
    macro_recall = _require_metric(output_scores, 'macro_recall')
    micro_f1 = _require_metric(output_scores, 'micro_f1')
    macro_f1 = _require_metric(output_scores, 'macro_f1')

    log.info('%s', '\n' + '-' * 80)
    log.info('top-1      %.3f', top1)
    log.info('roc_auc    %.3f', rocauc)
    log.info('cmap       %.3f', cmap)
    log.info('micro_p    %.3f', micro_precision)
    log.info('macro_p    %.3f', macro_precision)
    log.info('micro_r    %.3f', micro_recall)
    log.info('macro_r    %.3f', macro_recall)
    log.info('micro_f1   %.3f', micro_f1)
    log.info('macro_f1   %.3f', macro_f1)
    log.info('weak_neg_batch %s', params.weak_negatives_batch_size)

    training_info = {
        'resolved_weak_negatives_batch_size': params.weak_negatives_batch_size,
        'balance_dataset_enabled': bool(balance_dataset),
        'oversampling_applied': oversampling_applied,
        'oversampled_train_ids_requested_count': oversampled_used_count,
        'oversampled_train_ids_used_count': oversampled_used_count,
        'excluded_eval_from_oversampled_count': 0, # Metric preserved for UI compatibility, but structurally eliminated
        'original_train_ids_count': int(len(original_train_ids)),
        'eval_ids_count': int(len(original_eval_ids)),
    }
    
    if auto_oversample_factors is not None:
        training_info['oversample_factors'] = auto_oversample_factors
        # Output breakdown strictly against the training split to verify logic
        train_set = set(original_train_ids.tolist())
        breakdown = {}
        for label in target_labels:
            if label in auto_oversample_factors:
                pos_ids = db.get_embeddings_by_label(label, hoplite_interface.LabelType.POSITIVE, None)
                neg_ids = db.get_embeddings_by_label(label, hoplite_interface.LabelType.NEGATIVE, None)
                breakdown[label] = {
                    'train_pos_count': sum(1 for x in pos_ids if x in train_set),
                    'train_neg_count': sum(1 for x in neg_ids if x in train_set),
                    'pos_factor': auto_oversample_factors[label]['pos_factor'],
                    'neg_factor': auto_oversample_factors[label]['neg_factor'],
                }
        training_info['per_label_oversample_breakdown'] = breakdown

    return TrainingResult(
        classifier_model=linear_classifier,
        eval_scores=output_scores,
        training_info=training_info,
    )


def training_result_to_json(result: TrainingResult) -> dict[str, Any]:
    """Create a JSON-safe dict for UI consumption from TrainingResult."""

    def _to_json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_json_safe(v) for v in value]
        return value

    metrics = dict(result.eval_scores)
    # Large raw arrays are not needed by the UI table rendering.
    metrics.pop('eval_preds', None)
    metrics.pop('eval_labels', None)

    return {
        'metrics': _to_json_safe(metrics),
        'info': _to_json_safe(result.training_info),
    }


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

    if logits is None or labels is None:
        return {
            'per_class_confusion': [],
            'micro_precision': float('nan'),
            'macro_precision': float('nan'),
            'micro_recall': float('nan'),
            'macro_recall': float('nan'),
            'micro_f1': float('nan'),
            'macro_f1': float('nan'),
        }

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.size == 0 or labels.size == 0:
        return {
            'per_class_confusion': [],
            'micro_precision': float('nan'),
            'macro_precision': float('nan'),
            'micro_recall': float('nan'),
            'macro_recall': float('nan'),
            'micro_f1': float('nan'),
            'macro_f1': float('nan'),
        }

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
