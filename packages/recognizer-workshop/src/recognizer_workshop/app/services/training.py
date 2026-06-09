from dataclasses import dataclass
import logging
from typing import Any, Sequence

import numpy as np
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data

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


def train_classifier(db, target_labels: Sequence[str], params: TrainClassifierParams) -> TrainingResult:
    """Train a linear classifier and return model + metrics.

    This function does not mutate engine/UI state. It handles training summary
    logging and returns the model with eval scores.
    """
    data_manager = classifier_data.AgileDataManager(
        target_labels=tuple(target_labels),
        db=db,
        train_ratio=params.train_ratio,
        min_eval_examples=1,
        batch_size=params.batch_size,
        weak_negatives_batch_size=params.weak_negatives_batch_size,
        rng=np.random.default_rng(seed=params.seed),
    )

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

    return TrainingResult(classifier_model=linear_classifier, eval_scores=output_scores)


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
