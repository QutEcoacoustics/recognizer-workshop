import asyncio

import numpy as np
import pandas as pd
import panel as pn

from recognizer_workshop.ui.panel.utils.patch_tqdm import patch_tqdm


def build(vm):
    # Create help buttons
    train_help_btn = vm.help_system.create_help_button("help-train-overview", "Help: Training")
    algorithm_params_help_btn = vm.help_system.create_help_button("help-training-algorithm-params", "Help: Algorithm Parameters")
    labeled_data_params_help_btn = vm.help_system.create_help_button("help-labeled-data-params", "Help: Labeled Data Parameters")

    learning_rate = pn.widgets.FloatInput(name="Learning Rate", value=1e-3, step=1e-4)
    num_steps     = pn.widgets.IntInput(name="Num Steps", value=101, start=1)
    batch_size    = pn.widgets.IntInput(name="Batch Size", value=32, start=1)
    train_ratio   = pn.widgets.FloatInput(name="Train Ratio", value=0.9, start=0.1, end=0.99, step=0.01)
    weak_neg_weight = pn.widgets.FloatInput(name="Weak Negative Weight", value=0.05, step=0.01)
    weak_negs_batch = pn.widgets.IntInput(name="Weak Negatives Batch Size", value=16, start=0)
    btn_auto_weak_negs = pn.widgets.Button(name="Auto Set", button_type="default", sizing_mode="fixed", width=100)
    balance_dataset = pn.widgets.Checkbox(name="Balance dataset", value=True)

    btn_train = pn.widgets.Button(name="Train classifier", button_type="success", sizing_mode="fixed", width=150)
    
    progress_bar = pn.widgets.Progress(name='Training Progress', active=True, visible=False, sizing_mode="stretch_width")

    results_view = pn.Column(
        pn.pane.Markdown("Training results will be shown here."),
        sizing_mode="stretch_width"
    )

    cls_name = pn.widgets.TextInput(name="Classifier name", value="my_classifier", sizing_mode="fixed", width=320)
    btn_save = pn.widgets.Button(name="Save classifier", button_type="primary", sizing_mode="fixed", width=140)

    async def _train(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")

        # Disable the button and show the progress bar to give user feedback
        btn_train.disabled = True
        progress_bar.visible = True
        results_view.objects = [pn.pane.Markdown("⏳ Training in progress... please wait.")]

        def run_training():
            return vm.eng.train_classifier(
                learning_rate=learning_rate.value,
                weak_neg_weight=weak_neg_weight.value,
                num_steps=num_steps.value,
                train_ratio=train_ratio.value,
                batch_size=batch_size.value,
                weak_negatives_batch_size=weak_negs_batch.value,
                balance_dataset=balance_dataset.value,
            )
        

        try:

            progress_bar.bar_color = 'success'
            progress_bar.value = 0 
            with patch_tqdm(progress_bar):
                train_result = await asyncio.to_thread(run_training)

            scores = train_result.get('metrics', train_result)
            info = train_result.get('info', {}) if isinstance(train_result, dict) else {}

            metrics_df = format_metrics_matrix_df(scores)
            view_objects = [
                pn.pane.Markdown("### Metrics"),
                render_html_table(metrics_df, show_header=True),
            ]
            if info:
                info_df = format_training_info_df(info)
                view_objects.append(pn.pane.Markdown("### Training Setup & Balancing"))
                view_objects.append(render_html_table(info_df, show_header=True))

                factors_df = format_oversample_factors_df(info)
                if factors_df is not None:
                    view_objects.append(pn.pane.Markdown("### Oversample Factors"))
                    view_objects.append(render_html_table(factors_df, show_header=True))
            results_view.objects = view_objects
            vm.notify("Training complete.", "success")
            
        except Exception as e:
            results_view.objects = [pn.pane.Alert(f"**Error:** {e}", alert_type="danger")]
            vm.notify(f"Training failed: {e}", "error")
        finally:
            # ALWAYS re-enable the button and hide the progress bar
            btn_train.disabled = False
            progress_bar.visible = False

    def _auto_set_weak_negs(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        try:
            resolved_batch, adjusted_weight, message = vm.eng.calculate_balanced_weak_negatives_batch_size(
                weak_neg_weight=weak_neg_weight.value,
                batch_size=batch_size.value,
            )
            weak_negs_batch.value = resolved_batch
            weak_neg_weight.value = adjusted_weight
            vm.notify(message, "info")
        except Exception as e:
            vm.notify(f"Could not auto-set weak negatives batch size: {e}", "error")


    def _save(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not vm.eng.has_trained_classifier():
            return vm.notify("No classifier has been trained yet.", "warn")
        try:
            out = (vm.eng.config.models_folder / f"{cls_name.value}.json").absolute()
            out.parent.mkdir(parents=True, exist_ok=True)
            vm.eng.save_classifier(out)
            vm.notify(f"Classifier saved: {out}", "success")
        except Exception as e:
            vm.notify(f"Save classifier failed: {e}", "error")

    btn_train.on_click(_train)
    btn_auto_weak_negs.on_click(_auto_set_weak_negs)
    btn_save.on_click(_save)

    # training params
    hyperparameters = pn.Card(
        pn.Row(
            pn.Column(
                pn.Row("Algorithm Parameters", pn.Spacer(), algorithm_params_help_btn),
                pn.Row(learning_rate, num_steps, batch_size),
                sizing_mode="stretch_width",
            ),
            pn.Column(
                pn.Row("Labeled Data Parameters", pn.Spacer(), labeled_data_params_help_btn),
                pn.Row(train_ratio, weak_neg_weight, weak_negs_batch, btn_auto_weak_negs, balance_dataset),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
        title="Hyperparameters",
        collapsible=False
    )

    return pn.Column(
        pn.Row("## Train Classifier", pn.Spacer(), train_help_btn),
        hyperparameters,
        pn.Row(btn_train),
        progress_bar,
        pn.layout.Divider(),
        results_view,
        pn.layout.Divider(),
        pn.pane.Markdown("### Save Model"),
        pn.Row(cls_name, btn_save, align="start"),
        sizing_mode="stretch_both",
    )


def format_metrics_matrix_df(scores: dict) -> pd.DataFrame:
    """Create a single metrics table with class columns and optional aggregate column."""
    per_class = scores.get('per_class_confusion')
    if not isinstance(per_class, list):
        per_class = []

    class_labels = [str(item.get('class_label', item.get('class_index', i))) for i, item in enumerate(per_class)]
    class_count = len(class_labels)
    include_aggregate = class_count > 1

    if class_count == 0:
        rows = []
        for key in ['top1_acc', 'roc_auc', 'cmap', 'micro_precision', 'macro_precision', 'micro_recall', 'macro_recall', 'micro_f1', 'macro_f1']:
            if key in scores:
                rows.append({'Metric': prettify_metric_name(key), 'Value': _format_metric_value(scores.get(key))})
        return pd.DataFrame(rows, columns=['Metric', 'Value'])

    def _series_from_key(key: str, default: float | int = 0):
        return [item.get(key, default) for item in per_class]

    def _array_metric(key: str):
        value = scores.get(key)
        if value is None:
            return [None] * class_count
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, list):
            return [None] * class_count
        if len(value) < class_count:
            value = value + [None] * (class_count - len(value))
        return value[:class_count]

    table_rows = []

    def add_row(metric_name: str, per_class_values: list, aggregate_value=None):
        row = {'Metric': metric_name}
        for cls, val in zip(class_labels, per_class_values):
            row[cls] = _format_metric_value(val)
        if include_aggregate:
            row['Aggregate'] = _format_metric_value(aggregate_value)
        table_rows.append(row)

    add_row('ROC AUC', _array_metric('roc_auc_individual'), scores.get('roc_auc'))
    add_row('CMAP', _array_metric('cmap_individual'), scores.get('cmap'))
    add_row('TP', _series_from_key('tp'), sum(_series_from_key('tp')))
    add_row('FP', _series_from_key('fp'), sum(_series_from_key('fp')))
    add_row('FN', _series_from_key('fn'), sum(_series_from_key('fn')))
    add_row('TN', _series_from_key('tn'), sum(_series_from_key('tn')))
    add_row('Support', _series_from_key('support'), sum(_series_from_key('support')))
    add_row('Precision', _series_from_key('precision', float('nan')), scores.get('micro_precision'))
    add_row('Recall', _series_from_key('recall', float('nan')), scores.get('micro_recall'))
    add_row('F1', _series_from_key('f1', float('nan')), scores.get('micro_f1'))

    if include_aggregate:
        add_row('Macro Precision', [''] * class_count, scores.get('macro_precision'))
        add_row('Macro Recall', [''] * class_count, scores.get('macro_recall'))
        add_row('Macro F1', [''] * class_count, scores.get('macro_f1'))
        if 'top1_acc' in scores:
            add_row('Top-1 Accuracy', [''] * class_count, scores.get('top1_acc'))

    column_order = ['Metric', *class_labels]
    if include_aggregate:
        column_order.append('Aggregate')

    return pd.DataFrame(table_rows, columns=column_order)


def prettify_metric_name(metric_key: str) -> str:
    return metric_key.replace('_', ' ').title()


def _format_metric_value(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ''
        return f'{float(value):.3f}'
    return str(value)


def format_training_info_df(info: dict) -> pd.DataFrame:
    weak_negs = info.get('resolved_weak_negatives_batch_size')
    balance_enabled = info.get('balance_dataset_enabled')
    oversampling_applied = info.get('oversampling_applied')
    requested = info.get('oversampled_train_ids_requested_count')
    used = info.get('oversampled_train_ids_used_count')
    excluded_eval = info.get('excluded_eval_from_oversampled_count')
    original_train = info.get('original_train_ids_count')
    eval_count = info.get('eval_ids_count')

    rows = [
        {'Setting': 'Weak negatives batch size', 'Value': weak_negs if weak_negs is not None else 'n/a'},
        {'Setting': 'Balance dataset enabled', 'Value': str(bool(balance_enabled))},
        {'Setting': 'Oversampling applied', 'Value': str(bool(oversampling_applied))},
    ]

    if requested is not None:
        rows.append({'Setting': 'Oversampled IDs requested', 'Value': str(requested)})
    if used is not None:
        rows.append({'Setting': 'Oversampled IDs used in train split', 'Value': str(used)})
    if excluded_eval is not None:
        rows.append({'Setting': 'Oversampled IDs excluded (in eval split)', 'Value': str(excluded_eval)})
    if original_train is not None:
        rows.append({'Setting': 'Original train IDs before oversampling', 'Value': str(original_train)})
    if eval_count is not None:
        rows.append({'Setting': 'Eval IDs', 'Value': str(eval_count)})

    return pd.DataFrame(rows, columns=['Setting', 'Value'])


def format_oversample_factors_df(info: dict) -> pd.DataFrame | None:
    # Prefer the richer per-label breakdown when available (includes original counts).
    breakdown = info.get('per_label_oversample_breakdown')
    if isinstance(breakdown, dict) and breakdown:
        rows = []
        for label, entry in breakdown.items():
            rows.append(
                {
                    'Label': str(label),
                    'Pos count': entry.get('pos_count', ''),
                    'Neg count': entry.get('neg_count', ''),
                    'Pos factor': _format_metric_value(entry.get('pos_factor', 1.0)),
                    'Neg factor': _format_metric_value(entry.get('neg_factor', 1.0)),
                }
            )
        return pd.DataFrame(rows, columns=['Label', 'Pos count', 'Neg count', 'Pos factor', 'Neg factor'])

    # Fall back to the bare factors dict for older results.
    factors = info.get('oversample_factors')
    if not isinstance(factors, dict) or not factors:
        return None
    rows = []
    for label, factor_info in factors.items():
        rows.append(
            {
                'Label': str(label),
                'Pos factor': _format_metric_value(factor_info.get('pos_factor', 1.0)),
                'Neg factor': _format_metric_value(factor_info.get('neg_factor', 1.0)),
            }
        )
    return pd.DataFrame(rows, columns=['Label', 'Pos factor', 'Neg factor'])


def render_html_table(df: pd.DataFrame, show_header: bool = True):
    """Render a non-interactive HTML table for stable Panel UX."""
    html_table = df.to_html(
        index=False,
        header=show_header,
        classes='rw-table',
        border=0,
        escape=True,
    )
    html = (
        "<style>"
        ".rw-table {width: 100%; border-collapse: collapse; font-size: 0.95rem;}"
        ".rw-table th, .rw-table td {border: 1px solid #d9d9d9; padding: 6px 8px; text-align: left;}"
        ".rw-table th {background: #f6f6f6; font-weight: 600;}"
        ".rw-table tbody tr:nth-child(even) {background: #fcfcfc;}"
        "</style>"
        f"{html_table}"
    )
    return pn.pane.HTML(html, sizing_mode='stretch_width')




