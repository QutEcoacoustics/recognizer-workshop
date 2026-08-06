import asyncio
import io

import numpy as np
import pandas as pd
import panel as pn

from recognizer_workshop.app.agile_engine import sanitize_model_name
from recognizer_workshop.ui.panel.utils.patch_tqdm import patch_tqdm
from recognizer_workshop.ui.panel.utils.validation import ensure_validation_css, set_validity


def build(vm):
    ensure_validation_css()

    # Create help buttons
    train_help_btn = vm.help_system.create_help_button("help-train-overview", "Help: Training")
    algorithm_params_help_btn = vm.help_system.create_help_button("help-training-algorithm-params", "Help: Algorithm Parameters")
    labeled_data_params_help_btn = vm.help_system.create_help_button("help-labeled-data-params", "Help: Labeled Data Parameters")

    # --- Model name (persisted in config) ---
    def _default_classifier_name():
        if vm.eng:
            if vm.eng.config.classifier_name:
                return vm.eng.config.classifier_name
            labels = vm.eng.get_target_labels() if hasattr(vm.eng, 'get_target_labels') else []
            if len(labels) == 1:
                return str(labels[0])
        return ""

    cls_name = pn.widgets.TextInput(
        name="Recognizer Name",
        value=_default_classifier_name(),
        placeholder="Required — e.g. imperial_pigeon",
        sizing_mode="stretch_width",
    )

    def _update_cls_name_validation(value: str):
        is_valid = bool(value.strip())
        set_validity(cls_name, is_valid=is_valid)
        btn_train.disabled = not is_valid
        return is_valid

    def _on_cls_name_input_change(event):
        _update_cls_name_validation(event.new)

    cls_name.param.watch(_on_cls_name_input_change, 'value_input')

    learning_rate = pn.widgets.FloatInput(name="Learning Rate", value=1e-3, step=1e-4, sizing_mode="fixed", width=100)
    num_steps     = pn.widgets.IntInput(name="Num Steps", value=101, start=1, sizing_mode="fixed", width=100)
    batch_size    = pn.widgets.IntInput(name="Batch Size", value=32, start=1, sizing_mode="fixed", width=100)
    train_ratio   = pn.widgets.FloatInput(name="Train Ratio", value=0.9, start=0.1, end=0.99, step=0.01, sizing_mode="fixed", width=100)
    weak_neg_weight = pn.widgets.FloatInput(name="Weak Neg Weight", value=0.05, start=0.0, step=0.01, sizing_mode="fixed", width=140)
    weak_negs_batch = pn.widgets.IntInput(name="Weak Neg Batch Size", value=16, start=0, sizing_mode="fixed", width=140)
    btn_auto_weak_negs = pn.widgets.Button(name="Auto Set", button_type="default", sizing_mode="fixed", width=100)
    balance_dataset = pn.widgets.Checkbox(name="Balance", value=True)

    btn_train = pn.widgets.Button(name="Train classifier", button_type="success", sizing_mode="fixed", width=150)
    btn_train.height = 40

    progress_bar = pn.widgets.Progress(name='Training Progress', active=True, visible=False, sizing_mode="stretch_width")

    # --- Results area ---
    metrics_view = pn.Column(sizing_mode="stretch_width")
    balancing_view = pn.Column(visible=False, sizing_mode="stretch_width")
    btn_toggle_balancing = pn.widgets.Toggle(
        name="▶ Show balancing info",
        button_type="light",
        sizing_mode="fixed",
        width=200,
        visible=False,
    )

    def _on_toggle_balancing(event):
        balancing_view.visible = event.new
        btn_toggle_balancing.name = "▼ Hide balancing info" if event.new else "▶ Show balancing info"

    btn_toggle_balancing.param.watch(_on_toggle_balancing, 'value')

    results_view = pn.Column(
        pn.pane.Markdown("Training results will be shown here."),
        sizing_mode="stretch_width",
    )

    # --- Saved models table ---
    models_table_view = pn.Column(sizing_mode="stretch_width")

    def _refresh_models_table():
        if not vm.eng:
            models_table_view.objects = [pn.pane.Markdown("_Initialize engine to view saved models._")]
            return
        models = vm.eng.list_saved_models()
        if not models:
            models_table_view.objects = [pn.pane.Markdown("_No saved models yet._")]
            return

        rows = []
        for m in models:
            dl_btn = pn.widgets.FileDownload(
                filename=f"{m['name']}.json",
                button_type="primary",
                label="Download",
                sizing_mode="fixed",
                width=110,
                embed=False,
            )
            model_path = m['path']

            def _make_callback(p):
                def _cb():
                    return io.BytesIO(p.read_bytes())
                return _cb

            dl_btn.callback = _make_callback(model_path)

            del_btn = pn.widgets.Button(
                name="Delete",
                button_type="danger",
                sizing_mode="fixed",
                width=80,
            )
            model_name = m['name']

            def _make_delete(n):
                def _del(_):
                    try:
                        vm.eng.delete_model(n)
                        vm.notify(f"Deleted model '{n}'.", "success")
                    except Exception as e:
                        vm.notify(f"Delete failed: {e}", "error")
                    _refresh_models_table()
                return _del

            del_btn.on_click(_make_delete(model_name))
            rows.append(pn.Row(
                pn.pane.Markdown(f"**{m['name']}**", width=200, sizing_mode="fixed"),
                pn.pane.Markdown(m['created'].strftime("%Y-%m-%d %H:%M"), width=160, sizing_mode="fixed"),
                dl_btn,
                del_btn,
                align="center",
            ))

        # header = pn.Row(
        #     pn.pane.Markdown("**Model**", width=200, sizing_mode="fixed"),
        #     pn.pane.Markdown("**Created**", width=160, sizing_mode="fixed"),
        #     pn.pane.Markdown("", width=110, sizing_mode="fixed"),
        #     pn.pane.Markdown("", width=80, sizing_mode="fixed"),
        # )
        models_table_view.objects = rows

    async def _train(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        raw_name = cls_name.value_input if cls_name.value_input is not None else cls_name.value
        if not raw_name.strip():
            _update_cls_name_validation(raw_name)
            return

        sanitized_name = sanitize_model_name(raw_name)
        if cls_name.value != sanitized_name:
            cls_name.value = sanitized_name
        vm.eng.config.classifier_name = sanitized_name
        try:
            vm.eng.config.save()
        except ValueError:
            pass  # config_path not set yet — fine, name is held in memory
        except Exception as e:
            vm.notify(f"Could not persist model name: {e}", "warning")

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

            # Auto-save
            try:
                version_dir = await asyncio.to_thread(vm.eng.save_classifier_versioned, train_result)
                vm.notify(f"Model saved: {version_dir}", "success")
                _refresh_models_table()
            except Exception as e:
                vm.notify(f"Auto-save failed: {e}", "warning")

            metrics_df = format_metrics_matrix_df(scores)
            view_objects = [
                pn.pane.Markdown("### Metrics"),
                render_html_table(metrics_df, show_header=True),
            ]

            # Balancing info (hidden by default)
            has_balancing = bool(info)
            btn_toggle_balancing.visible = has_balancing
            btn_toggle_balancing.value = False
            balancing_view.visible = False
            balancing_view.objects = []
            if has_balancing:
                bal_objects = []
                info_df = format_training_info_df(info)
                bal_objects.append(pn.pane.Markdown("### Training Setup & Balancing"))
                bal_objects.append(render_html_table(info_df, show_header=True))
                factors_df = format_oversample_factors_df(info)
                if factors_df is not None:
                    bal_objects.append(pn.pane.Markdown("### Oversample Factors"))
                    bal_objects.append(render_html_table(factors_df, show_header=True))
                balancing_view.objects = bal_objects

            results_view.objects = view_objects + [btn_toggle_balancing, balancing_view]
            vm.notify("Training complete.", "success")

        except Exception as e:
            results_view.objects = [pn.pane.Alert(f"**Error:** {e}", alert_type="danger")]
            vm.notify(f"Training failed: {e}", "error")
        finally:
            _update_cls_name_validation(cls_name.value_input if cls_name.value_input is not None else cls_name.value)
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

    btn_train.on_click(_train)
    btn_auto_weak_negs.on_click(_auto_set_weak_negs)
    _update_cls_name_validation(cls_name.value)

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        _refresh_models_table()

    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _on_config_state_change(_):
        _refresh_models_table()

    # Seed the models table on load
    _refresh_models_table()

    hyperparameters = pn.Card(
        pn.Column(
            pn.Row("Algorithm Parameters", pn.Spacer(), algorithm_params_help_btn, sizing_mode="stretch_width"),
            pn.Row(learning_rate, num_steps, batch_size, sizing_mode="stretch_width"),
            pn.Row("Labeled Data Parameters", pn.Spacer(), labeled_data_params_help_btn, sizing_mode="stretch_width"),
            pn.Row(train_ratio, weak_neg_weight, weak_negs_batch, btn_auto_weak_negs, balance_dataset, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        title="Hyperparameters",
        collapsible=False,
    )

    return pn.Column(
        pn.Row("## Train Classifier", pn.Spacer(), train_help_btn),
        pn.Row(
            cls_name,
            btn_train,
            align="end",
            sizing_mode="stretch_width",
        ),
        hyperparameters,
        progress_bar,
        pn.layout.Divider(),
        results_view,
        pn.layout.Divider(),
        pn.pane.Markdown("### Saved Models"),
        models_table_view,
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
        {'Setting': 'Balance dataset enabled', 'Value': str(bool(balance_enabled)) if balance_enabled is not None else 'n/a'},
        {'Setting': 'Oversampling applied', 'Value': str(bool(oversampling_applied)) if oversampling_applied is not None else 'n/a'},
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
                    'Pos count': entry.get('train_pos_count', ''),
                    'Neg count': entry.get('train_neg_count', ''),
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




