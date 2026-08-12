import asyncio

import numpy as np
import pandas as pd
import panel as pn
from panel_material_ui import Dialog

from recognizer_workshop.app.agile_engine import sanitize_classifier_name
from recognizer_workshop.app.services import training as training_service
from recognizer_workshop.ui.panel.utils.file_download import make_file_download_button
from recognizer_workshop.ui.panel.utils.patch_tqdm import patch_tqdm
from recognizer_workshop.ui.panel.utils.validation import ensure_validation_css, set_validity


def build(vm):
    ensure_validation_css()

    # Create help buttons
    train_help_btn = vm.help_system.create_help_button("help-train-overview", "Help: Training")
    algorithm_params_help_btn = vm.help_system.create_help_button("help-training-algorithm-params", "Help: Algorithm Parameters")
    labeled_data_params_help_btn = vm.help_system.create_help_button("help-labeled-data-params", "Help: Labeled Data Parameters")

    # --- Classifier name (auto-populated from app state) ---
    def _preferred_classifier_name() -> str:
        if not vm.eng:
            return ""

        index = getattr(vm.eng, "saved_training_results_index", None)
        selected_item = index.get_active_classifier_entry() if index else None
        if selected_item and selected_item.classifier_name:
            return str(selected_item.classifier_name)

        labels = vm.eng.get_target_labels() if hasattr(vm.eng, 'get_target_labels') else []
        if len(labels) == 1:
            return str(labels[0])

        return ""

    _recognizer_name_autofill_state = {"last_auto": _preferred_classifier_name()}

    recognizer_name_field = pn.widgets.TextInput(
        name="Recognizer Name",
        value=_recognizer_name_autofill_state["last_auto"],
        placeholder="Required — e.g. imperial_pigeon",
        sizing_mode="stretch_width",
    )

    _training_state = {"in_progress": False}

    def _update_recognizer_name_validation(value: str):
        is_valid = bool(value.strip())
        set_validity(recognizer_name_field, is_valid=is_valid)
        btn_train.disabled = not is_valid or _training_state["in_progress"]
        return is_valid

    def _on_recognizer_name_input_change(event):
        _update_recognizer_name_validation(event.new)

    def _sync_recognizer_name_autofill_from_state():
        preferred = _preferred_classifier_name()
        current = (recognizer_name_field.value or "").strip()
        last_auto = (_recognizer_name_autofill_state["last_auto"] or "").strip()

        # Only auto-overwrite when the user hasn't diverged from the auto-filled value.
        if current == "" or current == last_auto:
            if recognizer_name_field.value != preferred:
                recognizer_name_field.value = preferred
            _update_recognizer_name_validation(preferred)
        _recognizer_name_autofill_state["last_auto"] = preferred

    recognizer_name_field.param.watch(_on_recognizer_name_input_change, 'value_input')

    learning_rate = pn.widgets.FloatInput(name="Learning Rate", value=1e-3, step=1e-4, sizing_mode="fixed", width=100)
    l1_strength = pn.widgets.FloatInput(name="L1 Regularization", value=0.0, start=0.0, step=1e-4, sizing_mode="fixed", width=140)
    l2_strength = pn.widgets.FloatInput(name="L2 Regularization", value=0.0, start=0.0, step=1e-4, sizing_mode="fixed", width=140)
    num_steps     = pn.widgets.IntInput(name="Num Steps", value=101, start=1, sizing_mode="fixed", width=100)
    batch_size    = pn.widgets.IntInput(name="Batch Size", value=32, start=1, sizing_mode="fixed", width=100)
    train_ratio   = pn.widgets.FloatInput(name="Train Ratio", value=0.9, start=0.1, end=0.99, step=0.01, sizing_mode="fixed", width=100)
    weak_neg_weight = pn.widgets.FloatInput(name="Weak Neg Weight", value=0.05, start=0.0, step=0.01, sizing_mode="fixed", width=140)
    weak_neg_batch_size = pn.widgets.IntInput(name="Weak Neg Batch Size", value=16, start=0, sizing_mode="fixed", width=140)
    btn_auto_set_weak_negs = pn.widgets.Button(name="Auto Set", button_type="default", sizing_mode="fixed", width=100)
    balance_dataset = pn.widgets.Checkbox(name="Balance", value=True)

    btn_train = pn.widgets.Button(name="Train recognizer", button_type="success", sizing_mode="fixed", width=150)
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

    # --- Saved classifiers selector ---
    classifier_select = pn.widgets.Select(name="Recognizer", options=[], sizing_mode="fixed", width=290)
    version_select = pn.widgets.Select(name="Version", options=[], sizing_mode="fixed", width=200)
    btn_delete = pn.widgets.Button(name="Delete", button_type="danger", sizing_mode="fixed", width=90)
    saved_classifiers_empty_msg = pn.pane.Markdown("_No saved recognizers found._", visible=False)
    saved_classifiers_dialogs = pn.Column()

    _selector_sync_guard = {"active": False}

    def _get_saved_training_results_index():
        return vm.eng.saved_training_results_index if vm.eng else None

    def _notify_active_classifier_changed():
        vm.obs.bump("engine_state")

    def _get_recognizer_path():
        index = _get_saved_training_results_index()
        selected_item = index.get_active_classifier_entry() if index else None
        return selected_item.recognizer_path if selected_item else None

    btn_download = make_file_download_button(
        get_path=_get_recognizer_path,
        filename="recognizer.json",
        label="Download",
        button_type="primary",
        width=110,
    )

    def _refresh_saved_classifiers_selector_from_engine():
        """Mirrors the engine's saved-classifier index/active-selection onto the dropdowns."""
        _selector_sync_guard["active"] = True
        try:
            index = _get_saved_training_results_index()
            classifier_names = index.get_classifier_list() if index else []
            if not classifier_names:
                classifier_select.options = []
                classifier_select.value = None
                version_select.options = []
                version_select.value = None
                saved_classifiers_empty_msg.visible = True
                btn_download.disabled = True
                btn_delete.disabled = True
                results_view.objects = []
                btn_toggle_balancing.visible = False
                balancing_view.visible = False
                balancing_view.objects = []
                return

            saved_classifiers_empty_msg.visible = False
            btn_download.disabled = False
            btn_delete.disabled = False

            selected_item = index.get_active_classifier_entry() if index else None
            selected_classifier_name = selected_item.classifier_name if selected_item else classifier_names[0]

            classifier_select.options = classifier_names
            classifier_select.value = selected_classifier_name

            versions = index.get_versions_for_classifier(selected_classifier_name) if index else []
            version_select.options = {v.display_name: v.version_name for v in versions}
            if selected_item and selected_item.classifier_name == selected_classifier_name:
                selected_classifier_version = selected_item.version_name
            else:
                selected_classifier_version = versions[0].version_name if versions else None
            version_select.value = selected_classifier_version

            if selected_classifier_name and selected_classifier_version:
                btn_download.filename = f"{selected_classifier_name}_{selected_classifier_version}.json"
                _render_active_saved_classifier_results()
                load_error = getattr(vm.eng, 'train_result_load_error', None)
                if load_error:
                    vm.notify(f"Could not load {selected_classifier_name} ({selected_classifier_version}): {load_error}", "warning")
        finally:
            _selector_sync_guard["active"] = False

    def _render_active_saved_classifier_results():
        index = _get_saved_training_results_index()
        selected_item = index.get_active_classifier_entry() if index else None
        if not selected_item or not vm.eng:
            return

        if not vm.eng.train_result:
            load_error = getattr(vm.eng, 'train_result_load_error', None)
            if load_error:
                results_view.objects = [pn.pane.Alert(
                    f"**Could not load {selected_item.classifier_name} ({selected_item.display_name}):**\n\n{load_error}",
                    alert_type="warning",
                )]
            btn_toggle_balancing.visible = False
            balancing_view.visible = False
            balancing_view.objects = []
            return

        view_result = training_service.training_result_to_json(vm.eng.train_result)
        scores = view_result.get('metrics')
        info = view_result.get('info') or {}

        if not scores:
            results_view.objects = [pn.pane.Markdown(
                f"**{selected_item.classifier_name}** - {selected_item.display_name}\n\n_No metrics found._"
            )]
            btn_toggle_balancing.visible = False
            balancing_view.visible = False
            balancing_view.objects = []
            return

        metrics_df = format_metrics_matrix_df(scores)
        view_objects = [
            pn.pane.Markdown(f"### Metrics - {selected_item.classifier_name} ({selected_item.display_name})"),
            render_html_table(metrics_df, show_header=True),
        ]

        has_balancing = bool(info)
        btn_toggle_balancing.visible = has_balancing
        btn_toggle_balancing.value = False
        balancing_view.visible = False
        balancing_view.objects = []
        if has_balancing:
            balancing_objects = []
            info_df = format_training_info_df(info)
            balancing_objects.append(pn.pane.Markdown("### Training Setup & Balancing"))
            balancing_objects.append(render_html_table(info_df, show_header=True))
            factors_df = format_oversample_factors_df(info)
            if factors_df is not None:
                balancing_objects.append(pn.pane.Markdown("### Oversample Factors"))
                balancing_objects.append(render_html_table(factors_df, show_header=True))
            balancing_view.objects = balancing_objects

        results_view.objects = view_objects + [btn_toggle_balancing, balancing_view]

    def _on_classifier_select_change(event):
        if _selector_sync_guard["active"] or not vm.eng or event.new is None:
            return
        classifier_name = event.new
        index = _get_saved_training_results_index()
        versions = index.get_versions_for_classifier(classifier_name) if index else []
        if not versions:
            return
        newest_version = versions[0].version_name
        try:
            vm.eng.set_active_saved_classifier(classifier_name, newest_version)
        except Exception as e:
            vm.notify(f"Could not switch active recognizer: {e}", "error")
            return

        _selector_sync_guard["active"] = True
        try:
            version_select.options = {v.display_name: v.version_name for v in versions}
            version_select.value = newest_version
            btn_download.filename = f"{classifier_name}_{newest_version}.json"
        finally:
            _selector_sync_guard["active"] = False
        _notify_active_classifier_changed()
        _render_active_saved_classifier_results()

    def _on_version_select_change(event):
        if _selector_sync_guard["active"] or not vm.eng or event.new is None:
            return
        classifier_name = classifier_select.value
        version_name = event.new
        try:
            vm.eng.set_active_saved_classifier(classifier_name, version_name)
            btn_download.filename = f"{classifier_name}_{version_name}.json"
        except Exception as e:
            vm.notify(f"Could not switch active version: {e}", "error")
            return
        _notify_active_classifier_changed()
        _render_active_saved_classifier_results()

    classifier_select.param.watch(_on_classifier_select_change, 'value')
    version_select.param.watch(_on_version_select_change, 'value')

    def _on_delete_click(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        index = _get_saved_training_results_index()
        selected_item = index.get_active_classifier_entry() if index else None
        if not selected_item:
            return vm.notify("No saved recognizer selected.", "warning")

        btn_cancel = pn.widgets.Button(name="Cancel", button_type="default")
        btn_confirm = pn.widgets.Button(name="Yes, delete it", button_type="danger")

        def _cancel(_evt):
            dialog.open = False

        def _confirm(_evt):
            try:
                vm.eng.delete_active_saved_classifier_version()
                vm.notify(f"Deleted '{selected_item.classifier_name}' ({selected_item.display_name}).", "success")
            except Exception as e:
                vm.notify(f"Delete failed: {e}", "error")
            dialog.open = False
            _refresh_saved_classifiers_selector_from_engine()
            _notify_active_classifier_changed()

        btn_cancel.on_click(_cancel)
        btn_confirm.on_click(_confirm)

        dialog = Dialog(
            pn.Column(
                f"Delete **{selected_item.classifier_name}** version **{selected_item.display_name}**? This cannot be undone.",
                pn.Row(btn_cancel, btn_confirm),
            ),
            title="Delete Saved Recognizer",
        )
        saved_classifiers_dialogs[:] = [dialog]
        dialog.open = True

    btn_delete.on_click(_on_delete_click)


    async def _train(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        raw_name = recognizer_name_field.value
        if not raw_name.strip():
            _update_recognizer_name_validation(raw_name)
            return

        sanitized_name = sanitize_classifier_name(raw_name)
        if recognizer_name_field.value != sanitized_name:
            recognizer_name_field.value = sanitized_name

        btn_train.disabled = True
        _training_state["in_progress"] = True
        progress_bar.visible = True
        results_view.objects = [pn.pane.Markdown("⏳ Training in progress... please wait.")]

        def run_training():
            return vm.eng.train_classifier(
                classifier_name=sanitized_name,
                learning_rate=learning_rate.value,
                l1_strength=l1_strength.value,
                l2_strength=l2_strength.value,
                weak_neg_weight=weak_neg_weight.value,
                num_steps=num_steps.value,
                train_ratio=train_ratio.value,
                batch_size=batch_size.value,
                weak_negatives_batch_size=weak_neg_batch_size.value,
                balance_dataset=balance_dataset.value,
            )

        try:
            progress_bar.bar_color = 'success'
            progress_bar.value = 0
            with patch_tqdm(progress_bar):
                train_result = await asyncio.to_thread(run_training)

            vm.notify(f"Recognizer saved: {train_result.get('version_dir')}", "success")
            _refresh_saved_classifiers_selector_from_engine()
            _notify_active_classifier_changed()

            scores = train_result.get('metrics', train_result)
            info = train_result.get('info', {}) if isinstance(train_result, dict) else {}

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
                balancing_objects = []
                info_df = format_training_info_df(info)
                balancing_objects.append(pn.pane.Markdown("### Training Setup & Balancing"))
                balancing_objects.append(render_html_table(info_df, show_header=True))
                factors_df = format_oversample_factors_df(info)
                if factors_df is not None:
                    balancing_objects.append(pn.pane.Markdown("### Oversample Factors"))
                    balancing_objects.append(render_html_table(factors_df, show_header=True))
                balancing_view.objects = balancing_objects

            results_view.objects = view_objects + [btn_toggle_balancing, balancing_view]
            vm.notify("Training complete.", "success")

        except Exception as e:
            results_view.objects = [pn.pane.Alert(f"**Error:** {e}", alert_type="danger")]
            vm.notify(f"Training failed: {e}", "error")
        finally:
            _training_state["in_progress"] = False
            _update_recognizer_name_validation(recognizer_name_field.value)
            progress_bar.visible = False

    def _auto_set_weak_negs(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        try:
            resolved_batch, adjusted_weight, message = vm.eng.calculate_balanced_weak_negatives_batch_size(
                weak_neg_weight=weak_neg_weight.value,
                batch_size=batch_size.value,
            )
            weak_neg_batch_size.value = resolved_batch
            weak_neg_weight.value = adjusted_weight
            vm.notify(message, "info")
        except Exception as e:
            vm.notify(f"Could not auto-set weak negatives batch size: {e}", "error")

    btn_train.on_click(_train)
    btn_auto_set_weak_negs.on_click(_auto_set_weak_negs)
    _update_recognizer_name_validation(recognizer_name_field.value)

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        _refresh_saved_classifiers_selector_from_engine()
        _sync_recognizer_name_autofill_from_state()

    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _on_config_state_change(_):
        _refresh_saved_classifiers_selector_from_engine()
        _sync_recognizer_name_autofill_from_state()

    # Seed the selector on load
    _refresh_saved_classifiers_selector_from_engine()
    _sync_recognizer_name_autofill_from_state()

    hyperparameters = pn.Card(
        pn.Column(
            pn.Row("Algorithm Parameters", pn.Spacer(), algorithm_params_help_btn, sizing_mode="stretch_width"),
            pn.Row(learning_rate, l1_strength, l2_strength, num_steps, batch_size, sizing_mode="stretch_width"),
            pn.Row("Labeled Data Parameters", pn.Spacer(), labeled_data_params_help_btn, sizing_mode="stretch_width"),
            pn.Row(train_ratio, weak_neg_weight, weak_neg_batch_size, btn_auto_set_weak_negs, balance_dataset, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        title="Hyperparameters",
        collapsible=False,
    )

    return pn.Column(
        pn.Row("## Train Recognizer", pn.Spacer(), train_help_btn),
        pn.Row(
            recognizer_name_field,
            btn_train,
            align="end",
            sizing_mode="stretch_width",
        ),
        hyperparameters,
        progress_bar,
        pn.layout.Divider(),
        pn.pane.Markdown("### Saved Recognizers"),
        saved_classifiers_empty_msg,
        pn.Row(classifier_select, version_select, btn_download, btn_delete, align="end"),
        saved_classifiers_dialogs,
        pn.layout.Divider(),
        results_view,
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




