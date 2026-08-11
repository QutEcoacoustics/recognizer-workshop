from pathlib import Path
import io
import tempfile

import pandas as pd
import panel as pn

from recognizer_workshop.app.agile_modeling_helpers import make_inference_subset

import logging

from recognizer_workshop.app.inference.inference_result import InferenceResultItem

log = logging.getLogger(__name__)

def build(vm):
    def _ensure_tabulator():
        if getattr(_ensure_tabulator, "_loaded", False):
            return
        pn.extension('tabulator')
        _ensure_tabulator._loaded = True

    # Create help buttons
    inference_help_btn = vm.help_system.create_help_button("help-inference-overview", "Help: Inference")

    # --- Widgets ---
    threshold = pn.widgets.FloatInput(name="Threshold", value=0.0, step=0.1)
    subset = pn.widgets.FloatInput(
        name="Subset proportion (0..1)", value=0.001, start=0.001, end=1.0, step=0.05
    )
    target_label_selector = pn.widgets.CheckBoxGroup(
        name="Labels", options=[], value=[], inline=False, sizing_mode="stretch_width"
    )
    btn_run = pn.widgets.Button(
        name="Run inference", button_type="primary", sizing_mode="fixed", width=180, height=40
    )

    # Results viewer and uploader widgets
    inference_results_selector = pn.widgets.Select(name="Inference Run", options=[], sizing_mode="stretch_width")
    download_subset = pn.widgets.Checkbox(name="Download subset", value=False)
    subset_limit = pn.widgets.IntInput(name="Subset limit per label", value=80, start=1, step=10, visible=False)
    btn_subset = pn.widgets.FileDownload(
        filename="inference_results_subset.csv",
        label="Download results",
        button_type="primary",
        sizing_mode="fixed",
        width=180,
        height=40,
        embed=False,
    )
    project_id = pn.widgets.IntInput(name="Project ID (optional)", value=0, start=0)
    btn_upload = pn.widgets.Button(name="Upload detections", button_type="success")

    step1_left_column = pn.Column(
        pn.Row(threshold, subset),
        sizing_mode="stretch_width",
        styles={"flex": "1 1 100%"},
    )
    step1_right_column = pn.Column(
        target_label_selector,
        sizing_mode="stretch_width",
        styles={"flex": "0 0 25%"},
    )
    step1_columns = pn.Row(
        step1_left_column,
        sizing_mode="stretch_width",
    )

    # Sections that depend on file existence
    section2 = pn.Column()
    section3 = pn.Column()

    def _get_active_saved_classifier_entry():
        if not vm.eng:
            return None
        index = getattr(vm.eng, "saved_training_results_index", None)
        return index.get_active_classifier_entry() if index else None

    def _build_inference_heading_text() -> str:
        active_entry = _get_active_saved_classifier_entry()
        if not active_entry:
            return "## Run Inference"
        return f"## Run Inference - {active_entry.classifier_name} ({active_entry.display_name})"

    inference_heading = pn.pane.Markdown(_build_inference_heading_text())

    def _get_active_classifier_labels() -> list[str]:
        if not vm.eng or not vm.eng.has_trained_classifier():
            return []
        classifier = vm.eng.get_trained_classifier()
        return [str(label) for label in getattr(classifier, "classes", ())]

    def _sync_target_label_selector() -> None:
        labels = _get_active_classifier_labels()
        has_multiple_labels = len(labels) > 1
        target_label_selector.options = labels
        target_label_selector.value = labels
        target_label_selector.visible = has_multiple_labels
        step1_left_column.styles = {
            "flex": "0 0 75%" if has_multiple_labels else "1 1 100%"
        }
        step1_columns.objects = (
            [step1_left_column, step1_right_column]
            if has_multiple_labels
            else [step1_left_column]
        )


    def _create_csv_display(inference_result_item: InferenceResultItem):
        """Create a display widget for CSV information"""
        info = inference_result_item.get_info()
        
        if 'error' in info:
            return pn.pane.Markdown(f"Error reading file: {info['error']}")

        preview_row_count = len(info['preview_df'])

        info_text = f"**Rows:** {info['total_row_count']:,} "
        if preview_row_count < info['total_row_count']:
            info_text += f" -\n_Preview: showing {preview_row_count:,} randomly selected rows of {info['total_row_count']:,}._"
            if len(info['unique_labels']) > 0:
                labels = ", ".join(map(str, info['unique_labels']))
                info_text += f"\n**Labels in preview:** {labels} (may not include all labels in full file)"
        elif 'label' in info['preview_df'].columns:
            label_counts = info['preview_df']['label'].value_counts()
            info_text += "**Label counts:** " + ", ".join([f"{label}: {count}" for label, count in label_counts.items()])
        
        # Keep the preview bounded so large inference files do not overload the browser.
        _ensure_tabulator()
        tabulator_widget = pn.widgets.Tabulator(
            info['preview_df'],
            pagination='local',
            page_size=25,         # Show 25 rows per page
            sizing_mode='stretch_width',
            height=400,
            disabled=True,        # Make it read-only
            show_index=False
        )
        
        return pn.Column(
            pn.pane.Markdown(info_text),
            tabulator_widget,
            sizing_mode='stretch_width'
        )

    def _subset_filename_for_download(output_path: Path, total_rows: int) -> str:
        limit = int(subset_limit.value)
        if limit >= total_rows:
            return output_path.name
        return f"{output_path.stem}.subset.{limit}.csv"

    def _prepare_subset_download():
        if not vm.eng:
            raise ValueError("Initialize first.")

        selected_item = inference_results_selector.value
        if not selected_item:
            raise ValueError("Select an inference result first.")

        original_csv = selected_item.csv_path
        if not original_csv.exists():
            raise FileNotFoundError("Selected inference result no longer exists.")

        total_rows = selected_item.get_num_rows()
        if not download_subset.value or int(subset_limit.value) >= total_rows:
            btn_subset.filename = original_csv.name
            return io.BytesIO(original_csv.read_bytes())

        btn_subset.filename = _subset_filename_for_download(original_csv, total_rows)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "subset.csv"
            make_inference_subset(
                inference_csv=original_csv,
                subset_csv=temp_path,
                limit=int(subset_limit.value),
            )
            return io.BytesIO(temp_path.read_bytes())

    btn_subset.callback = _prepare_subset_download

    def _refresh_inference_selector(select_newest: bool = False):
        """Fetches inference results and populates the selector."""
        if not vm.eng:
            inference_results_selector.options = []
            return

        results = vm.eng.list_inference_results()
        if not results:
            inference_results_selector.options = []
            inference_results_selector.value = None
            return

        # Format options for the dropdown
        options = {item.display_name: item for item in results}
        inference_results_selector.options = options

        if select_newest:
            inference_results_selector.value = results[0]
        elif not inference_results_selector.value or inference_results_selector.value not in options.values():
            if results:
                inference_results_selector.value = results[0]
            else:
                inference_results_selector.value = None

    def _update_results_view(selected_inference_results_item: InferenceResultItem | None):
        """Updates the results preview section based on the selected CSV."""
        if not selected_inference_results_item:
            section2.objects = [pn.pane.Markdown(
                "_No inference result selected._"
            )]
            section3.objects = []
            return

        section2_content = [
            _create_csv_display(selected_inference_results_item)
        ]

        section2_content.append(
            pn.Row(download_subset, subset_limit, btn_subset, sizing_mode="stretch_width")
        )

        subset_limit.visible = download_subset.value

        section2.objects = section2_content

        # Section 3: Upload Detections
        section3.objects = [
            "### 3. Upload Detections",
            pn.pane.Markdown(
                "Upload the detection results to Ecosounds for verification. "
                "If a subset CSV exists for the selected run, it will be uploaded; otherwise the full inference CSV will be used."
            ),
            pn.Row(project_id, btn_upload, sizing_mode="stretch_width"),
        ]

    def _run(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not vm.eng.has_trained_classifier():
            return vm.notify("No classifier available. Train a classifier first.", "error")
        
        try:
            labels_arg = list(target_label_selector.value) or None
            
            vm.notify("Running inference... this may take a while.", "info")
            
            out = vm.eng.run_inference(
                threshold=float(threshold.value),
                labels=labels_arg,
                dataset=vm.eng.config.search_dataset_name,
                subset=float(subset.value),
                exclude_labeled=True,
            )
            vm.notify(f"Inference complete! Saved to: {out}", "success")

            # Refresh the selector, which will show the new result
            _refresh_inference_selector(select_newest=True)
            
        except Exception as e:
            log.exception("Inference run failed")
            vm.notify(f"Inference failed: {e}", "error")

    def _upload(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        
        try:
            selected_item = inference_results_selector.value
            if not selected_item or not selected_item.csv_path.exists():
                return vm.notify("No inference result selected to upload.", "error")

            original_csv = selected_item.csv_path
            subset_csv = original_csv.with_suffix(f'.{subset_limit.value}.csv')
            upload_file = subset_csv if subset_csv.exists() else original_csv
            
            if not upload_file.exists():
                return vm.notify("No CSV file found. Run inference first.", "error")
            
            pid = int(project_id.value) if int(project_id.value) > 0 else None
            
            vm.notify(f"Uploading detections from: {upload_file.name}...", "info")
            
            # Get the success status and verification link
            success_status, verification_link = vm.eng.upload_detections(upload_file, project_id=pid)
            
            if success_status and verification_link:
                vm.notify(f"Upload successful! View at: {verification_link}", "success")
            elif success_status:
                vm.notify("Upload successful, but no verification link available.", "success")
            else:
                vm.notify("Upload failed. Check logs for details.", "error")
            
        except Exception as e:
            log.exception("Inference upload failed")
            vm.notify(f"Upload failed: {e}", "error")

    # Set up event handlers
    btn_run.on_click(_run)
    btn_upload.on_click(_upload)
    
    # Watch for changes in selection to update the preview
    @pn.depends(inference_results_selector.param.value, watch=True)
    def _on_result_selection_change(selected_path):
        _update_results_view(selected_path)

    def _on_download_subset_change(event):
        subset_limit.visible = event.new

    download_subset.param.watch(_on_download_subset_change, "value")

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        """Update heading when the active recognizer changes."""
        inference_heading.object = _build_inference_heading_text()
        _sync_target_label_selector()
        _refresh_inference_selector()

    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _on_config_state_change(_):
        _sync_target_label_selector()
        _refresh_inference_selector()

    # Initial check for file availability
    _sync_target_label_selector()
    _refresh_inference_selector()
    _update_results_view(inference_results_selector.value)

    return pn.Column(
        pn.Row(inference_heading, pn.Spacer(), inference_help_btn),
        pn.pane.Markdown(
            "Run the trained classifier on the database embeddings to generate predictions. "
            "This will create a CSV file with detections above the specified threshold."
        ),
        step1_columns,
        pn.Row(btn_run, sizing_mode="stretch_width"),
        
        pn.layout.Divider(),
        pn.Row(
            pn.pane.Markdown("Preview and download results for:", width=220, margin=(8, 0, 0, 0)),
            inference_results_selector,
            sizing_mode="stretch_width",
        ),

        section2,
        # Upload controls are retained above for future reactivation, but are
        # intentionally not rendered yet.
        sizing_mode="stretch_both",
    )
