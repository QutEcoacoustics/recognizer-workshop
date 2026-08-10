from pathlib import Path
import random

import pandas as pd
import panel as pn

from recognizer_workshop.app.agile_modeling_helpers import make_inference_subset

import logging

log = logging.getLogger(__name__)

def build(vm):
    def _ensure_tabulator():
        if getattr(_ensure_tabulator, "_loaded", False):
            return
        pn.extension('tabulator')
        _ensure_tabulator._loaded = True

    # Create help buttons
    inference_help_btn = vm.help_system.create_help_button("help-inference-overview", "Help: Inference")
    batch_inference_help_btn = vm.help_system.create_help_button("help-batch-inference", "Help: Batch Processing")
    result_review_help_btn = vm.help_system.create_help_button("help-result-review", "Help: Result Review")
    export_results_help_btn = vm.help_system.create_help_button("help-export-results", "Help: Export Results")

    out_filename = pn.widgets.TextInput(name="Output filename", value="output.csv")
    threshold    = pn.widgets.FloatInput(name="Threshold", value=0.0, step=0.1)
    subset       = pn.widgets.FloatInput(
        name="Subset proportion (0..1)", value=0.001, start=0.001, end=1.0, step=0.001
    )
    target_label_selector = pn.widgets.CheckBoxGroup(
        name="Labels",
        options=[],
        value=[],
        inline=False,
        sizing_mode="stretch_width",
    )
    btn_run = pn.widgets.Button(
        name="Run inference",
        button_type="primary",
        sizing_mode="fixed",
        width=180,
    )

    step1_left_column = pn.Column(
        pn.Row(out_filename, threshold, subset, sizing_mode="stretch_width"),
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

    # Subset creation controls
    subset_limit = pn.widgets.IntInput(name="Subset limit per label", value=80, start=1, step=10)
    btn_subset   = pn.widgets.Button(name="Create subset CSV", button_type="default")

    # Upload controls
    project_id   = pn.widgets.IntInput(name="Project ID (optional)", value=0, start=0)
    btn_upload   = pn.widgets.Button(name="Upload detections", button_type="success")
    
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


    def _get_csv_info(csv_path):
        """Get information about a CSV file including row count and preview"""
        try:
            preview_limit = 1000
            reservoir = []
            columns = None
            row_count = 0
            rng = random.Random()

            # Reservoir sampling keeps a uniform random sample without retaining
            # the complete inference CSV in memory.
            for chunk in pd.read_csv(csv_path, chunksize=10_000):
                if columns is None:
                    columns = list(chunk.columns)
                for row in chunk.itertuples(index=False, name=None):
                    row_count += 1
                    if len(reservoir) < preview_limit:
                        reservoir.append(row)
                    else:
                        replacement_index = rng.randrange(row_count)
                        if replacement_index < preview_limit:
                            reservoir[replacement_index] = row

            df_preview = pd.DataFrame(reservoir, columns=columns)
            
            # Get unique labels if 'label' column exists
            label_info = ""
            if 'label' in df_preview.columns:
                # For large files, we'll estimate label counts from the preview
                if row_count > 1000:
                    unique_labels = df_preview['label'].unique()
                    label_info = f"**Labels in preview:** {', '.join(unique_labels)} (may not include all labels in full file)"
                else:
                    label_counts = df_preview['label'].value_counts()
                    label_info = "**Label counts:** " + ", ".join([f"{label}: {count}" for label, count in label_counts.items()])
            
            return {
                'row_count': row_count,
                'columns': columns or [],
                'preview_df': df_preview,
                'label_info': label_info
            }
        except Exception as e:
            return {'error': str(e)}

    def _create_csv_display(csv_path, title):
        """Create a display widget for CSV information"""
        info = _get_csv_info(csv_path)
        
        if 'error' in info:
            return pn.pane.Markdown(f"**{title}**\n\nError reading file: {info['error']}")
        
        # Create info text
        info_text = f"""**{title}**

📊 **Rows:** {info['row_count']:,}  
📋 **Columns:** {', '.join(info['columns'])}  
_Preview: showing {len(info['preview_df']):,} randomly selected rows of {info['row_count']:,}._  
{info['label_info']}"""
        
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

    def _check_file_availability():
        """Check if output files exist and update section visibility"""
        if not vm.eng:
            section2.objects = []
            section3.objects = []
            return
            
        output_path = (vm.eng.config.predictions_folder / out_filename.value).absolute()
        subset_path = output_path.with_suffix(f'.{subset_limit.value}.csv')
        
        # Section 2: Show if main inference file exists
        if output_path.exists():
            section2_content = [
                "### 2. Create Subset (Optional)",
                pn.pane.Markdown(
                    "Create a smaller subset of the inference results for verification. "
                    "This uses smart sampling to select representative examples for each label."
                ),
                pn.Row(subset_limit, btn_subset, sizing_mode="stretch_width"),
                _create_csv_display(output_path, "📊 Inference Results")
            ]
            
            # Add subset file info if it exists
            if subset_path.exists():
                section2_content.append(_create_csv_display(subset_path, "📋 Subset Results"))
            
            section2.objects = section2_content
            
            # Section 3: Show if either file exists
            if output_path.exists() or subset_path.exists():
                section3.objects = [
                    "### 3. Upload Detections",
                    pn.pane.Markdown(
                        "Upload the detection results to Ecosounds for verification. "
                        "If a subset CSV exists, it will be uploaded; otherwise the full inference CSV will be used."
                    ),
                    pn.Row(project_id, btn_upload, sizing_mode="stretch_width"),
                ]
            else:
                section3.objects = []
        else:
            section2.objects = []
            section3.objects = []

    def _run(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not vm.eng.has_trained_classifier():
            return vm.notify("No classifier available. Train a classifier first.", "error")
        
        try:
            out = (vm.eng.config.predictions_folder / out_filename.value).absolute()
            
            # Check if file exists and ask for confirmation
            if out.exists() and btn_run.name != "Confirm Overwrite":
                vm.notify(f"File '{out.name}' already exists. Click again to overwrite.", "warn")
                btn_run.name = "Confirm Overwrite"
                btn_run.button_type = "danger"
                return
            
            # Reset button state after confirmation
            btn_run.name = "Run inference"
            btn_run.button_type = "primary"
            
            out.parent.mkdir(parents=True, exist_ok=True)
            labels_arg = list(target_label_selector.value) or None
            
            vm.notify("Running inference... this may take a while.", "info")
            
            vm.eng.run_inference(
                output_filepath=out,
                threshold=float(threshold.value),
                labels=labels_arg,
                dataset=vm.eng.config.search_dataset_name,
                subset=float(subset.value),
                exclude_labeled=True,
            )
            vm.notify(f"Inference complete! Saved to: {out}", "success")
            
            # Update section visibility
            _check_file_availability()
            
        except Exception as e:
            vm.notify(f"Inference failed: {e}", "error")

    def _create_subset(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        
        try:
            original_csv = (vm.eng.config.predictions_folder / out_filename.value).absolute()
            if not original_csv.exists():
                return vm.notify("Run inference first to create the CSV file.", "error")
            
            subset_csv = original_csv.with_suffix(f'.{subset_limit.value}.csv')
            
            vm.notify(f"Creating subset with {subset_limit.value} examples per label...", "info")
            
            make_inference_subset(
                inference_csv=original_csv,
                subset_csv=subset_csv,
                limit=subset_limit.value
            )
            vm.notify("Subset created.", "success")
            
            # Update section visibility
            _check_file_availability()
            
        except Exception as e:
            log.error(f"❌ **Error creating subset:**\n```\n{e}\n```")
            vm.notify(f"Subset creation failed: {e}", "error")

    def _upload(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        
        try:
            # Check if we should upload the subset or the full file
            original_csv = (vm.eng.config.predictions_folder / out_filename.value).absolute()
            subset_csv = original_csv.with_suffix(f'.{subset_limit.value}.csv')
            
            # If subset exists, upload that; otherwise upload the original
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
            vm.notify(f"Upload failed: {e}", "error")

    # Set up event handlers
    btn_run.on_click(_run)
    btn_subset.on_click(_create_subset)
    btn_upload.on_click(_upload)
    
    # Watch for changes in filename to update section visibility
    def _on_filename_change(event):
        _check_file_availability()
        # Reset overwrite confirmation button if filename changes
        if btn_run.name == "Confirm Overwrite":
            btn_run.name = "Run inference"
            btn_run.button_type = "primary"


    
    def _on_subset_limit_change(event):
        _check_file_availability()
    
    out_filename.param.watch(_on_filename_change, 'value')
    subset_limit.param.watch(_on_subset_limit_change, 'value')
    
    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        """Update heading when the active recognizer changes."""
        inference_heading.object = _build_inference_heading_text()
        _sync_target_label_selector()

    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _on_config_state_change(_):
        _sync_target_label_selector()

    # Initial check for file availability
    _sync_target_label_selector()
    _check_file_availability()

    return pn.Column(
        pn.Row(inference_heading, pn.Spacer(), batch_inference_help_btn),
        pn.pane.Markdown(
            "Run the trained classifier on the database embeddings to generate predictions. "
            "This will create a CSV file with detections above the specified threshold."
        ),
        step1_columns,
        pn.Row(btn_run, sizing_mode="stretch_width"),
        
        # Dynamic sections that appear when files exist
        section2,
        # Upload controls are retained above for future reactivation, but the
        # upload section is intentionally disabled for now.
        sizing_mode="stretch_both",
    )
