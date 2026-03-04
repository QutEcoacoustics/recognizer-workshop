from pathlib import Path

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
    subset       = pn.widgets.FloatInput(name="Subset proportion (0..1)", value=0.001, step=0.001)
    labels_txt   = pn.widgets.TextInput(name="Labels (comma-sep) or empty for all", value="")
    btn_run      = pn.widgets.Button(name="Run inference", button_type="primary")

    # Subset creation controls
    subset_limit = pn.widgets.IntInput(name="Subset limit per label", value=80, start=1, step=10)
    btn_subset   = pn.widgets.Button(name="Create subset CSV", button_type="default")

    # Upload controls
    project_id   = pn.widgets.IntInput(name="Project ID (optional)", value=0, start=0)
    btn_upload   = pn.widgets.Button(name="Upload detections", button_type="success")
    
    # Sections that depend on file existence
    section2 = pn.Column()
    section3 = pn.Column()

    def _get_csv_info(csv_path):
        """Get information about a CSV file including row count and preview"""
        try:
            # First, get row count efficiently
            row_count = sum(1 for _ in open(csv_path)) - 1  # -1 for header
            
            # Read a small sample for preview (first 100 rows for performance)
            df_preview = pd.read_csv(csv_path, nrows=100)
            
            # Get unique labels if 'label' column exists
            label_info = ""
            if 'label' in df_preview.columns:
                # For large files, we'll estimate label counts from the preview
                if row_count > 1000:
                    unique_labels = df_preview['label'].unique()
                    label_info = f"**Labels in preview:** {', '.join(unique_labels)} (may not include all labels in full file)"
                else:
                    # For smaller files, read all to get accurate label counts
                    df_full = pd.read_csv(csv_path)
                    label_counts = df_full['label'].value_counts()
                    label_info = "**Label counts:** " + ", ".join([f"{label}: {count}" for label, count in label_counts.items()])
            
            return {
                'row_count': row_count,
                'columns': list(df_preview.columns),
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
{info['label_info']}"""
        
        # Create tabulator widget for preview
        # Use pagination to handle large datasets efficiently
        _ensure_tabulator()
        tabulator_widget = pn.widgets.Tabulator(
            info['preview_df'],
            pagination='remote',  # This enables server-side pagination
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
        if not hasattr(vm.eng, 'classifier') or vm.eng.classifier is None:
            return vm.notify("No classifier available. Train a classifier first.", "error")
        
        try:
            out = (vm.eng.config.predictions_folder / out_filename.value).absolute()
            
            # # Check if file exists and ask for confirmation
            # if out.exists():
            #     # Since Panel doesn't have built-in confirmation dialogs, we'll use a simple approach
            #     # You might want to implement a proper modal dialog in the future
            #     log.info(f"⚠️ **File exists!**\n\nThe file `{out.name}` already exists. Click 'Run inference' again to overwrite it."
            #     # Set a flag to indicate next click should proceed
            #     if not hasattr(_run, 'confirmed'):
            #         _run.confirmed = True
            #         return
            #     else:
            #         # Reset the flag
            #         delattr(_run, 'confirmed')


            
            out.parent.mkdir(parents=True, exist_ok=True)
            labels_arg = None
            if labels_txt.value.strip():
                labels_arg = [x.strip() for x in labels_txt.value.split(",") if x.strip()]
            
            log.info("Running inference... this may take a while.")
            
            vm.eng.run_inference(
                output_filepath=out,
                threshold=float(threshold.value),
                labels=labels_arg,
                dataset=vm.eng.config.search_dataset_name,
                subset=float(subset.value),
                exclude_labeled=True,
            )
            log.info(f"✅ **Inference complete!**\n\nSaved detections to: `{out}`")
            vm.notify("Inference complete.", "success")
            
            # Update section visibility
            _check_file_availability()
            
        except Exception as e:
            log.error(f"❌ **Error during inference:**\n```\n{e}\n```")
            vm.notify(f"Inference failed: {e}", "error")

    def _create_subset(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        
        try:
            original_csv = (vm.eng.config.predictions_folder / out_filename.value).absolute()
            if not original_csv.exists():
                return vm.notify("Run inference first to create the CSV file.", "error")
            
            subset_csv = original_csv.with_suffix(f'.{subset_limit.value}.csv')
            
            log.info(f"Creating subset with {subset_limit.value} examples per label...")
            
            make_inference_subset(
                inference_csv=original_csv,
                subset_csv=subset_csv,
                limit=subset_limit.value
            )
            
            log.info(f"✅ **Subset created!**\n\nSaved subset to: `{subset_csv}`")
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
            
            log.info(f"Uploading detections from: `{upload_file.name}`...")
            
            # Get the success status and verification link
            success_status, verification_link = vm.eng.upload_detections(upload_file, project_id=pid)
            
            if success_status and verification_link:
                log.info(f"✅ **Upload successful!**")
                log.info(f"Uploaded file: `{upload_file.name}`")
                log.info(f"Verification link: {verification_link}")

            elif success_status:
                log.info(f"✅ **Upload successful!**\n\nUploaded file: `{upload_file.name}`\n\nVerification link not available.")
                vm.notify("Upload successful, but no verification link available.", "success")
            else:
                log.info(f"❌ **Upload failed!**\n\nFile: `{upload_file.name}`")
                vm.notify("Upload failed. Check logs for details.", "error")
            
        except Exception as e:
            log.info(f"❌ **Error during upload:**\n```\n{e}\n```")
            vm.notify(f"Upload failed: {e}", "error")

    # Set up event handlers
    btn_run.on_click(_run)
    btn_subset.on_click(_create_subset)
    btn_upload.on_click(_upload)
    
    # Watch for changes in filename to update section visibility
    def _on_filename_change(event):
        _check_file_availability()
    
    def _on_subset_limit_change(event):
        _check_file_availability()
    
    out_filename.param.watch(_on_filename_change, 'value')
    subset_limit.param.watch(_on_subset_limit_change, 'value')
    
    # Initial check for file availability
    _check_file_availability()

    return pn.Column(
        #pn.Row("## Inference", pn.Spacer(), inference_help_btn),
        pn.Row("## Run Inference", pn.Spacer(), batch_inference_help_btn),
        pn.pane.Markdown(
            "Run the trained classifier on the database embeddings to generate predictions. "
            "This will create a CSV file with detections above the specified threshold."
        ),
        pn.Row(out_filename, threshold, subset, sizing_mode="stretch_width"),
        pn.Row(labels_txt, sizing_mode="stretch_width"),
        pn.Row(btn_run, sizing_mode="stretch_width"),
        
        # # Dynamic sections that appear when files exist
        # pn.Row("### 2. Create Subset (Optional)", pn.Spacer(), result_review_help_btn) if section2.objects else None,
        # section2,
        # pn.Row("### 3. Upload Detections", pn.Spacer(), export_results_help_btn) if section3.objects else None,
        # section3,
        
        sizing_mode="stretch_both",
    )
