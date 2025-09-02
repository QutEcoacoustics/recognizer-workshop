import panel as pn

def build(vm):
    out_filename = pn.widgets.TextInput(name="Output filename", value="output.csv")
    threshold    = pn.widgets.FloatInput(name="Threshold", value=1.0, step=0.1)
    subset       = pn.widgets.FloatInput(name="Subset proportion (0..1)", value=0.001, step=0.001)
    labels_txt   = pn.widgets.TextInput(name="Labels (comma-sep) or empty for all", value="")
    btn_run      = pn.widgets.Button(name="Run inference", button_type="primary")

    project_id   = pn.widgets.IntInput(name="Project ID (optional)", value=0, start=0)
    btn_upload   = pn.widgets.Button(name="Upload detections", button_type="success")
    log          = pn.pane.Markdown("")

    def _run(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        try:
            out = (vm.eng.config.predictions_folder / out_filename.value).absolute()
            out.parent.mkdir(parents=True, exist_ok=True)
            labels_arg = None
            if labels_txt.value.strip():
                labels_arg = [x.strip() for x in labels_txt.value.split(",") if x.strip()]
            vm.eng.run_inference(
                output_filepath=out,
                threshold=float(threshold.value),
                labels=labels_arg,
                dataset=vm.eng.config.search_dataset_name,
                subset=float(subset.value),
                exclude_labeled=True,
            )
            log.object = f"Saved detections to: {out}"
            vm.notify("Inference complete.", "success")
        except Exception as e:
            log.object = f"Error: {e}"
            vm.notify(f"Inference failed: {e}", "error")

    def _upload(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        try:
            out = (vm.eng.config.predictions_folder / out_filename.value).absolute()
            pid = int(project_id.value) if int(project_id.value) > 0 else None
            vm.eng.upload_detections(out, project_id=pid)
            vm.notify("Upload triggered. Check the printed verification link.", "success")
        except Exception as e:
            vm.notify(f"Upload failed: {e}", "error")

    btn_run.on_click(_run)
    btn_upload.on_click(_upload)

    return pn.Column(
        "## Inference",
        pn.Row(out_filename, threshold, subset, sizing_mode="stretch_width"),
        pn.Row(labels_txt, sizing_mode="stretch_width"),
        pn.Row(btn_run, project_id, btn_upload),
        log,
        sizing_mode="stretch_both",
    )
