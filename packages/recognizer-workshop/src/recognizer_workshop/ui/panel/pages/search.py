import panel as pn
from pathlib import Path
from recognizer_workshop.ui.panel.utils import resolve_query_uri, parse_target_score
from recognizer_workshop.ui.panel.components.native.label_picker import build_active_label_picker

def build(vm):

    print("[search.build] vm.id:", id(vm), "vm.ui.id:", id(vm.ui))
    # regions for query and for results
    selector_box = vm.ui.get_region_box("search.query_selector")
    results_box  = vm.ui.get_region_box("search.results")
    print("[search.build] selector_box.id:", id(selector_box), "results_box.id:", id(results_box))

    # text field to input the query (XC id or URL) and a button to load it
    uri = pn.widgets.TextInput(name="Query (XC id / URL )", placeholder="e.g. XC979796 or https://...")
    btn_load_uri = pn.widgets.Button(name="Load the remote query", button_type="primary")

    # file input widget to upload and save local audio file
    file_input = pn.widgets.FileInput(accept="audio/*", multiple=False)
    # This stash will hold the path to the temporarily saved uploaded file
    upload_stash = {"temp_path": None}

    # inputs for how many results to fetch, target score
    num_results = pn.widgets.IntInput(name="Num results", value=10, start=1)
    target_score = pn.widgets.TextInput(name="Target score (float or 'None')", value="None")
    active_label_picker = build_active_label_picker(vm)

    # button to embed and search for the selected window
    # todo: change text if the duration is < 5s
    btn_search = pn.widgets.Button(name="Select 5s window and search", button_type="primary")

    # we don't want a save button. it should save as soon as the label is picked
    # but for now we will keep it
    btn_save = pn.widgets.Button(name="Save Labels to DB", button_type="warning")

    # --- Input method switcher ---
    input_tabs = pn.Tabs(
        ("Search by ID / URL", pn.Column(uri, btn_load_uri)),
        ("Search by File Upload", pn.Column(file_input))
    )

    # placeholders for the selector and results
    vm.ui.render(
        pn.pane.Alert("Load a remote ID/URL or upload a file to show the selector.", alert_type="info"),
        "search.query_selector",
    )

    vm.ui.render(pn.pane.Markdown("_No results yet._"), "search.results")

    def _handle_upload(event):
        """Callback to save the uploaded file to a temporary location."""
        if file_input.value is None:
            upload_stash["temp_path"] = None
            return

        # We'll save the uploaded file into a sub-directory of the examples folder
        # to ensure it has a stable path for the app to access.
        if not vm.eng:
            vm.notify("Initialize on Config tab first.", "error")
            return
        
        uploads_dir = vm.eng.config.labeled_examples_folder 
        uploads_dir.mkdir(exist_ok=True)
        
        dest_path = uploads_dir / file_input.filename
        
        try:
            with open(dest_path, "wb") as f:
                f.write(file_input.value)
            upload_stash["temp_path"] = dest_path
            vm.notify(f"File '{file_input.filename}' uploaded.", "success")

            # Auto-show selector after upload
            vm.ui.show_query_window_selector(str(dest_path))
        except Exception as e:
            vm.notify(f"Upload failed: {e}", "error")

    # Watch for changes to the file input's value and trigger the save
    file_input.param.watch(_handle_upload, 'value')


    def _load_remote(_):
        if not vm.eng:
            return vm.notify("Initialize on Config tab first.", "error")

        if not uri.value:
            return vm.notify("Please provide an ID, URL, or path.", "warn")
        try:
            vm.ui.show_query_window_selector(uri.value)
            vm.notify("Selector loaded.", "success")
        except Exception as e:
            vm.notify(f"Could not load remote: {e}", "error")

    def _search(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        ts = parse_target_score(target_score.value)
        vm.ui.search_with_selected_window(vm.eng, num_results=int(num_results.value), target_score=ts)

    def _save(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        labels = list(vm.ui.harvest_labels(vm.eng))  # <-- pass eng now
        if not labels:
            return vm.notify("No labels to save.", "warn")
        try:
            before, after, ok, dup = vm.eng.save_labels(labels)
            vm.notify(f"Saved {ok} labels ({dup} duplicates skipped).", "success")
        except Exception as e:
            vm.notify(f"Save failed: {e}", "error")



    btn_load_uri.on_click(_load_remote)
    btn_search.on_click(_search)
    btn_save.on_click(_save)



    return pn.Column(
        "## Search (single example)",
        input_tabs, # tab for choosing between xeno canto and file upload
        selector_box,
        pn.Row(active_label_picker, num_results, target_score, btn_search, btn_save, sizing_mode="stretch_width"),
        results_box,
        sizing_mode="stretch_both",
    )