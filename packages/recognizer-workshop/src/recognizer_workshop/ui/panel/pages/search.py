import asyncio
from pathlib import Path
import logging

import panel as pn

from recognizer_workshop.app.services.search import resolve_query_uri
from recognizer_workshop.ui.panel.utils.utils import parse_target_score
from recognizer_workshop.ui.panel.components.native.label_picker import build_active_label_picker

log = logging.getLogger(__name__)

def build(vm):

    # Create help button
    search_help_btn = vm.help_system.create_help_button("help-search-overview", "Help: Search")

    log.debug("vm.id: %s, vm.ui.id: %s", id(vm), id(vm.ui))
    # regions for query and for results
    selector_box = vm.ui.get_region_box("search.query_selector")
    results_box  = vm.ui.get_region_box("search.results")
    log.debug("selector_box.id: %s, results_box.id: %s", id(selector_box), id(results_box))

    # text field to input the query (XC id or URL) and a button to load it
    uri = pn.widgets.TextInput(name="Query (XC id / URL )", placeholder="e.g. XC979796 or https://...")
    btn_load_uri = pn.widgets.Button(name="Load the remote query", button_type="primary", sizing_mode="fixed")

    # file input widget to upload and save local audio file
    file_input = pn.widgets.FileInput(accept="audio/*", multiple=False)
    # This stash will hold the path to the temporarily saved uploaded file
    upload_stash = {"temp_path": None}

    # inputs for how many results to fetch, target score
    num_results = pn.widgets.IntInput(name="Num results", value=10, start=1)
    active_label_picker = build_active_label_picker(vm)

    # button to embed and search for the selected window
    # todo: change text if the duration is < 5s
    btn_search = pn.widgets.Button(name="Select 5s window and search", button_type="primary")


    # --- Input method switcher ---
    input_tabs = pn.Tabs(
        ("Search by ID / URL", pn.Row(uri, btn_load_uri, align="end", sizing_mode="stretch_width")),
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
            # Normalize the query URI before passing it to the selector
            resolved_uri = resolve_query_uri(uri.value, vm.eng.config.labeled_examples_folder)
            vm.ui.show_query_window_selector(resolved_uri)
            vm.notify("Selector loaded.", "success")
        except Exception as e:
            vm.notify(f"Could not load remote: {e}", "error")

    async def _do_search():
        ts = None
        await vm.ui.search_with_selected_window(
            vm.eng,
            num_results=int(num_results.value),
            target_score=ts,
            on_save=_save,
        )

    async def _search(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        if vm.ui.has_unsaved_labels(results_region="search.results"):
            vm.ui.show_unsaved_warning(
                results_region="search.results",
                on_save=lambda: (_save(), asyncio.ensure_future(_do_search())),
                on_discard=lambda: asyncio.ensure_future(_do_search()),
            )
            return
        await _do_search()

    def _save(_=None):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        labels = list(vm.ui.harvest_labels(vm.eng, results_region="search.results"))
        if not labels:
            return vm.notify("No labels to save.", "warn")
        try:
            before, after, ok, dup = vm.eng.save_labels(labels)
            vm.notify(f"Saved {ok} labels ({dup} duplicates skipped).", "success")
        except Exception as e:
            vm.notify(f"Save failed: {e}", "error")

    btn_load_uri.on_click(_load_remote)
    btn_search.on_click(_search)





    return pn.Column(
        pn.Row("## Search (single example)", pn.Spacer(), search_help_btn),
        input_tabs, # tab for choosing between xeno canto and file upload
        selector_box,
        pn.Row(active_label_picker, num_results, btn_search, sizing_mode="stretch_width"),
        results_box,
        sizing_mode="stretch_both",
    )