import panel as pn
from recognizer_workshop.ui.panel.utils import resolve_query_uri, parse_target_score

def build(vm):
    uri          = pn.widgets.TextInput(name="Query (XC id / URL / path / index)", value="XC979796")
    btn_window   = pn.widgets.Button(name="Select 5s window", button_type="primary")
    num_results  = pn.widgets.IntInput(name="Num results", value=10, start=1)
    target_score = pn.widgets.TextInput(name="Target score (float or 'None')", value="None")
    btn_search   = pn.widgets.Button(name="Embed & Search", button_type="success")
    btn_save     = pn.widgets.Button(name="Save Labels to DB", button_type="warning")
    view_area    = pn.Column()

    def _mount():
        vm.mount_ui_view(view_area)

    def _select(_):
        if not vm.eng: return vm.notify("Initialize on Config tab first.", "error")
        _mount()
        try:
            resolved = resolve_query_uri(uri.value, vm.eng.config.labeled_examples_folder)
            vm.ui.select_query_window(resolved)
            vm.notify("Query window selected.", "success")
        except Exception as e:
            vm.notify(f"Window selection failed: {e}", "error")

    def _search(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        _mount()
        if vm.ui._last_query_samples is None or len(vm.ui._last_query_samples) == 0:
            return vm.notify("Select a 5s window first.", "warn")
        try:
            vm.eng.embed_query(vm.ui._last_query_samples)
            labels = vm.eng.get_target_labels()
            if not labels: return vm.notify("Pick target label(s) on Config tab.", "warn")
            ts = parse_target_score(target_score.value)
            vm.eng.search_with_query(num_results=int(num_results.value), sample_size=None, target_score=ts)
            lbl = labels[0]
            vm.ui.show_results(vm.eng, vm.eng.search_results[lbl], positive_labels=[lbl])
            vm.notify(f"Search complete for '{lbl}'.", "success")
        except Exception as e:
            vm.notify(f"Search failed: {e}", "error")

    def _save(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        labels = list(vm.ui.harvest_labels())
        if not labels: return vm.notify("No labels to save.", "warn")
        try:
            vm.eng.save_labels(labels)
            vm.notify(f"Saved {len(labels)} labels.", "success")
        except Exception as e:
            vm.notify(f"Save failed: {e}", "error")

    btn_window.on_click(_select)
    btn_search.on_click(_search)
    btn_save.on_click(_save)

    return pn.Column(
        "## Search (single example)",
        pn.Row(uri, btn_window, sizing_mode="stretch_width"),
        pn.Row(num_results, target_score, btn_search, btn_save, sizing_mode="stretch_width"),
        view_area,
        sizing_mode="stretch_both",
    )
