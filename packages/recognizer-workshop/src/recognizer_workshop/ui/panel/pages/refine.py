import panel as pn
from recognizer_workshop.ui.panel.utils import parse_target_score

def build(vm):
    num_results  = pn.widgets.IntInput(name="Num results", value=10, start=1)
    target_score = pn.widgets.TextInput(name="Target score (float or 'None')", value="None")
    btn_search   = pn.widgets.Button(name="Search with classifier", button_type="success")
    btn_save     = pn.widgets.Button(name="Save Labels to DB", button_type="warning")
    view_area    = pn.Column()

    def _mount(): vm.mount_ui_view(view_area)

    def _run(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not getattr(vm.eng, "classifier", None):
            return vm.notify("Train a classifier on the Train tab first.", "warn")
        _mount()
        try:
            ts = parse_target_score(target_score.value)
            vm.ui.clear_results()
            vm.eng.search_with_classifier(num_results=int(num_results.value), sample_size=None, target_score=ts)
            for lbl in vm.eng.classifier.classes:
                vm.ui.show_results(vm.eng, vm.eng.search_results[lbl], positive_labels=[lbl])
            vm.notify("Classifier search complete. Review and label.", "success")
        except Exception as e:
            vm.notify(f"Refine search failed: {e}", "error")

    def _save(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        labels = list(vm.ui.harvest_labels())
        if not labels: return vm.notify("No labels to save.", "warn")
        try:
            vm.eng.save_labels(labels)
            vm.notify(f"Saved {len(labels)} labels.", "success")
        except Exception as e:
            vm.notify(f"Save failed: {e}", "error")

    btn_search.on_click(_run)
    btn_save.on_click(_save)

    return pn.Column(
        "## Refine",
        pn.Row(num_results, target_score, btn_search, btn_save, sizing_mode="stretch_width"),
        view_area,
        sizing_mode="stretch_both",
    )
