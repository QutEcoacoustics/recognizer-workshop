import panel as pn
from pathlib import Path
from recognizer_workshop.ui.panel.utils.utils import parse_target_score
from recognizer_workshop.ui.panel.components.native.label_picker import build_active_label_picker

def build(vm):

    # Create help buttons
    refine_help_btn = vm.help_system.create_help_button("help-refine-overview", "Help: Refine")
    active_learning_help_btn = vm.help_system.create_help_button("help-active-learning", "Help: Active Learning")

    results_box  = vm.ui.get_region_box("refine.results")

    # inputs for how many results to fetch, target score
    num_results = pn.widgets.IntInput(name="Num results", value=10, start=1)
    target_score = pn.widgets.TextInput(name="Target score (float or 'None')", value="None")
    active_label_picker = build_active_label_picker(vm)

    btn_search = pn.widgets.Button(name="Search with classifier", button_type="primary")

    # we don't want a save button. it should save as soon as the label is picked
    # but for now we will keep it
    btn_save = pn.widgets.Button(name="Save Labels to DB", button_type="warning")

    vm.ui.render(pn.pane.Markdown("_No results yet._"), "refine.results")

    def _search(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        ts = parse_target_score(target_score.value)
        vm.ui.search_with_classifier(vm.eng, 
                                     num_results=int(num_results.value), 
                                     target_score=ts)


    def _save(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        labels = list(vm.ui.harvest_labels(vm.eng, results_region="refine.results"))
        if not labels:
            return vm.notify("No labels to save.", "warn")
        try:
            before, after, ok, dup = vm.eng.save_labels(labels)
            vm.notify(f"Saved {ok} labels ({dup} duplicates skipped).", "success")
        except Exception as e:
            vm.notify(f"Save failed: {e}", "error")



    btn_search.on_click(_search)
    btn_save.on_click(_save)


    return pn.Column(
        pn.Row("## Refine Model", pn.Spacer(), refine_help_btn),
        pn.Row("Active Learning Search", pn.Spacer(), active_learning_help_btn),
        pn.Row(active_label_picker, num_results, target_score, btn_search, btn_save, sizing_mode="stretch_width"),
        results_box,
        sizing_mode="stretch_both",
    )