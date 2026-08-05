import panel as pn

from recognizer_workshop.ui.panel.components.native.label_picker import build_active_label_picker


def build(vm):

    # Create help buttons
    refine_help_btn = vm.help_system.create_help_button("help-refine-overview", "Help: Refine")

    results_box  = vm.ui.get_region_box("refine.results")

    # inputs for how many results to fetch, target score
    num_results = pn.widgets.IntInput(name="Num results", value=10, start=1, width=140)
    search_strategy = pn.widgets.Select(
        name="Search strategy",
        options=["Best match", "Margin sampling", "Target score"],
        value="Best match",
        width=180,
    )
    target_score = pn.widgets.TextInput(name="Target score", value="max", width=180, disabled=True)
    active_label_picker = build_active_label_picker(vm)

    btn_search = pn.widgets.Button(name="Search with classifier", button_type="primary")


    vm.ui.render(pn.pane.Markdown("_No results yet._"), "refine.results")

    def _on_strategy_change(event):
        strategy = event.new
        if strategy == "Best match":
            target_score.value = "max"
            target_score.disabled = True
        elif strategy == "Margin sampling":
            target_score.value = "0"
            target_score.disabled = True
        else:
            target_score.value = ""
            target_score.disabled = False

    search_strategy.param.watch(_on_strategy_change, "value")

    def _resolve_target_score():
        strategy = search_strategy.value
        if strategy == "Best match":
            return None
        elif strategy == "Margin sampling":
            return 0
        else:
            raw_score = target_score.value.strip()
            normalized = raw_score.lower()
            if not normalized or normalized == "max":
                return None
            else:
                try:
                    return float(raw_score)
                except (TypeError, ValueError):
                    target_score.value = "0"
                    return 0.0

    def _do_search():
        vm.ui.search_with_classifier(
            vm.eng,
            num_results=int(num_results.value),
            target_score=_resolve_target_score(),
            on_save=_save,
        )

    def _search(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        if vm.ui.has_unsaved_labels(results_region="refine.results"):
            vm.ui.show_unsaved_warning(
                results_region="refine.results",
                on_save=lambda: (_save(), _do_search()),
                on_discard=_do_search,
            )
            return
        _do_search()


    def _save(_=None):
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


    return pn.Column(
        pn.Row("## Refine Model", pn.Spacer(), refine_help_btn),
        pn.Row(active_label_picker, num_results, search_strategy, target_score, btn_search, sizing_mode="stretch_width"),
        results_box,
        sizing_mode="stretch_both",
    )