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

    btn_search = pn.widgets.Button(name="Search with recognizer", button_type="primary")

    def _get_active_saved_classifier_entry():
        if not vm.eng:
            return None
        index = getattr(vm.eng, "training_result_index", None)
        return index.get_active_classifier_entry() if index else None

    def _build_refine_heading_text() -> str:
        active_entry = _get_active_saved_classifier_entry()
        if not active_entry:
            return "## Refine Recognizer"
        return f"## Refine Recognizer - {active_entry.classifier_name} ({active_entry.display_name})"

    refine_heading = pn.pane.Markdown(_build_refine_heading_text())


    vm.ui.render(pn.pane.Markdown("_No results yet._"), "refine.results")

    refine_page = pn.Column(sizing_mode="stretch_both")

    def _refresh_refine_page():
        active_entry = _get_active_saved_classifier_entry()
        if not active_entry:
            refine_page.objects = [pn.pane.Markdown("Train a recognizer first before refining it")]
            return

        refine_heading.object = _build_refine_heading_text()
        refine_page.objects = [
            pn.Row(refine_heading, pn.Spacer(), refine_help_btn),
            pn.Row(active_label_picker, num_results, search_strategy, target_score, btn_search, sizing_mode="stretch_width"),
            results_box,
        ]

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _on_engine_state_change(_):
        _refresh_refine_page()

    @pn.depends(vm.obs.ensure("config_state").param.version, watch=True)
    def _on_config_state_change(_):
        _refresh_refine_page()

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
        )

    def _search(_):
        if not vm.eng:
            return vm.notify("Initialize first.", "error")
        _do_search()


    btn_search.on_click(_search)
    _refresh_refine_page()

    return refine_page