import panel as pn
from recognizer_workshop.app.agile_modeling_helpers import labels_from_event_taggings, add_labels_to_db

def build(vm):
    region_id = pn.widgets.IntInput(name="Region ID", value=85)
    btn_fetch = pn.widgets.Button(name="Fetch labels from server", button_type="primary")
    btn_add   = pn.widgets.Button(name="Add fetched labels to DB", button_type="success", disabled=True)
    log       = pn.pane.Markdown("")
    stash     = {"items": None}

    def _fetch(_):
        if not vm.eng: return vm.notify("Initialize on Config tab first.", "error")
        tlabels = vm.eng.get_target_labels() or []
        if not tlabels: return vm.notify("Pick target label(s) on Config tab first.", "warn")
        try:
            log.object = "Fetching…"
            items = labels_from_event_taggings(vm.eng.baw_api, {"region_id":[int(region_id.value)]}, tlabels)
            stash["items"] = items
            btn_add.disabled = False
            total = sum(len(v.get('pos',[]))+len(v.get('neg',[])) for s in items.values() for v in s.values())
            log.object = f"Fetched {total} segments across {len(items)} sources."
            vm.notify("Fetched labels; click 'Add fetched labels to DB' to insert.", "success")
        except Exception as e:
            log.object = f"Error: {e}"
            vm.notify(f"Fetch failed: {e}", "error")

    def _add(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not stash["items"]: return vm.notify("Fetch labels first.", "warn")
        try:
            add_labels_to_db(vm.eng, vm.eng.config, stash["items"])
            vm.notify("Labels added to DB.", "success")
        except Exception as e:
            vm.notify(f"Insert failed: {e}", "error")

    btn_fetch.on_click(_fetch)
    btn_add.on_click(_add)

    return pn.Column(
        "## Labelled examples",
        pn.Row(region_id, btn_fetch, btn_add),
        log,
        sizing_mode="stretch_both",
    )
