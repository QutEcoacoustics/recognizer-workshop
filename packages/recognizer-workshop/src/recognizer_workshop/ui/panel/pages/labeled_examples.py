import panel as pn
import json

from recognizer_workshop.app.services.db_labels.db_labels import add_labels_to_db

def build(vm):

    btn_fetch = pn.widgets.Button(name="Fetch labels from server", button_type="primary")
    btn_add   = pn.widgets.Button(name="Add fetched labels to DB", button_type="success", disabled=True)
    log       = pn.pane.Markdown("")
    stash     = {"result_sets": None}


    filter_display = pn.pane.JSON(depth=-1, name="Current BAW Filters", sizing_mode="stretch_width")
    
    project_ids_input = pn.widgets.TextInput(name="Project IDs (comma-separated)", placeholder="e.g., 123, 456")
    region_ids_input = pn.widgets.TextInput(name="Region IDs (comma-separated)", placeholder="e.g., 789")
    site_ids_input = pn.widgets.TextInput(name="Site IDs (comma-separated)", placeholder="e.g., 101, 102")

    btn_edit = pn.widgets.Button(name="Edit", icon='edit')
    btn_save = pn.widgets.Button(name="Save Filters", button_type="primary", icon='save')
    btn_cancel = pn.widgets.Button(name="Cancel")


    display_view = pn.Row(filter_display, btn_edit, align="start")
    edit_view = pn.Column(
        project_ids_input,
        region_ids_input,
        site_ids_input,
        pn.Row(btn_save, btn_cancel)
    )


    def _update_filter_display():
        """Helper to fetch filters and update the UI state."""
        if not (vm.eng and vm.eng.db):
            display_view.visible = False
            edit_view.visible = False
            return

        filters = vm.eng.get_baw_filters()
        if filters:
            filter_display.object = filters
            display_view.visible = True
            edit_view.visible = False
        else:
            display_view.visible = False
            edit_view.visible = True
            # Clear fields if no filter exists
            project_ids_input.value = ""
            region_ids_input.value = ""
            site_ids_input.value = ""


    def _populate_editors_from_filters():
        """Helper to parse existing filters and fill the text inputs."""
        filters = vm.eng.get_baw_filters()
        # Reset fields before populating
        project_ids_input.value, region_ids_input.value, site_ids_input.value = "", "", ""

        if not filters or not isinstance(filters, list) or not filters[0]:
            return
            
        filter_dict = filters[0] # We assume the first filter in the list holds the IDs

        def get_ids(key):
            if key in filter_dict and "in" in filter_dict[key]:
                return ", ".join(map(str, filter_dict[key]["in"]))
            return ""

        project_ids_input.value = get_ids("projects.id")
        region_ids_input.value = get_ids("regions.id")
        site_ids_input.value = get_ids("sites.id")


    def _toggle_to_edit_view(_):
        """Callback for the 'Edit' button."""
        _populate_editors_from_filters()
        display_view.visible = False
        edit_view.visible = True


    def _save_filters(_):
        """Callback for 'Save'. Parses the 3 fields and generates the filter."""
        if not vm.eng: return vm.notify("Initialize on Config tab first.", "error")

        inputs = {
            "projects.id": project_ids_input.value,
            "regions.id": region_ids_input.value,
            "sites.id": site_ids_input.value
        }

        # Validate that at least one field is filled
        if not any(val.strip() for val in inputs.values()):
            vm.notify("Please enter IDs in at least one field.", "warn")
            return

        new_filter = {}
        try:
            for key, value_str in inputs.items():
                if not value_str.strip():
                    continue
                
                # Parse comma-separated string into a list of integers
                id_list = [int(s.strip()) for s in value_str.split(',') if s.strip()]
                
                if id_list:
                    new_filter[key] = {"in": id_list}
        
        except ValueError:
            vm.notify("Invalid input. Please ensure all IDs are comma-separated numbers.", "error")
            return
        
        # The filter is a list containing one dictionary
        final_filters = [new_filter]

        try:
            vm.eng.set_baw_filters(final_filters)
            vm.notify("BAW filters saved successfully!", "success")
            _update_filter_display()
        except Exception as e:
            vm.notify(f"Could not save filters: {e}", "error")


    def _fetch(_):
        if not vm.eng: return vm.notify("Initialize on Config tab first.", "error")
        try:
            result = vm.eng.fetch_labels()
            stash["result_sets"] = result
            log.object = f"Successfully fetched **{sum(len(rs['labels']) for rs in result)}** labels across {len(result)} result sets."
            btn_add.disabled = False
            vm.notify("Fetch successful. Ready to add to DB.", "success")
        except Exception as e:
            log.object = f"Error: {e}"
            vm.notify(f"Fetch failed: {e}", "error")

    def _add(_):
        if not vm.eng: return vm.notify("Initialize first.", "error")
        if not stash["result_sets"]: return vm.notify("Fetch labels first.", "warn")

        for result_set in stash["result_sets"]:
            try:
                add_labels_to_db(vm.eng.db, vm.eng.config, result_set["labels"])
                vm.notify("Labels added to DB.", "success")
            except Exception as e:
                vm.notify(f"Insert failed: {e}", "error")


    btn_edit.on_click(_toggle_to_edit_view)
    btn_save.on_click(_save_filters)
    btn_cancel.on_click(lambda _: _update_filter_display())
    btn_fetch.on_click(_fetch)
    btn_add.on_click(_add)

    _update_filter_display()

    return pn.Column(
        "## Labelled examples",
        pn.Card(
            display_view,
            edit_view,
            title=" Baw API Filters",
            collapsible=True,
            sizing_mode="stretch_width"
        ),
        pn.layout.Divider(),
        pn.Row(btn_fetch, btn_add),
        log,
        sizing_mode="stretch_both",
    )