"""
A reusable component for displaying and editing BAW filters for recordings.

The user supplies comma-separated IDs for one or more of projects, regions, and sites.

"""
import panel as pn

from recognizer_workshop.app.services.baw_insertions import get_last_filter_for_dataset
from recognizer_workshop.app.services.baw_filters import parse_csv_ints, normalize_first_filter


def resolve_filters_for_editor(
    getter_fn,
    vm,
    *,
    dataset_priority: list[str] | None = None,
    fallback_to_legacy: bool = True,
):
    """Resolve filters from page state first, then optional DB metadata fallbacks.

    Priority order:
    1. getter_fn() result, if present
    2. latest filter from baw_insertions for each dataset in dataset_priority order
    3. legacy baw_filters metadata (if fallback_to_legacy=True)
    """
    current = getter_fn()
    if current and current[0]:
        return current

    if not (vm and getattr(vm, "eng", None) and getattr(vm.eng, "db", None)):
        return current

    db = vm.eng.db

    for dataset_name in (dataset_priority or []):
        latest = get_last_filter_for_dataset(db, dataset_name)
        if latest:
            return [latest]

    if fallback_to_legacy:
        try:
            legacy = db.get_metadata('baw_filters')
            legacy_filters = legacy.get('filters', None)
            if legacy_filters and legacy_filters[0]:
                return legacy_filters
        except KeyError:
            pass

    return current

def build_filter_editor(
    getter_fn,
    setter_fn,
    vm,
    *,
    dataset_priority: list[str] | None = None,
    fallback_to_legacy: bool = True,
):
    """
    A reusable component for displaying and editing a single BAW filter,
    where the user selects the filter type from a dropdown.
    Args:
        getter_fn: A function that returns the current filter list (from the ViewModel).
        setter_fn: A function that accepts a new filter list to save it (to the ViewModel).
        vm: The ViewModel instance, used for notifications.
    Returns:
        A Panel Column containing the filter editor UI.
    """
    # --- Mappings for UI display and API keys ---
    FILTER_MAP = {
        'Project IDs': 'projects.id',
        'Region IDs': 'regions.id',
        'Site IDs': 'sites.id'
    }
    # Create a reverse map for looking up the display name from the API key
    KEY_MAP = {v: k for k, v in FILTER_MAP.items()}

    # --- Widgets ---
    filter_type_select = pn.widgets.Select(name="Filter by", options=FILTER_MAP)
    ids_input = pn.widgets.TextInput(name="IDs (comma-separated)", placeholder="e.g., 123, 456")
    
    btn_save = pn.widgets.Button(name="Save Filter", button_type="primary", icon='save')
    btn_cancel = pn.widgets.Button(name="Cancel")
    
    filter_display = pn.pane.Markdown(sizing_mode="stretch_width")
    btn_edit = pn.widgets.Button(name="Edit", icon='edit')
    container = pn.Column(sizing_mode="stretch_width")
    edit_controls = pn.Row(filter_type_select, ids_input, btn_save)

    display_view = pn.Column(
        pn.pane.Markdown("#### Current Recording Filter"),
        pn.Row(filter_display, btn_edit, align="center")
    )
    # The edit view is now much simpler
    edit_view = pn.Column(
        pn.pane.Markdown("#### Edit Recording Filter"),
        edit_controls
    )

    # --- Logic ---
    def _parse_editor_value():
        selected_key = filter_type_select.value
        id_list = parse_csv_ints(ids_input.value)
        if not id_list:
            return None
        return {selected_key: {"in": id_list}}

    def _resolved_filters():
        return resolve_filters_for_editor(
            getter_fn,
            vm,
            dataset_priority=dataset_priority,
            fallback_to_legacy=fallback_to_legacy,
        )

    def _update_save_state(*_):
        current = normalize_first_filter(_resolved_filters())
        candidate = _parse_editor_value()
        btn_save.disabled = (candidate is None) or (candidate == current)

    def _show_edit_mode():
        has_existing = normalize_first_filter(_resolved_filters()) is not None
        edit_controls.objects = (
            [filter_type_select, ids_input, btn_save, btn_cancel]
            if has_existing else
            [filter_type_select, ids_input, btn_save]
        )
        _populate_editors()
        _update_save_state()
        container.objects = [edit_view]

    def _format_filters(filters):
        """Formats the single filter for display."""
        if not filters or not filters[0]:
            return "No filter set."
        
        filter_dict = filters[0]
        # Find the first (and only) key in the dictionary (e.g., 'projects.id')
        key = next(iter(filter_dict), None)
        if not key or 'in' not in filter_dict[key]:
            return "Invalid filter format."

        display_name = KEY_MAP.get(key, key) # Get the user-friendly name
        ids_str = ', '.join(map(str, filter_dict[key]['in']))
        return f"**{display_name}:** `{ids_str}`"

    def _update_display():
        filters = _resolved_filters()
        filter_display.object = _format_filters(filters)
        if filters and filters[0]:
            container.objects = [display_view]
        else:
            _show_edit_mode()

    def _populate_editors():
        """Populates the edit fields based on the current filter."""
        filters = _resolved_filters()
        if filters and filters[0]:
            filter_dict = filters[0]
            key = next(iter(filter_dict), 'projects.id') # Default to projects.id
            
            # Set the dropdown value to the key from the filter
            filter_type_select.value = key
            
            # Set the text input value to the list of IDs
            ids_str = ", ".join(map(str, filter_dict.get(key, {}).get("in", [])))
            ids_input.value = ids_str
        else:
            # Default to a blank project filter
            filter_type_select.value = 'projects.id'
            ids_input.value = ""

    def _save():
        """Constructs and saves the filter based on the dropdown and text input."""
        candidate = _parse_editor_value()
        if candidate is None:
            vm.notify("Please enter at least one numeric ID.", "warn")
            return

        if candidate == normalize_first_filter(_resolved_filters()):
            return
        
        try:
            # The filter structure is still a list containing our one dictionary
            setter_fn([candidate]); vm.notify("Filter saved!", "success"); _update_display()
        except Exception as e:
            vm.notify(f"Could not save filter: {e}", "error")

    # --- Bindings ---
    filter_type_select.param.watch(_update_save_state, 'value')
    ids_input.param.watch(_update_save_state, 'value')
    btn_edit.on_click(lambda _: _show_edit_mode())
    btn_cancel.on_click(lambda _: _update_display())
    btn_save.on_click(lambda _: _save())
    
    _update_display()
    return container