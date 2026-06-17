"""
A reusable component for displaying and editing BAW filters for recordings.

The user supplies comma-separated IDs for one or more of projects, regions, and sites.

"""
import panel as pn

from recognizer_workshop.app.services.baw_filters import (
    format_filter_params_for_display,
    normalize_first_filter,
    parse_csv_ints,
)


def resolve_filters_for_editor(
    vm,
    *,
    dataset_priority: list[str] | None = None,
    fallback_to_legacy: bool = True,
):
    """Resolve the recording filter for the editor from the database metadata.

    Thin UI-side wrapper around the engine's ``get_baw_filters`` resolver: it
    guards against the engine/DB not being loaded and otherwise delegates.
    Returns ``None`` when no database is available.
    """
    if not (vm and getattr(vm, "eng", None) and getattr(vm.eng, "db", None)):
        return None
    return vm.eng.get_baw_filters(
        dataset_priority=dataset_priority,
        fallback_to_legacy=fallback_to_legacy,
    )

# --- Mappings for UI display and API keys ---
FILTER_MAP = {
    'Project IDs': 'projects.id',
    'Region IDs': 'regions.id',
    'Site IDs': 'sites.id'
}
# Reverse map for looking up the display name from the API key
KEY_MAP = {v: k for k, v in FILTER_MAP.items()}


class FilterEditor:
    """A reusable, self-contained editor for a single BAW recording filter.

    The editor owns its value. Read the current filter via ``.value`` and mount
    ``.view`` in a Panel layout. The dropdown and text input are only a transient
    editing area used while the user is in "Edit" mode; the canonical value lives
    in ``self._value``.

    Args:
        vm: The ViewModel instance, used for notifications.
        dataset_priority: Dataset names to consult (in order) when seeding the
            value from DB metadata.
        fallback_to_legacy: Whether to fall back to legacy ``baw_filters`` metadata
            when seeding.
        on_save: Optional callable invoked with the new filter list (``[filter]``)
            when the user saves. Use this to persist the value (e.g. write it to
            the engine). If it raises, the new value is not retained.
    """

    def __init__(
        self,
        vm,
        *,
        dataset_priority: list[str] | None = None,
        fallback_to_legacy: bool = True,
        on_save=None,
    ):
        self.vm = vm
        self._dataset_priority = dataset_priority
        self._fallback_to_legacy = fallback_to_legacy
        self._on_save = on_save
        self._value: list = []

        # --- Widgets ---
        self._filter_type_select = pn.widgets.Select(name="Filter by", options=FILTER_MAP)
        self._ids_input = pn.widgets.TextInput(name="IDs (comma-separated)", placeholder="e.g., 123, 456")
        self._btn_save = pn.widgets.Button(name="Save Filter", button_type="primary", icon='save')
        self._btn_cancel = pn.widgets.Button(name="Cancel")
        self._filter_display = pn.pane.Markdown(sizing_mode="stretch_width")
        self._btn_edit = pn.widgets.Button(name="Edit", icon='edit')

        self._container = pn.Column(sizing_mode="stretch_width")
        self._edit_controls = pn.Row(self._filter_type_select, self._ids_input, self._btn_save)
        self._display_view = pn.Column(
            pn.pane.Markdown("#### Recording Filter"),
            pn.Row(self._filter_display, self._btn_edit, align="center")
        )
        self._edit_view = pn.Column(
            pn.pane.Markdown("#### Edit Recording Filter"),
            self._edit_controls
        )

        # --- Bindings ---
        self._filter_type_select.param.watch(self._update_save_state, 'value')
        self._ids_input.param.watch(self._update_save_state, 'value')
        self._btn_edit.on_click(lambda _: self._show_edit_mode())
        self._btn_cancel.on_click(lambda _: self._update_display())
        self._btn_save.on_click(lambda _: self._save())

        # Seed the initial value from the DB (if available) and render.
        self.seed_from_db()

    # --- Public API ---
    @property
    def view(self) -> pn.Column:
        """The Panel component to mount in a layout."""
        return self._container

    @property
    def value(self) -> list:
        """The current filter list (``[filter]`` or ``[]``)."""
        return self._value

    def set_value(self, filters):
        """Set the filter value directly and refresh the display."""
        self._value = filters or []
        self._update_display()

    def seed_from_db(self):
        """Populate the value from DB metadata, but only if it is currently empty.

        Safe to call repeatedly (e.g. from a config/engine observer). It never
        clobbers a value the user has already set. Safe to call when no DB is
        loaded (it simply leaves the value untouched).
        """
        if self._value and self._value[0]:
            self._update_display()
            return
        resolved = resolve_filters_for_editor(
            self.vm,
            dataset_priority=self._dataset_priority,
            fallback_to_legacy=self._fallback_to_legacy,
        )
        self._value = resolved or []
        self._update_display()

    # --- Internal logic ---
    def _parse_editor_value(self):
        selected_key = self._filter_type_select.value
        input_text = self._ids_input.value.strip()
        
        # Special case: "*" means fetch all recordings (no filter)
        if input_text == "*":
            return False
        
        id_list = parse_csv_ints(self._ids_input.value)
        if not id_list:
            return None
        return {selected_key: {"in": id_list}}

    def _set_container_view(self, view):
        # Slice assignment tends to be more reliably reactive in Panel callbacks.
        self._container[:] = [view]

    def _update_save_state(self, *_):
        current = normalize_first_filter(self._value) if self._value and self._value[0] is not False else False
        candidate = self._parse_editor_value()
        self._btn_save.disabled = (candidate is None) or (candidate == current)

    def _show_edit_mode(self):
        has_existing = (self._value and self._value[0] is False) or (normalize_first_filter(self._value) is not None)
        self._edit_controls.objects = (
            [self._filter_type_select, self._ids_input, self._btn_save, self._btn_cancel]
            if has_existing else
            [self._filter_type_select, self._ids_input, self._btn_save]
        )
        self._populate_editors()
        self._update_save_state()
        self._set_container_view(self._edit_view)

    def _update_display(self):
        if not self._value or (self._value[0] is not False and not self._value[0]):
            self._filter_display.object = "No filter set."
        elif self._value[0] is False:
            # Special case: False represents wildcard ("*")
            self._filter_display.object = "**Fetch all recordings** (no filter)"
        else:
            self._filter_display.object = format_filter_params_for_display(
                self._value[0],
                fallback_name="Invalid filter format.",
                key_labels=KEY_MAP,
            )
        if self._value and self._value[0] is not False and self._value[0]:
            self._set_container_view(self._display_view)
        elif self._value and self._value[0] is False:
            # False is a valid saved filter state, show display view
            self._set_container_view(self._display_view)
        else:
            self._show_edit_mode()

    def _populate_editors(self):
        """Populates the edit fields based on the current filter."""
        if self._value and self._value[0] is False:
            # Special case: False represents wildcard ("*")
            self._filter_type_select.value = 'projects.id'
            self._ids_input.value = "*"
        elif self._value and self._value[0]:
            filter_dict = self._value[0]
            key = next(iter(filter_dict), 'projects.id')  # Default to projects.id
            self._filter_type_select.value = key
            ids_str = ", ".join(map(str, filter_dict.get(key, {}).get("in", [])))
            self._ids_input.value = ids_str
        else:
            # Default to a blank project filter
            self._filter_type_select.value = 'projects.id'
            self._ids_input.value = ""

    def _save(self):
        """Constructs and saves the filter based on the dropdown and text input."""
        candidate = self._parse_editor_value()
        if candidate is None:
            self.vm.notify("Please enter at least one numeric ID or '*' for all recordings.", "warn")
            return
        # Check if candidate is unchanged from current value (handle False specially)
        current = self._value[0] if self._value else None
        if candidate == current:
            self._update_display()
            return

        try:
            # Persist first (if a hook is provided); only retain the value on success.
            if self._on_save:
                self._on_save([candidate])
            self._value = [candidate]
            self.vm.notify("Filter saved!", "success")
            self._update_display()
        except Exception as e:
            self.vm.notify(f"Could not save filter: {e}", "error")