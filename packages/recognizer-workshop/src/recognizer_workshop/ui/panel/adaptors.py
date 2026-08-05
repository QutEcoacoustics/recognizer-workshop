from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, Callable
import logging

import numpy as np
import panel as pn
import param

from perch_hoplite.db import interface as hoplite_interface

from recognizer_workshop.app.agile_engine import AgileConfig, agile_state, LabelName
from recognizer_workshop.perch_hoplite_custom_modifications.embedding_display import QueryDisplay
from recognizer_workshop.ui.panel.components.native.embedding_display import PanelQuerySelector, PanelEmbeddingDisplay, PanelEmbeddingDisplayGroup

log = logging.getLogger(__name__)

@dataclass
class Region():
    """
    Holds state for a named region.
    A region is an area of the UI for a specific purpose, e.g. the query selector, search results etc
    Each region has a box - a pn.Column, which is where the UI components are rendered.
    and a state dict for any additional state (selected samples, etc)
    
    """
    name: str
    box: pn.Column = field(
        default_factory=lambda: pn.Column(sizing_mode="stretch_both")
    )
    state: dict[str, Any] = field(default_factory=dict)


class PanelUI(param.Parameterized):
    """Panel-based UI that adheres to recognizer_workshop.ui.contracts.AgileUI.

    Notes:
      - This adapter is stateful: it keeps result selections per active label.
      - `select_query_window` shows a small panel to pick the 5s offset and
        stores the chosen samples in `self._last_query_samples`.
      - `show_results` renders radio controls to pick pos/neg/skip per hit.
      - `harvest_labels` returns hoplite_interface.Label objects for the last
        active label’s selections.
    """


    # dictionary of pn columns where all the different elements are rendered. 
    _region_names = [
        "config.main", 
        "config.label_picker", 
        "search.query_selector", 
        "search.results",
        "refine.results",
        "status.main",
        "explore.selected_embedding",
    ]


    def __init__(self, vm):

        super().__init__()
        # initialise the state to have a dict of state for each region, with a 'region' property that contains the pn Column. 
        self._regions = { 
            region_name: Region(name=region_name) for region_name in self._region_names
            }
        
        self.vm = vm
        
        # list of callbacks to call when the engine is ready
        # TODO: probably change this to watch for for obvervable bumps
        # this seems unreliable if eng is updated or recreated
        self._on_engine_ready = []

    def on_engine_ready(self, fn):
        """Register a callback to be called when the engine is ready."""
        self._on_engine_ready.append(fn)

    def engine_attached(self, eng):
        # call after vm.eng is created/initialized
        for fn in list(self._on_engine_ready):
            try: fn()
            except Exception: pass

    def get_region_box(self, name: str) -> pn.Column:
        """
        Get or create a named region container.
        Pages use this to create named regions, and insert them into the layout
        and functions from this class can then populate them and update them.

        Region names are not arbitrary: the pages should name things correctly becasue
        the names are referred to in interactions between functions in this UI adaptor.       
        """

        try:
            return self._regions[name].box
        except KeyError as e:
            valid = ", ".join(self._region_names)
            raise KeyError(f"Unknown region '{name}'. Valid regions: {valid}") from e


    def render(self, view: pn.viewable.Viewable, region_name: str) -> None:
        box = self.get_region_box(region_name)
        box[:] = [view]


    def clear_region_box(self, name: str):
        """Empty a region's children but keep the container."""
        self.get_region_box(name).clear()


    def view_target_labels(self) -> pn.layout.Panel:
        """Builds and returns the UI component for defining target labels."""
        labels_container = pn.Column(sizing_mode="stretch_width")

        btn_add_label = pn.widgets.Button(
            name="Add Label",
            icon="plus",
            button_type="success",
            sizing_mode="fixed",
        )
        current_labels_display = pn.pane.Markdown()

        label_editor_section = pn.Column(
            labels_container,
            btn_add_label,
            sizing_mode="stretch_width"
        )
        is_rebuilding_labels = False
        last_rendered_signature = None

        def _current_config():
            return self.vm.eng.config if self.vm.eng else self.vm.config

        def _describe_target_labels(config_obj) -> str:
            labels = getattr(config_obj, "target_labels", []) or []
            if not labels:
                return ""

            parts = []
            for label in labels:
                if getattr(label, "call_type", None):
                    parts.append(f"{label.species} ({label.call_type})")
                else:
                    parts.append(label.species)
            return ", ".join(parts)

        def _labels_signature(config_obj) -> tuple[str, ...]:
            labels = getattr(config_obj, "target_labels", []) or []
            return tuple(str(label) for label in labels)

        def sync_and_update_display(*events, save: bool = True):
            config_obj = _current_config()

            if save and getattr(config_obj, "config_path", None):
                config_obj.save()

            description = _describe_target_labels(config_obj)
            current_labels_display.object = f"**Summary:** {description}" if description else "*(No labels defined)*"

        def _notify_target_labels_changed() -> None:
            # Keep active-label widgets in sync with config edits.
            self.vm.reseat_active_label()
            self.vm.obs.bump("target_labels")
            self.vm.obs.bump("config_state")


        def _remove_label(index, event):  
            config_obj = _current_config()
            if 0 <= index < len(config_obj.target_labels):
                config_obj.target_labels.pop(index)
                _notify_target_labels_changed()
                _build_labels_ui()
                sync_and_update_display()


        def _build_labels_ui():
            nonlocal is_rebuilding_labels, last_rendered_signature
            is_rebuilding_labels = True
            labels_container.clear()

            config_obj = _current_config()
            species_options = self.vm.eng.workbench_tags if self.vm.eng else []

            for i, label in enumerate(config_obj.target_labels):
                row_species_options = list(species_options)
                if label.species and label.species not in row_species_options:
                    row_species_options.append(label.species)

                species_widget = pn.widgets.AutocompleteInput(
                    name=f"Species {i+1}", 
                    options=row_species_options,
                    value=label.species,
                    placeholder="Type to search for a species...",
                    case_sensitive=False,
                    min_characters=1,
                    sizing_mode="stretch_width",
                )
                call_types_widget = pn.widgets.TextInput(
                    name="Call Type", 
                    value=str(label.call_type) if label.call_type else "", 
                    placeholder="(optional) e.g., song",
                    sizing_mode="stretch_width",
                )
                remove_btn = pn.widgets.Button(
                    name="Remove",
                    icon="trash",
                    button_type="danger",
                    sizing_mode="fixed",
                )
                remove_btn.on_click(partial(_remove_label, i))

                def _update_species(event, index=i):
                    if is_rebuilding_labels:
                        return
                    # Replace the LabelName instance instead of mutating it
                    current_config = _current_config()
                    old_label = current_config.target_labels[index]
                    current_config.target_labels[index] = LabelName(
                        species=(event.new or "").strip(),
                        call_type=old_label.call_type
                    )
                    _notify_target_labels_changed()
                    sync_and_update_display()

                def _update_call_type(event, index=i):
                    if is_rebuilding_labels:
                        return
                    # Replace the LabelName instance instead of mutating it
                    current_config = _current_config()
                    old_label = current_config.target_labels[index]
                    current_config.target_labels[index] = LabelName(
                        species=old_label.species,
                        call_type=event.new.strip() or None
                    )
                    _notify_target_labels_changed()
                    sync_and_update_display()

                species_widget.param.watch(_update_species, 'value')
                call_types_widget.param.watch(_update_call_type, 'value')

                labels_container.append(pn.Row(species_widget, call_types_widget, 
                                               remove_btn, sizing_mode="stretch_width"))

            last_rendered_signature = _labels_signature(config_obj)
            is_rebuilding_labels = False

        def _refresh_species_options_in_place():
            config_obj = _current_config()
            species_options = self.vm.eng.workbench_tags if self.vm.eng else []
            rows = getattr(labels_container, "objects", [])
            if not isinstance(rows, list):
                return

            for row, label in zip(rows, config_obj.target_labels):
                row_objects = getattr(row, "objects", None)
                if not isinstance(row_objects, list) or not row_objects:
                    continue

                species_widget = row_objects[0]
                if not isinstance(species_widget, pn.widgets.AutocompleteInput):
                    continue

                row_species_options = list(species_options)
                if label.species and label.species not in row_species_options:
                    row_species_options.append(label.species)
                species_widget.options = row_species_options

        def _add_label(event):
            config_obj = _current_config()
            config_obj.target_labels.append(LabelName(species="", call_type=None))
            _notify_target_labels_changed()
            _build_labels_ui()
            sync_and_update_display()

        btn_add_label.on_click(_add_label)



        def _init():
            label_editor_section.visible = True
            _build_labels_ui()
            sync_and_update_display(save=False)


        # Initial state
        _init()

        @pn.depends(self.vm.obs.ensure("config_state").param.version, watch=True)
        def _on_config_change(_):
            config_obj = _current_config()
            if _labels_signature(config_obj) != last_rendered_signature:
                _build_labels_ui()
            sync_and_update_display(save=False)

        @pn.depends(self.vm.obs.ensure("target_labels").param.version, watch=True)
        def _on_target_labels_change(_):
            config_obj = _current_config()
            if _labels_signature(config_obj) != last_rendered_signature:
                _build_labels_ui()
            else:
                _refresh_species_options_in_place()
            sync_and_update_display(save=False)


        return pn.Column(
            current_labels_display,
            label_editor_section,
            sizing_mode="stretch_width"
        )

    # ------------------------------------------------------------------
    # Protocol: label selection (updates eng.label)
    # ------------------------------------------------------------------
    def pick_label(self, eng, initial_species: str | None = None,
                   initial_call_types: list[str] | None = None) -> None:
        """Panel version of the label picker; writes into eng.label in-place."""
        initial_species = initial_species or ""
        initial_call_types = initial_call_types or []

        # Load species list from BAW (taxonomic tags)
        try:
            tags = eng.active_client.baw_filter(
                '/tags', {'is_taxonomic': {'eq': True}}, qsp="disable_paging=true"
            ).get('data', [])
            species_options = sorted([t['text'] for t in tags])
        except Exception as e:
            species_options = []
            self.notify(f"Could not fetch species tags from BAW: {e}", "warn")

        species = pn.widgets.AutocompleteInput(
            name="Species Label",
            value=initial_species,
            options=species_options,
            placeholder="Start typing species name...",
            case_sensitive=False,
            min_characters=1,
            sizing_mode="stretch_width",
        )

        call_types = pn.widgets.TextInput(
            name="Call Types (comma-separated)",
            value=", ".join(initial_call_types) if initial_call_types else "",
            placeholder="e.g. call, song",
            sizing_mode="stretch_width",
        )

        btn = pn.widgets.Button(name="Submit", button_type="success")

        def _submit(_):
            eng.label['species'] = species.value
            typed = [s.strip() for s in call_types.value.split(",") if s.strip()]
            eng.label['call_types'] = [t.lower().capitalize() for t in typed]
            self.notify(f"Label bound → species={eng.label['species']}, "
                        f"call_types={eng.label['call_types']}", "info")

        btn.on_click(_submit)

        card = pn.Card(
            "# Select target label",
            species,
            call_types,
            btn,
            collapsible=False,
            sizing_mode="stretch_width",
        )

        self._main_col[:] = [self._status, card]


    def show_query_window_selector(self, uri: str, window_size_s: float = 5.0, region_name="search.query_selector"):
        self.render(pn.Card("⏳ Loading selector..."), region_name)
        region = self._regions[region_name]
        selector = PanelQuerySelector(uri=uri, window_size_s=window_size_s)
        region.state["selector"] = selector
        self.render(selector, region_name)
    

    async def search_with_selected_window(
        self,
        eng,
        *,
        num_results: int = 10,
        target_score=None,
        on_save: Optional[Callable[[], None]] = None,
        selector_region: str = "search.query_selector",
        results_region: str = "search.results",
    ):
        """Read samples from the selector, embed, run search, then render results."""
        region = self._regions[selector_region]
        selector = region.state.get("selector")
        if selector is None:
            self.render(pn.pane.Alert("No selector loaded.", alert_type="warning"), results_region)
            return

        # get the 5s window from the selector's QueryDisplay
        audio_window = selector.qd.get_audio_window()
        if audio_window is None or len(audio_window) == 0:
            self.render(pn.pane.Alert("Select a 5s window first.", alert_type="warning"), results_region)
            return

        # embed & search
        await eng.embed_query_async(audio_window)
        active_label = eng.active_label
        if not active_label:
            self.render(pn.pane.Alert("Pick target label(s) on the Config tab.", alert_type="warning"), results_region)
            log.warning('No active label selected for search.')
            return
        else:
            self.clear_region_box(results_region)
        

        log.info(f'Starting search for label {active_label}...')
        eng.search_with_query(num_results=num_results, sample_size=None, target_score=target_score, target_label=active_label)
        log.info(f'Search completed for label {active_label}, showing results.')
        self.show_results(eng, active_label, results_region=results_region, on_save=on_save)
        log.info(f'Results rendered for label {active_label}.')


    def search_with_classifier(
        self,
        eng,
        *,
        num_results: int = 10,
        target_score=None,
        on_save: Optional[Callable[[], None]] = None,
        results_region: str = "refine.results",
    ):
        """Read samples from the selector, embed, run search, then render results."""

        eng.search_with_classifier(num_results=num_results, sample_size=None, target_score=target_score)

        active_label = eng.active_label
        if not active_label:
            self.render(pn.pane.Alert("Pick target label(s) on the Config tab.", alert_type="warning"), results_region)
            return
        self.show_results(
            eng,
            active_label,
            results_region=results_region,
            on_save=on_save,
        )



    def show_results(
        self,
        eng,
        label: str,
        *,
        results_region: str,
        page_size: int = 10,
        on_save: Optional[Callable[[], None]] = None,
    ):
        """Render a pageable group for a label’s TopK results."""
        label_str = str(label)
        results = eng.search_results.get(label_str)
        if not results:
            self.render(pn.pane.Alert(f"No results for '{label_str}'.", alert_type="info"), results_region)
            return

        # TODO: support search and verification of multiple labels
        

        group = PanelEmbeddingDisplayGroup.from_search_results(
            results,
            db=eng.db,
            labels=[label_str],
            page_size=page_size,
            on_save=on_save,
            sample_rate_hz=eng.db_model_config.model_config.sample_rate,
            frame_rate=100,
            window_size_s=5.0,
            audio_loader=eng.audio_filepath_loader,
        )
        self._regions[results_region].state["group"] = group
        self.render(group, results_region)


    def show_selected_embedding(self, eng, embedding_id, region_name="explore.selected_embedding"):
        """Creates and renders a PanelEmbeddingDisplay for a given embedding_id."""
        self.render(pn.pane.Markdown(f"Loading embedding `{embedding_id}`..."), region_name)
        
        try:
            source = eng.db.get_embedding_source(embedding_id)
            
            display_widget = PanelEmbeddingDisplay(
                embedding_id=embedding_id,
                dataset_name=source.dataset_name,
                uri=source.source_id,
                offset_s=source.offsets[0],
                score=0.0, # No score in this context
                labels=eng.get_target_labels(), # Or however you get your full label list
                sample_rate_hz=eng.db_model_config.model_config.sample_rate,
                audio_loader=eng.audio_filepath_loader,
            )
            self.render(display_widget, region_name)
        except Exception as e:
            self.render(pn.pane.Alert(f"Could not load embedding `{embedding_id}`: {e}", alert_type="danger"), region_name)



    def has_unsaved_labels(self, *, results_region: str) -> bool:
        """Return True if the current results group has any unsaved label states."""
        group = self._regions[results_region].state.get("group")
        return group is not None and group.has_unsaved_labels()

    def show_unsaved_warning(self, *, results_region: str, on_save, on_discard) -> None:
        """Prepend an inline warning banner to the results region.

        The banner is inserted above the existing results (which remain visible).
        Save and Discard remove the banner then execute their action; Cancel just
        removes the banner, leaving the results untouched.
        """
        box = self.get_region_box(results_region)
        btn_save = pn.widgets.Button(name="Save before searching", button_type="warning", sizing_mode="fixed")
        btn_discard = pn.widgets.Button(name="Discard and search", button_type="danger", sizing_mode="fixed")
        btn_cancel = pn.widgets.Button(name="Cancel", button_type="default", sizing_mode="fixed")
        banner = pn.Column(
            pn.pane.Alert(
                "⚠ You have unsaved verifications. Save them first, discard and search, or cancel.",
                alert_type="warning",
            ),
            pn.Row(btn_save, btn_discard, btn_cancel),
        )
        def _remove():
            box.remove(banner)
        btn_save.on_click(lambda _: (_remove(), on_save()))
        btn_discard.on_click(lambda _: (_remove(), on_discard()))
        btn_cancel.on_click(lambda _: _remove())
        box.insert(0, banner)

    def harvest_labels(self, eng, *, results_region: str):
        """Collect user judgements from the current results page(s)."""
        group = self._regions[results_region].state.get("group")
        if group is None:
            return []
        provenance = eng.config.annotator_id
        return group.harvest_labels(provenance)


    # # ------------------------------------------------------------------
    # # Protocol: plot_scores_histogram(eng, label)
    # # ------------------------------------------------------------------
    # def plot_scores_histogram(self, eng, label: str) -> None:
    #     scores = None
    #     if getattr(eng, "search_scores", None) and label in eng.search_scores:
    #         scores = eng.search_scores[label]
    #     if scores is None:
    #         self.notify(f"No stored scores for label '{label}'. Run search first.", "warn")
    #         return

    #     fig = plt.figure(figsize=(6, 3))
    #     _ = plt.hist(scores, bins=100)
    #     if eng.search_results and label in eng.search_results:
    #         hit_scores = [r.sort_score for r in eng.search_results[label].search_results]
    #         plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|', alpha=0.5)
    #     plt.title(f"Score distribution — {label}")

    #     pane = pn.pane.Matplotlib(fig, tight=True, height=260)
    #     self._main_col[:] = [self._status, pn.Card("## Scores histogram", pane, sizing_mode="stretch_width")]

    # # ------------------------------------------------------------------
    # # Protocol: utilities
    # # ------------------------------------------------------------------
    # def set_active_label(self, label: str | None) -> None:
    #     self._active_label = label

    # def clear_results(self) -> None:
    #     if self._active_label and self._active_label in self._rows_by_label:
    #         self._rows_by_label[self._active_label] = []
    #         self._selections_by_label[self._active_label] = {}
    #     self._status.object = "Cleared current results."

    # def confirm_overwrite(self, db_path: Path) -> bool:
    #     # For Panel apps you’d typically render a modal and await the user click.
    #     # Minimal stub: always confirm True for now.
    #     return True

    # def notify(self, msg: str, level: Literal["info","warn","error"] = "info") -> None:
    #     emoji = {"info":"ℹ️","warn":"⚠️","error":"❌"}.get(level, "ℹ️")
    #     self._status.object = f"{emoji} {msg}"
