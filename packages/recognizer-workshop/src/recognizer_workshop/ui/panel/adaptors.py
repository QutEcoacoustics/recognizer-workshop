from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence
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


    def view_target_labels(self, eng) -> pn.layout.Panel:
        """Builds and returns the UI component for defining target labels."""
        labels_container = pn.Column(sizing_mode="stretch_width")
        btn_add_label = pn.widgets.Button(name="Add Label", icon="plus", button_type="success")
        current_labels_display = pn.pane.Markdown()

        label_editor_section = pn.Column(
            labels_container,
            btn_add_label,
            sizing_mode="stretch_width"
        )

        def sync_and_update_display(*events):
            # Save the config whenever a change is made
            eng.config.save() # Assumes config_path is stored on eng.config
            description = eng.get_target_label_description()
            current_labels_display.object = f"**Summary:** {description}" if description else "*(No labels defined)*"


        def _remove_label(index, event):  
            if 0 <= index < len(eng.config.target_labels):
                eng.config.target_labels.pop(index)  
                _build_labels_ui()
                sync_and_update_display()


        def _build_labels_ui():
            labels_container.clear()

            if not hasattr(eng, 'baw_api') or eng.baw_api is None:
                return


            for i, label in enumerate(eng.config.target_labels):
                species_widget = pn.widgets.AutocompleteInput(
                    name=f"Species {i+1}", 
                    options=eng.workbench_tags, 
                    value=label.species,
                    placeholder="Type to search for a species...",
                    case_sensitive=False
                )
                call_types_widget = pn.widgets.TextInput(
                    name="Call Type", 
                    value=label.call_type, 
                    placeholder="(optional) e.g., song")
                remove_btn = pn.widgets.Button(name="Remove", icon="trash", button_type="danger")
                remove_btn.on_click(partial(_remove_label, label))

                def _update_species(event, index=i):
                    # Replace the LabelName instance instead of mutating it
                    old_label = eng.config.target_labels[index]
                    eng.config.target_labels[index] = LabelName(
                        species=event.new,
                        call_type=old_label.call_type
                    )
                    sync_and_update_display()

                def _update_call_type(event, index=i):
                    # Replace the LabelName instance instead of mutating it
                    old_label = eng.config.target_labels[index]
                    eng.config.target_labels[index] = LabelName(
                        species=old_label.species,
                        call_type=event.new.strip() or None
                    )
                    sync_and_update_display()

                species_widget.param.watch(_update_species, 'value')
                call_types_widget.param.watch(_update_call_type, 'value')

                labels_container.append(pn.Row(species_widget, call_types_widget, 
                                               remove_btn, sizing_mode="stretch_width"))

        def _add_label(event):
            eng.config.target_labels.append(LabelName(species="", call_type=None))
            _build_labels_ui()
            sync_and_update_display()

        btn_add_label.on_click(_add_label)



        def _init():
            has_baw_api = hasattr(eng, 'baw_api') and eng.baw_api is not None
            label_editor_section.visible = has_baw_api
            
            if has_baw_api:
                _build_labels_ui()
                sync_and_update_display()
            else:
                current_labels_display.object = "*(Connect to BAW API to define labels)*"
                labels_container.clear()


        # Initial state
        _init()

        @pn.depends(self.vm.obs.ensure("engine_state").param.version, watch=True)
        def _update_visibility(_):
            _init()


        return pn.Column(
            pn.pane.Alert(
                "Define one or more target species. For each species, you can optionally specify distinct call types.",
                alert_type="info"
            ),
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
            tags = eng.baw_api.baw_filter(
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
        self.show_results(eng, active_label, results_region=results_region)
        log.info(f'Results rendered for label {active_label}.')


    def search_with_classifier(
        self,
        eng,
        *,
        num_results: int = 10,
        target_score=None,
        results_region: str = "refine.results",
    ):
        """Read samples from the selector, embed, run search, then render results."""

        eng.search_with_classifier(num_results=num_results, sample_size=None, target_score=target_score)

        active_label = eng.active_label
        if not active_label:
            self.render(pn.pane.Alert("Pick target label(s) on the Config tab.", alert_type="warning"), results_region)
            return
        self.show_results(eng, active_label, results_region=results_region)



    def show_results(
        self,
        eng,
        label: str,
        *,
        results_region: str,
        page_size: int = 10,
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
