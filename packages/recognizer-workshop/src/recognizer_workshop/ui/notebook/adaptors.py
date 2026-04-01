from typing import Iterable, Literal, Optional
from pathlib import Path
import logging
import numpy as np

import ipywidgets as widgets
from IPython.display import display
from recognizer_workshop.ui.notebook import agile_widgets
from recognizer_workshop.perch_hoplite_custom_modifications import embedding_display
from matplotlib import pyplot as plt
from perch_hoplite.db import interface as hoplite_interface
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files

log = logging.getLogger(__name__)


class NotebookUI:
    """Stateful notebook UI adapter that wraps perch-hoplite UI widgets and stores EDGs per label."""

    def __init__(self,
                 annotator_id: str,
                 sample_rate_hz: int = 32000,
                 frame_rate: int = 100):
        self.annotator_id = annotator_id
        self.sample_rate_hz = sample_rate_hz
        self.frame_rate = frame_rate
        self._edg_by_label: dict[str, embedding_display.EmbeddingDisplayGroup] = {}
        self._active_label: Optional[str] = None
        self._last_query_samples: Optional[np.ndarray] = None


    # -------- Label selection --------
    def pick_label(self, eng, initial_species: str | None = None,
                   initial_call_types: list[str] | None = None) -> None:
        # Use your existing widget; bind to eng.label
        if initial_call_types is None:
            initial_call_types = []
        agile_widgets.label_selector(
            baw_api=eng.baw_api,
            binding=eng.label,
            initial_species=initial_species,
            initial_call_types=initial_call_types,
        )

    # -------- Query window selection --------
    def select_query_window(self, uri: str, window_size_s: float = 5.0) -> np.ndarray | None:
        # normalize Xeno-Canto id and relative path resolution happens outside here
        if uri.startswith('XC'):
            uri = 'xc' + uri[2:]

        qd = embedding_display.QueryDisplay(
            uri=str(uri),
            offset_s=0.0,
            window_size_s=window_size_s,
            sample_rate_hz=self.sample_rate_hz,
        )
        qd.display_interactive()   # user moves slider
        audio_samples = qd.get_audio_window()
        self._last_query_samples = audio_samples
        return audio_samples


    # -------- Show a few labeled examples --------
    def show_examples(self, examples: list[dict], eng) -> None:
        # examples: [{'eid','dataset_name','source_id','offset','score'}, ...]
        members = []
        for ex in examples:
            members.append(
                embedding_display.EmbeddingDisplay(
                    embedding_id=ex['eid'],
                    dataset_name=ex['dataset_name'],
                    uri=ex['source_id'],
                    offset_s=ex['offset'],
                    score=ex.get('score', 0.0),
                    sample_rate_hz=self.sample_rate_hz,
                    frame_rate=self.frame_rate,
                )
            )
        edg = embedding_display.EmbeddingDisplayGroup.create(
            members=members,
            sample_rate_hz=self.sample_rate_hz,
            audio_loader=eng.audio_filepath_loader,
        )
        edg.baw_config = eng.config.baw_config
        edg.display(positive_labels=[])  # just visualize; no active-label stored


    # -------- Show search results and remember EDG(s) --------
    def show_results(self, eng, results_or_label) -> None:
        # Accept either a TopKSearchResults object, or a label key to pull from eng.search_results
        if hasattr(results_or_label, "search_results"):
            # No label key provided; try to infer from engine target labels
            label_key = (eng.get_target_labels() or ["_"])[0]
            results = results_or_label
        else:
            label_key = str(results_or_label)
            results = eng.search_results.get(label_key)
            if results is None:
                self.notify(f"No search results stored for label '{label_key}'", "warn")
                return

        edg = embedding_display.EmbeddingDisplayGroup.from_search_results(
            results=results,
            db=eng.db,
            sample_rate_hz=self.sample_rate_hz,
            frame_rate=self.frame_rate,
            audio_loader=eng.audio_filepath_loader,
        )
        edg.baw_config = eng.config.baw_config
        # Show buttons with that label selected as "positive option"
        edg.display(positive_labels=[label_key])

        self._edg_by_label[label_key] = edg
        self._active_label = label_key


    # -------- Harvest labels from the active EDG --------
    def harvest_labels(self) -> Iterable[hoplite_interface.Label]:
        if not self._active_label:
            return []
        edg = self._edg_by_label.get(self._active_label)
        if not edg:
            return []
        return edg.harvest_labels(self.annotator_id)


    # -------- Histogram viewer --------
    def plot_scores_for(self, eng, label: str) -> None:
        scores = None
        # engine tweak below will populate eng.search_scores[label]
        if getattr(eng, "search_scores", None) and label in eng.search_scores:
            scores = eng.search_scores[label]
        if scores is None:
            self.notify(f"No stored scores for label '{label}'. Run search first.", "warn")
            return
        plt.figure()
        _ = plt.hist(scores, bins=100)
        # mark current hits if available
        if eng.search_results and label in eng.search_results:
            hit_scores = [r.sort_score for r in eng.search_results[label].search_results]
            plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|', alpha=0.5)
        plt.title(f"Score distribution — {label}")
        plt.show()


    # -------- Utilities --------
    def set_active_label(self, label: str | None) -> None:
        self._active_label = label

    def clear_results(self) -> None:
        self._edg_by_label.clear()
        self._active_label = None

    def confirm_overwrite(self, db_path: Path) -> bool:
        # Could wire an ipywidget dialog; default accept for now
        return True

    def notify(self, msg: str, level: Literal["info","warn","error"] = "info") -> None:
        prefix = {"info": "ℹ️", "warn": "⚠️", "error": "❌"}.get(level, "ℹ️")
        log_msg = f"{prefix} {msg}"
        if level == "error":
            log.error(log_msg)
        elif level == "warn":
            log.warning(log_msg)
        else:
            log.info(log_msg)


