from __future__ import annotations

from typing import Iterable, Literal, Optional
from pathlib import Path
import numpy as np

import panel as pn
import param
from matplotlib import pyplot as plt

from perch_hoplite import audio_io
from perch_hoplite.db import interface as hoplite_interface

# Optional: if you want to fetch tags from BAW
# and to build ecosounds audio URLs for examples
from baw_helpers import baw_utils


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

    # public, handy if the controller wants to introspect
    view = param.Parameter(default=None)

    # ---- internal state
    _annotator_id: str
    _sample_rate_hz: int
    _frame_rate: int

    _active_label: Optional[str]
    _last_query_samples: Optional[np.ndarray]

    # selections cache: label -> { embedding_id: "pos"/"neg"/"skip" }
    _selections_by_label: dict[str, dict[int, str]]

    # rows cache: label -> list of dicts describing each result row
    #   each row contains: {"embedding_id","dataset_name","uri","offset","score","radio"}
    _rows_by_label: dict[str, list[dict]]

    # panel containers
    _status = None
    _main_col = None

    def __init__(self,
                 annotator_id: str,
                 sample_rate_hz: int = 32000,
                 frame_rate: int = 100):
        super().__init__()
        pn.extension()
        self._annotator_id = annotator_id
        self._sample_rate_hz = sample_rate_hz
        self._frame_rate = frame_rate

        self._active_label = None
        self._last_query_samples = None
        self._selections_by_label = {}
        self._rows_by_label = {}

        self._status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._main_col = pn.Column(self._status, sizing_mode="stretch_width")
        self.view = self._main_col  # expose a top-level view to mount

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
            # normalize capitalization (like your notebook UI does)
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

    # ------------------------------------------------------------------
    # Protocol: select_query_window(uri) -> np.ndarray
    # ------------------------------------------------------------------
    def select_query_window(self, uri: str, window_size_s: float = 5.0) -> np.ndarray:
        """Display a small tool to choose the 5s window; store samples internally.

        Returns the *current* selection immediately (offset 0 initially).
        The controller should embed using `eng.embed_query(self._last_query_samples)`
        when the user clicks 'Use 5s window'.
        """
        # normalize Xeno-Canto ID
        if uri.startswith("XC"):
            uri = "xc" + uri[2:]

        status_txt = pn.pane.Markdown("", sizing_mode="stretch_width")
        offset = pn.widgets.FloatSlider(
            name="Offset (s)",
            start=0.0, end=30.0, step=0.1, value=0.0, sizing_mode="stretch_width"
        )
        use_btn = pn.widgets.Button(name="Use 5s window", button_type="primary")
        spec_pane = pn.pane.Matplotlib(height=240, tight=True)
        audio_info = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Lazily load audio once; if it fails, keep status
        _audio = {'arr': None}

        def _load_audio():
            if _audio['arr'] is None:
                try:
                    # audio_io.load_audio handles files/urls/xc
                    _audio['arr'] = np.array(audio_io.load_audio(uri, self._sample_rate_hz))
                    status_txt.object = f"Loaded audio: {len(_audio['arr'])/self._sample_rate_hz:.1f}s"
                    # set slider max to clip length
                    offset.end = max(0.0, max(0.0, len(_audio['arr']) / self._sample_rate_hz - window_size_s))
                except Exception as e:
                    status_txt.object = f"**Error loading audio**: {e}"
                    _audio['arr'] = None

        def _draw_spec():
            if _audio['arr'] is None:
                return
            st = int(offset.value * self._sample_rate_hz)
            ed = int(st + window_size_s * self._sample_rate_hz)
            win = _audio['arr'][st:ed]
            if win.size == 0:
                return
            # very simple waveform plot (feel free to replace with your web spectrogram)
            fig = plt.figure(figsize=(6, 2.5))
            t = np.arange(win.size) / self._sample_rate_hz
            plt.plot(t, win)
            plt.title(f"Query window: {offset.value:.2f}s → {offset.value+window_size_s:.2f}s")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            spec_pane.object = fig
            audio_info.object = f"Samples: {win.size}"

        def _on_offset(_):
            _load_audio()
            _draw_spec()

        def _on_use(_):
            _load_audio()
            if _audio['arr'] is None:
                return
            st = int(offset.value * self._sample_rate_hz)
            ed = int(st + window_size_s * self._sample_rate_hz)
            win = _audio['arr'][st:ed]
            self._last_query_samples = win
            status_txt.object = "✅ Window captured. You can now embed it."

        offset.param.watch(_on_offset, "value")
        use_btn.on_click(_on_use)

        _load_audio()
        _draw_spec()

        card = pn.Card(
            "# Select query window",
            pn.Row(offset, use_btn),
            spec_pane,
            audio_info,
            status_txt,
            collapsible=False,
            sizing_mode="stretch_width",
        )
        self._main_col[:] = [self._status, card]

        # return current window (offset=0) for convenience
        if _audio['arr'] is not None:
            st0 = 0
            ed0 = int(window_size_s * self._sample_rate_hz)
            win0 = _audio['arr'][st0:ed0]
            self._last_query_samples = win0
            return win0
        else:
            return np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Protocol: show_results(eng, results, *, positive_labels)
    # ------------------------------------------------------------------
    def show_results(self, eng, results, *, positive_labels: list[str]) -> None:
        """Render TopK results and store selections. One label expected here."""
        if not positive_labels:
            self.notify("No label provided for rendering results.", "warn")
            return
        label_key = positive_labels[0]
        self._active_label = label_key
        self._selections_by_label.setdefault(label_key, {})
        self._rows_by_label[label_key] = []

        rows = []
        for r in results:
            src = eng.db.get_embedding_source(r.embedding_id)
            uri = src.source_id
            off = float(src.offsets[0])
            score = float(r.sort_score)

            radio = pn.widgets.RadioButtonGroup(
                name=f"eid={r.embedding_id}",
                options=[("skip","skip"),("pos","pos"),("neg","neg")],
                value="skip",
                button_type="default",
            )

            # keep selection in the cache
            def _bind(embed_id: int, w=radio):
                def _cb(_):
                    self._selections_by_label[label_key][embed_id] = w.value
                return _cb
            radio.param.watch(_bind(r.embedding_id), "value")

            row_dict = dict(
                embedding_id=int(r.embedding_id),
                dataset_name=src.dataset_name,
                uri=uri,
                offset=off,
                score=score,
                radio=radio,
            )
            self._rows_by_label[label_key].append(row_dict)

            rows.append(
                pn.Row(
                    pn.pane.Markdown(
                        f"**eid**: {r.embedding_id} &nbsp;&nbsp; "
                        f"**offset**: {off:.2f}s &nbsp;&nbsp; **score**: {score:.2f}",
                        width=500
                    ),
                    radio,
                    sizing_mode="stretch_width",
                )
            )

        header = pn.pane.Markdown(f"### Search results – label: **{label_key}**")
        self._main_col[:] = [self._status, header, *rows]

    # ------------------------------------------------------------------
    # Protocol: show_examples(examples, eng)
    # ------------------------------------------------------------------
    def show_examples(self, examples: list[dict], eng) -> None:
        """Very simple viewer: just list the examples. Replace with spectrograms if needed."""
        items = []
        for ex in examples:
            items.append(pn.pane.Markdown(
                f"- eid: **{ex['eid']}** &nbsp; dataset: `{ex['dataset_name']}` "
                f"&nbsp; offset: {ex['offset']:.2f}s",
                sizing_mode="stretch_width",
            ))
        card = pn.Card("## Random labeled examples", *items, sizing_mode="stretch_width")
        self._main_col[:] = [self._status, card]

    # ------------------------------------------------------------------
    # Protocol: harvest_labels() -> Iterable[Label]
    # ------------------------------------------------------------------
    def harvest_labels(self) -> Iterable[hoplite_interface.Label]:
        if not self._active_label:
            return []
        label_key = self._active_label
        selected = self._selections_by_label.get(label_key, {})
        out: list[hoplite_interface.Label] = []
        for row in self._rows_by_label.get(label_key, []):
            eid = row["embedding_id"]
            choice = selected.get(eid, "skip")
            if choice == "skip":
                continue
            lt = hoplite_interface.LabelType.POSITIVE if choice == "pos" else hoplite_interface.LabelType.NEGATIVE
            out.append(
                hoplite_interface.Label(
                    embedding_id=int(eid),
                    label=str(label_key),
                    type=lt,
                    provenance=self._annotator_id,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Protocol: plot_scores_histogram(eng, label)
    # ------------------------------------------------------------------
    def plot_scores_histogram(self, eng, label: str) -> None:
        scores = None
        if getattr(eng, "search_scores", None) and label in eng.search_scores:
            scores = eng.search_scores[label]
        if scores is None:
            self.notify(f"No stored scores for label '{label}'. Run search first.", "warn")
            return

        fig = plt.figure(figsize=(6, 3))
        _ = plt.hist(scores, bins=100)
        if eng.search_results and label in eng.search_results:
            hit_scores = [r.sort_score for r in eng.search_results[label].search_results]
            plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|', alpha=0.5)
        plt.title(f"Score distribution — {label}")

        pane = pn.pane.Matplotlib(fig, tight=True, height=260)
        self._main_col[:] = [self._status, pn.Card("## Scores histogram", pane, sizing_mode="stretch_width")]

    # ------------------------------------------------------------------
    # Protocol: utilities
    # ------------------------------------------------------------------
    def set_active_label(self, label: str | None) -> None:
        self._active_label = label

    def clear_results(self) -> None:
        if self._active_label and self._active_label in self._rows_by_label:
            self._rows_by_label[self._active_label] = []
            self._selections_by_label[self._active_label] = {}
        self._status.object = "Cleared current results."

    def confirm_overwrite(self, db_path: Path) -> bool:
        # For Panel apps you’d typically render a modal and await the user click.
        # Minimal stub: always confirm True for now.
        return True

    def notify(self, msg: str, level: Literal["info","warn","error"] = "info") -> None:
        emoji = {"info":"ℹ️","warn":"⚠️","error":"❌"}.get(level, "ℹ️")
        self._status.object = f"{emoji} {msg}"
