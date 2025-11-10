import matplotlib
matplotlib.use('Agg')

import asyncio
import panel as pn
import numpy as np
from matplotlib import pyplot as plt
from perch_hoplite.agile.embedding_display import QueryDisplay, get_melspec_layer
from typing import Callable, List, Optional, Sequence, Dict
from perch_hoplite.db import interface as hoplite_interface
from perch_hoplite.db import search_results as hoplite_search_results
from perch_hoplite import audio_io

def _default_load_audio(uri: str, sample_rate_hz: int) -> np.ndarray:

    return np.array(audio_io.load_audio(uri, sample_rate_hz))


class PanelQuerySelector(pn.Column):
    def __init__(self, uri: str, window_size_s: float = 5.0, **params):
        self.uri = str(uri)
        self.window_size_s = float(window_size_s)

        # UI bits
        self.status = pn.pane.Markdown("Loading audio…", sizing_mode="stretch_width")
        self.full_pane   = pn.pane.Matplotlib(tight=True, height=260, sizing_mode="stretch_width")
        self.window_pane = pn.pane.Matplotlib(tight=True, height=200, sizing_mode="stretch_width")
        self.slider = pn.widgets.FloatSlider(name="Offset (s)", start=0.0, end=0.0, step=0.05,
                                             value=0.0, disabled=True, sizing_mode="stretch_width")
        self.audio_full = pn.pane.Audio(name="Full audio", sample_rate=32000, sizing_mode="stretch_width")
        self.audio_win  = pn.pane.Audio(name="Selected window", sample_rate=32000, sizing_mode="stretch_width")

        card = pn.Card(
            pn.pane.Markdown(f"**Source:** `{self.uri}`"),
            self.status,
            self.full_pane,
            self.window_pane,
            pn.Row(self.slider),
            pn.Row(self.audio_full, self.audio_win),
            title="# Select query window",
            collapsible=False,
            sizing_mode="stretch_width",
        )
        super().__init__(card, sizing_mode="stretch_width", **params)

        # Backend object from perch_hoplite
        self.qd = QueryDisplay(uri=self.uri, window_size_s=self.window_size_s, sample_rate_hz=32000)

        # Wire slider -> re-render
        self.slider.param.watch(self._on_slider, "value")

        # Schedule init once document is ready
        if pn.state.curdoc:
            pn.state.curdoc.add_next_tick_callback(self._init)
        else:
            pn.state.onload(self._init)

    # --- internal helpers -------------------------------------------------

    def _on_slider(self, _):
        self.qd.offset_s = float(self.slider.value)
        self._render_all()

    def _render_spec(self, spec: np.ndarray, *, draw_window=False):
        fig, ax = plt.subplots()
        # qd.plot_spectrogram returns (fig, img, ax) if you’d rather reuse it;
        # here we inline to keep this class self-contained:
        from librosa import display as librosa_display
        librosa_display.specshow(
            spec, sr=self.qd.sample_rate_hz, y_axis='mel', x_axis='time',
            hop_length=self.qd.sample_rate_hz // self.qd.frame_rate, cmap='Greys', ax=ax
        )
        if draw_window:
            start = self.qd.offset_s
            end = start + self.qd.window_size_s
            ax.plot([start, start], [0, 16000], 'r:')
            ax.plot([end,   end],   [0, 16000], 'r:')
        return fig

    def _render_all(self):
        # Safety: audio must exist
        if self.qd.audio is None:
            self.status.object = "❌ Could not load audio."
            return

        # Update spectrograms for current offset
        self.qd.update_spectrogram()

        # Full spec with window markers
        if self.qd.full_spectrogram is not None:
            fig = self._render_spec(self.qd.full_spectrogram.T, draw_window=True)
            self.full_pane.object = fig

        # Window spec
        if self.qd.window_spectrogram is not None:
            fig = self._render_spec(self.qd.window_spectrogram.T, draw_window=False)
            self.window_pane.object = fig

        # Audio panes
        self.audio_full.object = self.qd.audio
        start = int(self.qd.offset_s * self.qd.sample_rate_hz)
        end   = int(start + self.qd.window_size_s * self.qd.sample_rate_hz)
        self.audio_win.object = self.qd.audio[start:end]

        self.status.object = "✅ Audio loaded. Move the slider to pick a window."

    async def _init(self):
        try:
            # Load audio (this must succeed or everything stays empty)
            self.qd.update_audio()
            if self.qd.audio is None or len(self.qd.audio) == 0:
                self.status.object = f"❌ No audio loaded for `{self.uri}`."
                return

            duration = len(self.qd.audio) / self.qd.sample_rate_hz
            self.slider.end = max(0.0, duration - self.window_size_s)
            self.slider.disabled = (self.slider.end <= 0.0)

            # First render
            self._render_all()
        except Exception as e:
            self.status.object = f"❌ Error loading audio: `{e}`"




class PanelEmbeddingDisplay(pn.Column):
    """
    Panel-native renderer for an embedded audio example (multi-label version).
    Mirrors the old ipywidgets EmbeddingDisplay behavior:
      - shows a mel spectrogram for a fixed window (offset_s .. offset_s+window_size_s)
      - plays full audio and the window audio
      - renders one tri-state button per label (skip→pos→neg→skip)
      - harvests selections as interface.Label[] like before
    """

    def __init__(
        self,
        *,
        embedding_id: int,
        dataset_name: str,
        uri: str,
        offset_s: float,
        score: float,
        labels: Sequence[str] = (),  
        sample_rate_hz: int = 32000,
        frame_rate: int = 100,
        window_size_s: float = 5.0,
        melspec_layer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        audio_loader: Optional[Callable[[str, int], np.ndarray]] = None,
        button_columns: int = 4,
        **kwargs,
    ):
        super().__init__(sizing_mode="stretch_width", **kwargs)

        # --- config / deps ---
        self.embedding_id = int(embedding_id)
        self.dataset_name = str(dataset_name)
        self.uri = str(uri)
        self.offset_s = float(offset_s)
        self.score = float(score)
        self.labels = list(labels)
        self.sample_rate_hz = int(sample_rate_hz)
        self.frame_rate = int(frame_rate)
        self.window_size_s = float(window_size_s)
        self._melspec_layer = melspec_layer or get_melspec_layer(self.sample_rate_hz)
        self._audio_loader = audio_loader or (lambda u, sr: audio_io.load_audio(u, sr))

        # --- state caches ---
        self._audio: Optional[np.ndarray] = None
        self._spec: Optional[np.ndarray] = None

        # tri-state per label: 0=skip, 1=pos, -1=neg
        self._label_states: Dict[str, int] = {lbl: 0 for lbl in self.labels}
        self._buttons: Dict[str, pn.widgets.Button] = {}

        # --- UI elements ---
        self._status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Start with a loading message instead of empty matplotlib pane
        loading_msg = pn.pane.HTML(
            "<div style='text-align: center; padding: 60px; color: #666;'>"
            "<div style='font-size: 16px;'>🔄 Loading spectrogram...</div>"
            "</div>", 
            height=240, 
            sizing_mode="stretch_width"
        )
        self._spec_pane = pn.pane.Matplotlib(tight=True, height=240, sizing_mode="stretch_width")
        # Use a Column to switch between loading and spec
        self._spec_container = pn.Column(loading_msg, sizing_mode="stretch_width")
        self._audio_pane = pn.pane.Audio(name="audio", sizing_mode="stretch_width", sample_rate=self.sample_rate_hz)

        # metadata line
        meta = pn.pane.Markdown(
            f"**eid:** {self.embedding_id} &nbsp;&nbsp; **offset:** {self.offset_s:.2f}s "
            f"&nbsp;&nbsp; **score:** {self.score:.2f} &nbsp;&nbsp; *{self.dataset_name}*",
            sizing_mode="stretch_width",
        )

        # buttons grid
        self._buttons_grid = self._build_label_buttons(button_columns)

        # assemble
        card = pn.Card(
            meta,
            self._spec_container,
            pn.Row(self._audio_pane, sizing_mode="stretch_width"),
            self._buttons_grid,
            self._status,
            title=f"Embedding {self.embedding_id}",
            collapsible=False,
            sizing_mode="stretch_width",
        )
        self[:] = [card]

        # lazy load
        pn.state.onload(self._load_and_render) if not pn.state.curdoc else pn.state.curdoc.add_next_tick_callback(self._load_and_render)



    def _build_label_buttons(self, button_columns: int) -> pn.viewable.Viewable:
        """
        Create one Panel Button per label. Clicking cycles:
        skip(default) -> pos (success) -> neg (warning) -> skip (default).
        """
        btns = []
        for lbl in self.labels:
            b = pn.widgets.Button(name=lbl, button_type="default")
            self._buttons[lbl] = b

            def _cycle(event=None, label=lbl, button=b):
                state = self._label_states[label]
                # 0 -> 1 -> -1 -> 0
                nxt = 1 if state == 0 else (-1 if state == 1 else 0)
                self._label_states[label] = nxt
                # update color
                button.button_type = "success" if nxt == 1 else ("warning" if nxt == -1 else "default")

            b.on_click(_cycle)
            btns.append(b)

        # neat grid, wraps after n columns
        return pn.GridBox(*btns, ncols=max(1, int(button_columns)))

    # async because loading remote audio may take time, and may need to be retried
    async def _load_and_render(self):
        try:

            # load full audio
            self._audio = await asyncio.to_thread(self._audio_loader, self.uri, self.offset_s)

            if not self._audio.size:
                self._status.object = f"❌ Audio loaded an empty segment for URI: {self.uri}"
                # Clear any old spectrogram that might be showing.
                error_msg = pn.pane.HTML(
                    "<div style='text-align: center; padding: 60px; color: #d32f2f;'>"
                    "<div style='font-size: 16px;'>❌ Failed to load audio</div>"
                    "</div>", 
                    height=240, 
                    sizing_mode="stretch_width"
                )
                self._spec_container[:] = [error_msg]
                return

            # compute window spectrogram
            self._spec = await asyncio.to_thread(self._melspec_layer, self._audio)

            def _create_plot():
                fig, ax = plt.subplots()
                spec = self._spec.T
                import librosa.display as lrd
                lrd.specshow(
                    #np.flipud(spec),
                    spec,
                    sr=self.sample_rate_hz,
                    y_axis='mel',
                    x_axis='time',
                    hop_length=self.sample_rate_hz // self.frame_rate,
                    cmap='Greys',
                    ax=ax,
                )
                ax.set_title(f"Window spectrogram ({self.window_size_s:.2f}s @ {self.offset_s:.2f}s)")
                return fig
            
            fig = await asyncio.to_thread(_create_plot)
            self._spec_pane.object = fig
            # Replace loading message with actual spectrogram
            self._spec_container[:] = [self._spec_pane]

            # audio panes
            self._audio_pane.object = self._audio

            self._status.object = ""
        except Exception as e:
            self._status.object = f"❌ Error loading/plotting audio: {e}"

    # ---------- public API (matches old semantics) ----------

    def harvest_labels(self, provenance: str) -> Sequence[hoplite_interface.Label]:
        """
        Convert button states to interface.Labels (skip is omitted).
        """
        out = []
        for lbl, state in self._label_states.items():
            if state == 0:
                continue
            lt = hoplite_interface.LabelType.POSITIVE if state == 1 else hoplite_interface.LabelType.NEGATIVE
            out.append(
                hoplite_interface.Label(
                    embedding_id=self.embedding_id,
                    label=lbl,
                    type=lt,
                    provenance=provenance,
                )
            )
        return out



class PanelEmbeddingDisplayGroup(pn.Column):
    """
    Paginates a list of PanelEmbeddingResult.
    """
    def __init__(self, items: List[PanelEmbeddingDisplay], page_size: int = 10, **kwargs):
        self._items = items
        self._page_size = max(1, int(page_size))
        self._page = 0
        self._list_box = pn.Column(sizing_mode="stretch_width")

        self._prev = pn.widgets.Button(name="Prev", button_type="default")
        self._next = pn.widgets.Button(name="Next", button_type="default")
        self._label = pn.pane.Markdown("", sizing_mode="stretch_width")

        self._prev.on_click(lambda _: self._goto(-1))
        self._next.on_click(lambda _: self._goto(+1))

        header = pn.Row(self._prev, self._next, self._label, sizing_mode="stretch_width")
        super().__init__(header, self._list_box, sizing_mode="stretch_both", **kwargs)
        self._refresh()

    def harvest_labels(self, provenance: str):
        out = []
        for it in self._items:
            out.extend(it.harvest_labels(provenance))
        return out
    

    @classmethod
    def from_search_results(
        cls,
        results: hoplite_search_results.TopKSearchResults,
        *,
        db: hoplite_interface.HopliteDBInterface,
        labels: Sequence[str],
        page_size: int = 10,
        sample_rate_hz: int = 32000,
        frame_rate: int = 100,
        window_size_s: float = 5.0,
        melspec_layer=None,
        audio_loader=None,
        **kwargs,
    ) -> "PanelEmbeddingDisplayGroup":
        items = []
        for r in results:
            src = db.get_embedding_source(r.embedding_id)
            items.append(
                PanelEmbeddingDisplay(
                    embedding_id=int(r.embedding_id),
                    dataset_name=src.dataset_name,
                    uri=src.source_id,
                    offset_s=float(src.offsets[0]),
                    score=float(getattr(r, "sort_score", 0.0)),
                    labels=list(labels),
                    sample_rate_hz=sample_rate_hz,
                    frame_rate=frame_rate,
                    window_size_s=window_size_s,
                    melspec_layer=melspec_layer,
                    audio_loader=audio_loader,
                )
            )
        return cls(items=items, page_size=page_size, **kwargs)

    # --- internals ---
    def _pages(self) -> int:
        n = len(self._items)
        return (n - 1) // self._page_size if n else 0

    def _slice(self):
        lo = self._page * self._page_size
        hi = min(lo + self._page_size, len(self._items))
        return self._items[lo:hi]

    def _goto(self, inc: int):
        self._page = max(0, min(self._page + inc, self._pages()))
        self._refresh()

    def _refresh(self):
        current = self._slice()
        self._list_box[:] = current
        self._label.object = f"Page {self._page + 1} of {self._pages() + 1} — showing {len(current)} / {len(self._items)}"
        # Enable/disable nav
        self._prev.disabled = (self._page == 0)
        self._next.disabled = (self._page >= self._pages())
