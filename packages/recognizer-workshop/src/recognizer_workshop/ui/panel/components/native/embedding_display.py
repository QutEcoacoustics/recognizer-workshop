import matplotlib
matplotlib.use('Agg')

import asyncio
import html
import panel as pn
import numpy as np
from matplotlib import pyplot as plt
from perch_hoplite.agile.embedding_display import QueryDisplay, get_melspec_layer
from typing import Callable, List, Optional, Sequence, Dict, Any
from perch_hoplite.db import interface as hoplite_interface
from perch_hoplite.db import search_results as hoplite_search_results
from perch_hoplite import audio_io

from datetime import datetime, timedelta
from baw_helpers.baw_utils import make_baw_listen_link_from_arid, extract_domain_and_arid, parse_canonical_filename

def _default_load_audio(uri: str, sample_rate_hz: int) -> np.ndarray:

    return np.array(audio_io.load_audio(uri, sample_rate_hz))


def _parse_baw_timestamp(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    ts = timestamp
    if ts.endswith("Z"):
        ts = ts[:-1] + "+0000"
    if "-" in ts:
        try:
            return datetime.fromisoformat(ts.replace("+0000", "+00:00"))
        except ValueError:
            pass
    for fmt in ("%Y%m%dT%H%M%S%z", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _format_recording_datetime_parts(recording_info: dict | None, offset_s: float) -> tuple[str, str]:
    info = recording_info or {}
    base_dt = _parse_baw_timestamp(info.get("recorded_date"))
    if not base_dt:
        canonical_name = info.get("canonical_file_name")
        timestamp, _site, _arid = parse_canonical_filename(canonical_name) if canonical_name else (None, None, None)
        base_dt = _parse_baw_timestamp(timestamp)
    if not base_dt:
        return "Unknown date", "Unknown time"

    offset_dt = base_dt + timedelta(seconds=offset_s)
    tz_offset = offset_dt.utcoffset()
    date_text = offset_dt.strftime("%Y-%m-%d")
    if tz_offset is None:
        time_text = offset_dt.strftime("%H:%M:%S")
        return date_text, time_text

    total_minutes = int(tz_offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    abs_minutes = abs(total_minutes)
    hours = abs_minutes // 60
    minutes = abs_minutes % 60
    if minutes == 0:
        tz_text = f"{sign}{hours:02d}"
    else:
        tz_text = f"{sign}{hours:02d}:{minutes:02d}"
    time_text = f"{offset_dt.strftime('%H:%M:%S')} {tz_text}"
    return date_text, time_text


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
        
        # depending on the mood of the gremlins, the spec may be upside down
        #spec = np.flipud(spec)
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
        recording_info: Optional[dict[str, Any]] = None,
        domain: Optional[str] = None,
        arid: Optional[int] = None,
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
        self.recording_info = recording_info or {}
        self.domain = domain
        self.arid = arid
        self.labels = list(labels)
        self.sample_rate_hz = int(sample_rate_hz)
        self.frame_rate = int(frame_rate)
        self.window_size_s = float(window_size_s)
        self._melspec_layer = melspec_layer or get_melspec_layer(self.sample_rate_hz)
        self._audio_loader = audio_loader or (lambda u, sr: audio_io.load_audio(u, sr))
        self.listen_link = None
        listen_domain = self.domain
        listen_arid = self.arid
        if not (listen_domain and listen_arid):
            listen_domain, listen_arid = extract_domain_and_arid(uri)
        if listen_domain and listen_arid:
            self.listen_link = make_baw_listen_link_from_arid(
                arid=listen_arid,
                offset_s = self.offset_s,
                window_size_s = self.window_size_s,
                baw_domain = listen_domain,
                padding = 5)

        site_name = self.recording_info.get("sites.name")
        if not site_name:
            canonical_name = self.recording_info.get("canonical_file_name")
            _timestamp, parsed_site, _arid = parse_canonical_filename(canonical_name) if canonical_name else (None, None, None)
            site_name = parsed_site
        self.site_name = site_name or "Unknown site"
        self.recording_date_text, self.recording_time_text = _format_recording_datetime_parts(
            self.recording_info,
            self.offset_s,
        )

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
            height=200,
            sizing_mode="stretch_width"
        )
        self._spec_pane = pn.pane.Matplotlib(tight=True, height=200, sizing_mode="stretch_width")
        # Use a Column to switch between loading and spec
        self._spec_container = pn.Column(loading_msg, sizing_mode="stretch_width", margin=(0, 0, 4, 0))
        self._audio_pane = pn.pane.Audio(
            name="audio",
            sizing_mode="stretch_width",
            sample_rate=self.sample_rate_hz,
        )

        region_name = self.recording_info.get("regions.name")
        if isinstance(region_name, list):
            region_name = ", ".join([str(item) for item in region_name if item])

        site_id = self.recording_info.get("site_id")
        duration = self.recording_info.get("duration_seconds")

        info_lines = [
            f"eid: {self.embedding_id}",
            f"offset: {self.offset_s:.2f}s",
            f"dataset: {self.dataset_name}",
            f"domain: {self.domain or 'Unknown'}",
            f"arid: {self.arid or 'Unknown'}",
            f"date: {self.recording_date_text}",
            f"site: {self.site_name}",
        ]
        if site_id is not None:
            info_lines.append(f"site_id: {site_id}")
        if region_name:
            info_lines.append(f"region: {region_name}")
        if duration is not None:
            info_lines.append(f"duration: {duration}s")

        info_title = "\n".join(info_lines)
        info_icon = pn.pane.HTML(
            f"<div style='text-align: right;'>"
            f"<span title='{html.escape(info_title)}' "
            f"style='cursor: help; pointer-events: auto; display: inline-block;'>ℹ️</span>"
            f"</div>",
            sizing_mode="stretch_width",
        )

        # buttons grid
        self._buttons_grid = self._build_label_buttons(button_columns)

        # assemble
        score_text = f"({self.score:.2f})"
        title_left = f"{self.recording_time_text} • {self.site_name}"
        title_label = (
            "<div style='display: flex; align-items: baseline; justify-content: space-between; gap: 8px; "
            "font-size: 12px;'>"
            f"<span>{html.escape(title_left)}</span>"
            f"<span>{html.escape(score_text)}</span>"
            "</div>"
        )
        title_text = (
            f'<a href="{self.listen_link}" target="_blank" rel="noopener noreferrer">{title_label}</a>'
            if self.listen_link
            else title_label
        )
        card = pn.Card(
            info_icon,
            self._spec_container,
            pn.Row(self._audio_pane, sizing_mode="stretch_width", margin=(0, 0, 4, 0)),
            self._buttons_grid,
            self._status,
            title=title_text,
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
        def _show_audio_error(message: str):
            _ = message
            self._status.object = ""
            error_msg = pn.pane.HTML(
                "<div style='text-align: center; padding: 60px; color: #d32f2f;'>"
                "<div style='font-size: 16px;'>❌ Audio unavailable</div>"
                "<div style='font-size: 13px; margin-top: 6px;'>"
                "Could not load audio for this result."
                "</div>"
                "</div>",
                height=200,
                sizing_mode="stretch_width",
            )
            self._spec_container[:] = [error_msg]

        def _is_http_error(exc: Exception) -> bool:
            status = getattr(exc, "status", None) or getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if isinstance(status, int) and 400 <= status <= 599:
                return True
            msg = str(exc).lower()
            return any(token in msg for token in ["http", "404", "401", "403", "500", "502", "503", "504"])

        try:

            # load full audio
            raw_audio = await asyncio.to_thread(self._audio_loader, self.uri, self.offset_s)
            if raw_audio is None:
                _show_audio_error("❌ Audio unavailable for this result.")
                return

            self._audio = np.asarray(raw_audio)

            if self._audio.size == 0:
                _show_audio_error("❌ Audio unavailable for this result.")
                return

            # compute window spectrogram
            self._spec = await asyncio.to_thread(self._melspec_layer, self._audio)

            def _create_plot():
                fig, ax = plt.subplots()
                spec = self._spec.T
                # depending on the mood of the gremlins, the spec may be upside down
                #spec = np.flipud(spec)
                import librosa.display as lrd
                lrd.specshow(
                    spec,
                    sr=self.sample_rate_hz,
                    y_axis='mel',
                    x_axis='time',
                    hop_length=self.sample_rate_hz // self.frame_rate,
                    cmap='Greys',
                    ax=ax,
                )
                fig.subplots_adjust(left=0.06, right=0.995, top=0.99, bottom=0.08)
                return fig
            
            fig = await asyncio.to_thread(_create_plot)
            self._spec_pane.object = fig
            # Replace loading message with actual spectrogram
            self._spec_container[:] = [self._spec_pane]

            # audio panes
            self._audio_pane.object = self._audio

            self._status.object = ""
        except Exception as e:
            if _is_http_error(e):
                _show_audio_error("❌ Could not fetch remote audio (server/network error).")
            else:
                _show_audio_error("❌ Could not load or render audio for this result.")

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
    Displays a grid of PanelEmbeddingDisplay items with pagination.
    """
    def __init__(
        self,
        items: List[PanelEmbeddingDisplay],
        page_size: int = 10,
        item_width: int = 300,
        item_margin: int = 8,
        on_save: Optional[Callable[[], None]] = None,
        save_label: str = "Save responses",
        **kwargs,
    ):
        self._items = items
        self._page_size = max(1, int(page_size))
        self._page = 0
        self._item_width = max(1, int(item_width))
        self._item_margin = max(0, int(item_margin))
        self._list_box = pn.FlexBox(
            sizing_mode="stretch_width",
            flex_wrap="wrap",
            justify_content="flex-start",
            align_content="flex-start",
        )

        self._prev = pn.widgets.Button(name="Prev", button_type="default", width=60, sizing_mode="fixed")
        self._next = pn.widgets.Button(name="Next", button_type="default", width=60, sizing_mode="fixed")
        self._label = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._save_btn = None

        if on_save:
            self._save_btn = pn.widgets.Button(
                name=save_label,
                button_type="warning",
                width=160,
                sizing_mode="fixed",
            )
            self._save_btn.on_click(lambda _event: on_save())

        self._prev.on_click(lambda _: self._goto(-1))
        self._next.on_click(lambda _: self._goto(+1))

        header_items = [self._prev, self._next, self._label]
        if self._save_btn:
            header_items.extend([pn.Spacer(), self._save_btn])
        header = pn.Row(*header_items, sizing_mode="stretch_width")
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
        item_width: int = 300,
        item_margin: int = 8,
        on_save: Optional[Callable[[], None]] = None,
        save_label: str = "Save responses",
        sample_rate_hz: int = 32000,
        frame_rate: int = 100,
        window_size_s: float = 5.0,
        melspec_layer=None,
        audio_loader=None,
        **kwargs,
    ) -> "PanelEmbeddingDisplayGroup":
        _ = db
        items = []
        for r in results:
            metadata = getattr(r, "metadata", None) or {}
            source = metadata.get("source")
            if source is None:
                raise ValueError("Search results missing metadata.source; call attach_metadata_to_search_results")
            recording_info = metadata.get("recording_info")
            items.append(
                PanelEmbeddingDisplay(
                    embedding_id=int(r.embedding_id),
                    dataset_name=source.dataset_name,
                    uri=source.source_id,
                    offset_s=float(source.offsets[0]),
                    score=float(getattr(r, "sort_score", 0.0)),
                    recording_info=recording_info,
                    domain=metadata.get("domain"),
                    arid=metadata.get("arid"),
                    labels=list(labels),
                    sample_rate_hz=sample_rate_hz,
                    frame_rate=frame_rate,
                    window_size_s=window_size_s,
                    melspec_layer=melspec_layer,
                    audio_loader=audio_loader,
                    width=item_width,
                    margin=item_margin,
                )
            )
        return cls(
            items=items,
            page_size=page_size,
            item_width=item_width,
            item_margin=item_margin,
            on_save=on_save,
            save_label=save_label,
            **kwargs,
        )

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
        for item in current:
            item.width = self._item_width
            item.sizing_mode = "fixed"
            item.margin = self._item_margin
        self._list_box[:] = current
        self._label.object = f"Page {self._page + 1} of {self._pages() + 1}"
        # Enable/disable nav
        self._prev.disabled = (self._page == 0)
        self._next.disabled = (self._page >= self._pages())
