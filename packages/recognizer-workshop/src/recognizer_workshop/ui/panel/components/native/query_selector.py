# This has been replaced with the PanelQueryDisplay class


# import panel as pn
# import param
# import numpy as np
# from matplotlib import pyplot as plt
# from perch_hoplite.agile.embedding_display import QueryDisplay

# class QuerySelector(pn.Column):
#     """
#     A Panel component for selecting a window from an audio file.
#     Wraps the perch_hoplite.agile.embedding_display.QueryDisplay class
#     mainly to remove assumptions about where it's drawing to
    
#     """

#     # This parameter will hold the output of the component.
#     # The parent (PanelUI) can "watch" this parameter for changes.
#     value = param.Array(default=np.array([], dtype=np.float32))

#     def __init__(self, uri: str, window_size_s: float = 5.0, **params):

#         self.uri = uri
#         self.window_size_s = window_size_s
#         self.qd = QueryDisplay(
#             uri=str(uri),
#             window_size_s=window_size_s,
#             sample_rate_hz=32000
#         )
#         self.selection_lines = []


#         self.status_txt = pn.pane.Markdown("Loading audio...")
#         self.plot_pane = pn.pane.Matplotlib(tight=True, height=250)
#         self.audio_pane = pn.pane.Audio()
#         self.offset_slider = pn.widgets.FloatSlider(
#             name="Offset (s)", start=0.0, step=0.1, value=0.0, disabled=True
#         )
#         self.use_btn = pn.widgets.Button(name="Use This 5s Window", button_type="primary", disabled=True)


#         super().__init__(
#             pn.Card(
#                 self.status_txt,
#                 self.plot_pane,
#                 self.audio_pane,
#                 pn.Row(self.offset_slider, self.use_btn),
#                 title="# Select Query Window",
#                 collapsible=False
#             ),
#             sizing_mode="stretch_width",
#             **params
#         )

#         self.offset_slider.param.watch(self._update_lines, "value")
#         self.use_btn.on_click(self._on_use)

#         if pn.state.curdoc:
#             # we're already in a running session: schedule for the next tick
#             pn.state.curdoc.add_next_tick_callback(lambda: self._load_and_initialize())
#         else:
#             # register to run once the session is created (works under panel serve & notebooks)
#             pn.state.onload(self._load_and_initialize)


#     def _update_lines(self, event=None):
#         """Draws/updates the vertical lines that delimit the start and end."""
#         if not self.plot_pane.object: return
#         fig = self.plot_pane.object
#         ax = self.plot_pane.object.axes[0]
#         for line in self.selection_lines: line.remove()
#         self.selection_lines.clear()

#         start_s = self.offset_slider.value
#         end_s = start_s + self.window_size_s
#         line1 = ax.axvline(start_s, color='r', linestyle=':')
#         line2 = ax.axvline(end_s, color='r', linestyle=':')
#         self.selection_lines.extend([line1, line2])
#         fig.canvas.draw_idle()     
#         self.plot_pane.param.trigger('object')

#     def _on_use(self, _):
#         """When 'Use' is clicked, set the component's 'value' parameter."""
#         self.qd.offset_s = self.offset_slider.value
#         # This is the key change: we set our own 'value' to publish the result.
#         self.value = self.qd.get_audio_window()
#         self.status_txt.object = f"✅ **Window captured!** The main app can now use this selection."


#     def capture(self):
#         """Programmatically capture the current 5s window (equivalent to pressing the button)."""
#         self._on_use(None)


#     def _load_and_initialize(self):
#         """Loads audio and sets up the initial view."""
#         try:
#             self.qd.update_audio()
#             if self.qd.audio is None: raise ValueError("Audio data could not be loaded.")
            
#             duration = len(self.qd.audio) / self.qd.sample_rate_hz
#             self.qd.update_spectrogram()
#             fig, _, ax = self.qd.plot_spectrogram(self.qd.full_spectrogram.T)
#             ax.set_title(f"Full Spectrogram ({duration:.1f}s)")
#             self.plot_pane.object = fig
            
#             if duration > self.window_size_s:
#                 self.offset_slider.end = duration - self.window_size_s
#                 self.offset_slider.disabled = False
#                 self._update_lines()
            
#             self.audio_pane.object = self.qd.audio
#             self.status_txt.object = "Audio loaded. Move the slider to select a window."
#             self.use_btn.disabled = False
#         except Exception as e:
#             self.status_txt.object = f"❌ **Error loading audio**: {e}"