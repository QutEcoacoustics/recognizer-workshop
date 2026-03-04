# import panel as pn
# import param
# import numpy as np
# pn.extension('ipywidgets', sizing_mode="stretch_width")

# from recognizer_workshop.ui.panel.components.ipy_widgets.widget_wrapper import make_qd_widget

# # Import the third-party, unmodifiable class
# from perch_hoplite.agile.embedding_display import QueryDisplay

# class QuerySelector(pn.Column):
#     value = param.Array(default=np.array([], dtype=np.float32))

#     def __init__(self, uri: str, window_size_s: float = 5.0, **params):
#         self.uri = uri
#         self.window_size_s = window_size_s

#         self.qd = QueryDisplay(
#             uri=str(uri),
#             offset_s=0.0,
#             window_size_s=window_size_s,
#             sample_rate_hz=32000,
#         )

#         self.status_txt = pn.pane.Markdown("")
#         self.use_btn = pn.widgets.Button(name="Use This 5s Window", button_type="primary")

#         # ✅ robust ipywidget container (no interact/clear_output at cell scope)
#         ipyw = make_qd_widget(self.qd)
#         self.ipywidget_pane = pn.ipywidget(ipyw, sizing_mode="stretch_width")

#         card = pn.Card(
#             self.ipywidget_pane,
#             pn.Row(self.use_btn, self.status_txt),
#             title="# Select Query Window",
#             collapsible=False,
#         )
#         super().__init__(card, sizing_mode="stretch_width", **params)

#         self.use_btn.on_click(self._on_use)

#     def _on_use(self, _):
#         audio_samples = self.qd.get_audio_window()
#         if audio_samples is not None and audio_samples.size > 0:
#             self.value = audio_samples
#             self.status_txt.object = "✅ **Window captured!**"
#         else:
#             self.status_txt.object = "⚠️ Could not capture audio window."