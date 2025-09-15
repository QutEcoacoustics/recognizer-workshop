# probably can delete this file entirely - I think it's been replaced with the panel native components

# # A wrapper for QueryDisplay instances
# # This removes assumptions about running in a notebook cell, so that we can run in a panel app




# import ipywidgets as widgets
# from IPython.display import clear_output
# from contextlib import contextmanager

# def make_qd_widget(qd):
#     """
#     Wraps an existing unmodifiable QueryDisplay instance into a robust ipywidgets
#     UI that renders reliably under Panel via pn.ipywidget(...).
#     """
#     out = widgets.Output()                             # our “cell output”
#     slider = widgets.FloatSlider(
#         value=float(qd.offset_s),
#         min=0.0,
#         max=1.0,     # temp; we’ll update after audio loads
#         step=0.01,
#         description="offset_s",
#         continuous_update=False,
#         readout=True,
#         readout_format=".2f",
#         layout=widgets.Layout(width="100%"),
#     )

#     # 1) ensure audio is loaded so we can set the slider range
#     qd.update_audio()
#     if qd.audio is not None and qd.sample_rate_hz:
#         slider.max = qd.audio.shape[0] / qd.sample_rate_hz

#     def render():
#         # Render everything *inside* the Output widget
#         with out:
#             clear_output(wait=True)
#             qd.display()  # uses plt.show(), IPython.display.Audio, prints, etc.

#     def on_slider_change(change):
#         if change["name"] == "value":
#             qd.offset_s = float(change["new"])
#             qd.update_spectrogram()
#             render()

#     slider.observe(on_slider_change, names="value")

#     # initial render
#     qd.update_spectrogram()
#     render()

#     return widgets.VBox([slider, out], layout=widgets.Layout(width="100%"))
