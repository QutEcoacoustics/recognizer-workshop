
import os
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")


import panel as pn
from recognizer_workshop.ui.panel.view_model import ViewModel
from recognizer_workshop.ui.panel.pages import config, labeled_examples, search, train, refine, inference, status

import logging
from recognizer_workshop.ui.panel.utils import PanelLogHandler

from recognizer_workshop.ui.panel.adaptors import PanelUI

# Without ipywidgets, might get blank panes or “unknown model” 
# errors when embedding complex ipywidgets.
# TODO: probably don't need this now that we use pure Panel components?
pn.extension('ipywidgets', sizing_mode="stretch_width")
print("Panel extension loaded.")

def build_app(vm):


    tabs = pn.Tabs(
        ("Config",            config.build(vm)),
        ("Labelled examples", labeled_examples.build(vm)),
        ("Search",            search.build(vm)),
        ("Train",             train.build(vm)),
        ("Refine",            refine.build(vm)),
        ("Inference",         inference.build(vm)),
        ("Status",            status.build(vm)), 
        ("Logs",              vm.log_pane), 
        sizing_mode="stretch_both",
    )
    return pn.Column(
        "# Agile Modelling",
        tabs,
        sizing_mode="stretch_both",
    )

def create_app_session():
    """Creates a complete, isolated application session."""
    vm = ViewModel()
    vm.ui = PanelUI()
    app = build_app(vm)
    # Returns both the UI and the state object
    return app, vm