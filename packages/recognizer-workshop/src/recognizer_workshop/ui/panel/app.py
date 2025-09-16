import os
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

import panel as pn
import logging

from recognizer_workshop.ui.panel.view_model import ViewModel
from recognizer_workshop.ui.panel.pages import config, create_database, labeled_examples, search, train, refine, inference, status
from recognizer_workshop.ui.panel.adaptors import PanelUI

# Load the tabulator extension for displaying DataFrames.

pn.extension('tabulator', sizing_mode="stretch_width")

# Don't think we need 'ipywidgets' anymore since we use pure Panel components
#pn.extension('ipywidgets', 'tabulator', sizing_mode="stretch_width")

error_banner = pn.pane.Alert(visible=False, sizing_mode="stretch_width")

def global_exception_handler(ex):
    # Log the full error to the server console for your own debugging
    logging.error("An unhandled exception occurred", exc_info=ex)
    # Display a user-friendly message in the app's UI
    error_banner.object = (
        "💥 **An unexpected error occurred!** Please try refreshing the page, "
        "or click the 'Reset Session' button."
    )
    error_banner.alert_type = 'danger'
    error_banner.visible = True

pn.config.exception_handler = global_exception_handler


def build_app(vm):


    tabs = pn.Tabs(

        ("Config",            config.build(vm)),
        ("Database",          create_database.build(vm)),
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
        pn.Row("# Agile Modelling", error_banner),
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