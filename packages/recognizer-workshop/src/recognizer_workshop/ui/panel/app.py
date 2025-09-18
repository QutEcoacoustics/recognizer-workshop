import base64
import os

os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

import panel as pn
import logging
from pathlib import Path
import re

from recognizer_workshop.ui.panel.view_model import ViewModel
from recognizer_workshop.ui.panel.pages import config, create_database, labeled_examples, search, explore, train, refine, inference, status
from recognizer_workshop.ui.panel.adaptors import PanelUI

# Load the tabulator extension for displaying DataFrames.

pn.extension('tabulator', sizing_mode="stretch_width")

# Set up static file serving
assets_dir = Path(__file__).parent / "assets"

# Load HoloViews extension upfront to prevent conflicts when explore page loads
import holoviews as hv
hv.extension('bokeh')

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


def encode_png_to_base64(path: Path) -> str:
    """Reads a PNG file and returns a base64 encoded data URI."""
    with open(path, "rb") as f:
        binary_data = f.read()
    encoded_data = base64.b64encode(binary_data).decode("utf-8")
    return f"data:image/png;base64,{encoded_data}"


def build_app(vm):
    # Define paths and encode logos to base64
    app_dir = Path(__file__).parent
    qut_logo_path = app_dir / "assets" / "qut_logo_2.png"
    ardc_logo_path = app_dir / "assets" / "ardc_logo_2.png"

    qut_logo_base64 = encode_png_to_base64(qut_logo_path)
    ardc_logo_base64 = encode_png_to_base64(ardc_logo_path)

    tabs = pn.Tabs(
        ("Config", config.build(vm)),
        ("Database", create_database.build(vm)),
        ("Labelled examples", labeled_examples.build(vm)),
        ("Search", search.build(vm)),
        ("Explore", explore.build(vm)),
        ("Train", train.build(vm)),
        ("Refine", refine.build(vm)),
        ("Inference", inference.build(vm)),
        ("Status", status.build(vm)),
        ("Logs", vm.log_pane),
        sizing_mode="stretch_both",
    )

    # Create pure HTML header with both logos embedded
    header_html = f"""
    <div style="position: absolute; top: 10px; right: 20px; z-index: 1000;
                display: flex; align-items: center; gap: 12px; background-color: rgba(255, 255, 255, 0.8); padding: 5px; border-radius: 8px;">
        <img src="{qut_logo_base64}" style="height: 50px; width: auto;">
        <img src="{ardc_logo_base64}" style="height: 50px; width: auto;">
        <h2 style="margin: 0; color: #333; font-size: 18px; font-weight: bold;">Agile Modelling</h2>
    </div>
    """

    header_pane = pn.pane.HTML(header_html, sizing_mode="stretch_width")

    return pn.Column(
        header_pane,
        error_banner,
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


# Serve the app for interactive development
if __name__.startswith("bokeh"):
    create_app_session().servable()