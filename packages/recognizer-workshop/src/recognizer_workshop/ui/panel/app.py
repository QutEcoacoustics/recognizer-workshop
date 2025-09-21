import base64
import os

os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

import panel as pn
import logging
from pathlib import Path

from recognizer_workshop.ui.panel.view_model import ViewModel
from recognizer_workshop.ui.panel.pages import config, create_database, labeled_examples, search, explore, train, refine, inference, status
from recognizer_workshop.ui.panel.adaptors import PanelUI

# Load necessary extensions
pn.extension('tabulator', sizing_mode="stretch_width")
import holoviews as hv
hv.extension('bokeh')

# --- Global Error Handling Setup ---
error_banner = pn.pane.Alert(visible=False, sizing_mode="stretch_width")

def global_exception_handler(ex):
    # Log the full error to the server console for your own debugging
    logging.error("An unhandled exception occurred", exc_info=ex)
    # Display a user-friendly message in the app's UI
    error_banner.object = (
        "💥 **An unexpected error occurred!** Please try refreshing the page."
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
    """Builds the main application layout using a Panel template."""
    
    vm.setup_help_system()

    template = pn.template.FastListTemplate(
        title="Agile Recognizer Workshop",
        sidebar_width=400,
        sidebar=[vm.help_system, vm.help_system.content]  # Pass components directly
    )


    
    # def sync_vm_to_template(event):
    #     # When VM sidebar_open changes, update template's collapsed_sidebar
    #     template.collapsed_sidebar = not event.new  # Note: collapsed is opposite of open
    
    # def sync_template_to_vm(event):
    #     # When template collapsed_sidebar changes, update VM sidebar_open
    #     vm.help_system.sidebar_open = not event.new  # Note: collapsed is opposite of open
    
    # # Watch for changes
    # vm.help_system.param.watch(sync_vm_to_template, 'sidebar_open')
    # template.param.watch(sync_template_to_vm, 'collapsed_sidebar')



    app_dir = Path(__file__).parent
    qut_logo_path = app_dir / "assets" / "qut_logo_2.png"
    ardc_logo_path = app_dir / "assets" / "ardc_logo_2.png"
    qut_logo_base64 = encode_png_to_base64(qut_logo_path)
    ardc_logo_base64 = encode_png_to_base64(ardc_logo_path)

    header_logos = pn.pane.HTML(f"""
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="{qut_logo_base64}" style="height: 40px; width: auto;">
        <img src="{ardc_logo_base64}" style="height: 40px; width: auto;">
    </div>
    """)
    
    # 3. Create the main content (tabs)
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



    template.header.append(header_logos)
    template.main.append(error_banner)
    template.main.append(tabs)
    # template.sidebar.append(vm.help_system) # append the js custom component for sidebar control
    # template.sidebar.append(vm.help_system.content) # append the actual help content
    return template


def create_app_session():
    """Creates a complete, isolated application session."""
    vm = ViewModel()
    vm.ui = PanelUI()
    app = build_app(vm)
    return app, vm


# Serve the app for interactive development
if __name__.startswith("bokeh"):
    app, vm = create_app_session()
    app.servable()