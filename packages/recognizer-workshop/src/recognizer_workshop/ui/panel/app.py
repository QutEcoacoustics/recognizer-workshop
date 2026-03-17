
import base64
import logging
import os
import sys
import time
import importlib
from pathlib import Path

print(f"[startup] app.py import start {time.time():.3f}")
import holoviews as hv
import panel as pn

from recognizer_workshop.ui.panel.view_model import ViewModel
from recognizer_workshop.ui.panel.adaptors import PanelUI

os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

# Load necessary extensions
pn.extension('tabulator', sizing_mode="stretch_width")
print(f"[startup] panel extension loaded {time.time():.3f}")
hv.extension('bokeh')
print(f"[startup] holoviews extension loaded {time.time():.3f}")

def _import_page(name: str):
    start = time.time()
    module = importlib.import_module(f"recognizer_workshop.ui.panel.pages.{name}")
    print(f"[startup] imported pages.{name} {time.time() - start:.3f}s")
    return module


config = _import_page("config")
create_database = _import_page("create_database")
labeled_examples = _import_page("labeled_examples")
search = _import_page("search")
explore = _import_page("explore")
train = _import_page("train")
refine = _import_page("refine")
inference = _import_page("inference")
status = _import_page("status")
log = _import_page("log")
print(f"[startup] pages imported {time.time():.3f}")

_LOGO_CACHE: dict[Path, str] = {}


def encode_png_to_base64(path: Path) -> str:
    """Reads a PNG file and returns a base64 encoded data URI."""
    with open(path, "rb") as f:
        binary_data = f.read()
    encoded_data = base64.b64encode(binary_data).decode("utf-8")
    return f"data:image/png;base64,{encoded_data}"


def _get_cached_logo(path: Path) -> str:
    cached = _LOGO_CACHE.get(path)
    if cached:
        return cached
    encoded = encode_png_to_base64(path)
    _LOGO_CACHE[path] = encoded
    return encoded


def build_app(vm):
    """Builds the main application layout using a Panel template."""

    vm.ui = PanelUI(vm)

    def _session_exception_handler(ex):
        logging.error("An unhandled exception occurred", exc_info=ex)
        vm.notify(f"An unexpected error occurred. Please try refreshing the page. {ex}", "error")

    pn.config.exception_handler = _session_exception_handler

    vm.setup_help_system()

    template = pn.template.FastListTemplate(
        title="Recognizer Workshop",
        sidebar_width=400,
        sidebar=[vm.help_system, vm.help_system.content]  # Pass components directly
    )

    # Check for dev mode via query string parameter or environment variable
    # URL would be: http://localhost:5006/?dev=true
    dev_mode = pn.state.session_args.get('dev', [b'false'])[0].decode() == 'true' or \
               os.environ.get('RECOGNIZER_DEV_MODE', '').lower() == 'true'
    
    # Create a toggle for dev mode (hidden by default)
    dev_toggle = pn.widgets.Checkbox(
        name="🔧 Developer Mode", 
        value=dev_mode,
        visible=False  # Hidden by default
    )
    
    # Add keyboard shortcut to show dev toggle: Ctrl+Shift+D
    dev_toggle_button = pn.pane.HTML(f"""
    <script>
    document.addEventListener('keydown', function(e) {{
        if (e.ctrlKey && e.shiftKey && e.key === 'D') {{
            e.preventDefault();
            // Toggle visibility of dev mode checkbox
            var devToggle = document.querySelector('.dev-toggle-container');
            if (devToggle) {{
                devToggle.style.display = devToggle.style.display === 'none' ? 'block' : 'none';
            }}
        }}
    }});
    </script>
    <div class="dev-toggle-container" style="display: {'block' if dev_mode else 'none'}; 
         padding: 10px; 
         background: #f0f0f0; 
         border-radius: 4px; 
         margin: 10px 0;">
        <small>💡 Press <kbd>Ctrl+Shift+D</kbd> to toggle dev mode panel</small>
    </div>
    """)

    # seems to be problematic to do a relative src to the image files in the html
    app_dir = Path(__file__).parent
    qut_logo_path = app_dir / "assets" / "qut_logo_2.png"
    ardc_logo_path = app_dir / "assets" / "ardc_logo_2.png"
    perch_logo_path = app_dir / "assets" / "perch_humpback2.png"
    qut_logo_base64 = _get_cached_logo(qut_logo_path)
    ardc_logo_base64 = _get_cached_logo(ardc_logo_path)
    perch_logo_base64 = _get_cached_logo(perch_logo_path)

    header_logos = pn.pane.HTML(f"""
    <div style="display: flex; align-items: center;  justify-content: space-between; width: 290px; gap: 20px; background: white; padding: 5px 10px; border-radius: 2px;">
        <img src="{qut_logo_base64}" style="height: 40px; width: auto;">
        <img src="{ardc_logo_base64}" style="height: 40px; width: auto;">
        <img src="{perch_logo_base64}" style="height: 40px; width: auto;">
    </div>
    """)
    
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
        ("Logs", log.build(vm)),
        sizing_mode="stretch_width",
        min_height=500,
        scroll=True,
    )



    template.header.append(header_logos)
    template.header.append(dev_toggle_button)
    template.main.append(vm.status)
    template.main.append(tabs)
    # template.sidebar.append(vm.help_system) # append the js custom component for sidebar control
    # template.sidebar.append(vm.help_system.content) # append the actual help content
    return template


def create_app_session():
    """Creates a complete, isolated application session."""
    vm = ViewModel()
    app = build_app(vm)
    return app, vm


def start_app_from_notebook():
    """
    Creates and starts the Panel app from a Jupyter notebook.
    Automatically detects whether we're running via 'panel serve' or interactive Jupyter.
    
    Returns:
        tuple: (app, vm) - The Panel app object and view model
    """

    print(f"--- KERNEL PROCESS ID: {os.getpid()} ---")
    
    # Create the app
    app, vm = create_app_session()
    
    # Better detection logic for how we're being run
    is_bokeh_server = 'bokeh' in sys.modules and hasattr(sys.modules['bokeh'], 'server')
    is_panel_serve = __name__.startswith('bokeh') or 'bokeh_app' in globals()
    
    if is_panel_serve:
        # We're being served by Panel's server (panel serve command)
        print("Running via Panel server")
        print("📱 The app should be available at the server URL provided by Panel serve")
        app.servable(title="Recognizer Workshop")
    else:
        # We're running interactively in Jupyter
        print("Running in Jupyter - starting Panel server...")
        app.show(port=0)
        print("👆 Click the URL above to open your Recognizer Workshop app")
        
        # Also make it servable for flexibility
        app.servable(title="Recognizer Workshop")
    
    return app, vm






# Serve the app for interactive development
if __name__.startswith("bokeh"):
    app, vm = create_app_session()
    app.servable()