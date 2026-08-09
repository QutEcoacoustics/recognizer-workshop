
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


def _global_exception_handler(ex):
    logging.error("An unhandled exception occurred", exc_info=ex)
    try:
        if pn.state.notifications is not None:
            pn.state.notifications.error(
                "An unexpected error occurred. Please try refreshing the page."
            )
    except Exception:
        # Never fail while reporting an error
        pass


def _install_global_exception_handler():
    if getattr(_install_global_exception_handler, "_installed", False):
        return
    pn.config.exception_handler = _global_exception_handler
    _install_global_exception_handler._installed = True


_install_global_exception_handler()

def _import_page(name: str):
    start = time.time()
    module = importlib.import_module(f"recognizer_workshop.ui.panel.pages.{name}")
    print(f"[startup] imported pages.{name} {time.time() - start:.3f}s")
    return module


config = _import_page("config")
getting_started = _import_page("getting_started")
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


def build_app(vm):
    """Builds the main application layout using a Panel template."""

    vm.ui = PanelUI(vm)
    vm.setup_help_system()
    vm.debug_mode = pn.state.session_args.get('debug', [b'false'])[0].decode() == 'true'

    template = pn.template.FastListTemplate(
        title="Recognizer Workshop",
        header_background="#79bdd1",
        sidebar_width=400,
        sidebar=[vm.help_system, vm.help_system.content]
    )

    # Tab label → URL slug mapping (order must match the pn.Tabs definition below).
    _TAB_SLUGS = [
        "getting-started",
        "config",
        "database",
        "labelled-examples",
        "search",
        "explore",
        "train",
        "refine",
        "inference",
        "status",
        "logs",
    ]
    _slug_to_index = {slug: i for i, slug in enumerate(_TAB_SLUGS)}

    tabs = pn.Tabs(
        ("Getting started", getting_started.build(vm)),
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
        dynamic=True,
    )

    # Restore active tab from URL on load.
    initial_slug = pn.state.session_args.get('tab', [b''])[0].decode()
    if initial_slug in _slug_to_index:
        tabs.active = _slug_to_index[initial_slug]

    # Push the active tab slug into the URL (replaceState — no extra browser history entries).
    slugs_js = str(_TAB_SLUGS)
    tabs.jscallback(active=f"""
        const slugs = {slugs_js};
        const slug = slugs[cb_obj.active] || String(cb_obj.active);
        const url = new URL(window.location.href);
        url.searchParams.set('tab', slug);
        window.history.replaceState({{}}, '', url.toString());
    """)

    # Tab indices that require a loaded database (0-indexed):
    # 0=Getting started, 1=Config, 2=Database, 3=Labelled examples,
    # 4=Search, 5=Explore, 6=Train, 7=Refine, 8=Inference
    _DB_REQUIRED_TABS = [3, 4, 5, 6, 7, 8]

    @pn.depends(vm.obs.ensure("engine_state").param.version, watch=True)
    def _update_disabled_tabs(_=None):
        tabs.disabled_tabs = [] if (vm.eng and vm.eng.is_db_loaded) else _DB_REQUIRED_TABS

    _update_disabled_tabs()

    template.header.append(vm.status)
    template.main.append(tabs)
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