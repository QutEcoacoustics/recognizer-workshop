#!/usr/bin/env python
"""
Python-based launcher for the Panel application.

This script configures the Python path and launches the
server, bypassing the shell and the `panel serve` command's
which had some import path resolution issues.

To run the app, execute this script from your project's root directory:
  python ./scripts/run_server.py
"""
import panel as pn
import sys
from pathlib import Path
import argparse 

print(pn.__version__)

def main():


    parser = argparse.ArgumentParser(description="Run the Panel application server.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (sets autoreload=False and show=True)."
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    src_path = project_root / "packages" / "recognizer-workshop" / "src"

    # Had some problems trying to get panel to find the packages.
    print(f"Adding '{src_path}' to PYTHONPATH")
    sys.path.insert(0, str(src_path))

    try:
        from recognizer_workshop.ui.panel.app import create_app_session
        app, _ = create_app_session()

    except ImportError as e:
        print(f"\nError: Could not import the application.")
        print(f"Please ensure the path '{src_path}' is correct and contains your package.")
        print(f"Original error: {e}")
        sys.exit(1)
        
    print("Successfully imported the app. Starting server...")


    if args.debug:
        print("🚀 Launching in DEBUG mode...")
        serve_kwargs = {
            "autoreload": False,
            "show": True
        }
    else:
        print("Launching in standard mode...")
        serve_kwargs = {
            "autoreload": True,
            "show": False
        }


    # Use Panel's programmatic API to serve the application
    pn.serve(
        app,
        port=8866,
        log_level='debug',
        allow_websocket_origin=['localhost:8866', '127.0.0.1:8866'],
        **serve_kwargs
    )

if __name__ == "__main__":
    main()
