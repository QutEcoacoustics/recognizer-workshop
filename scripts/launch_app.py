"""
This is a launcher script for the Panel application.

Running `panel serve launch_app.py` from within the `src` directory
allows the server to discover and serve the main `app` object
from the actual application module, sidestepping complex import issues.
"""
from recognizer_workshop.ui.panel.app import app

# By importing `app` and calling .servable() in the global scope of this
# script, we make it the discoverable object for the `panel serve` command.
# We give it a title here to be explicit.
app.servable(title="Recognizer Workshop")
