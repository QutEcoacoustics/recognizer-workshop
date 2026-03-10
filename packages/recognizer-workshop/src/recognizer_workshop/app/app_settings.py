import json
import os
from pathlib import Path
from typing import Any, Dict
from baw_helpers.settings import baw_settings
import threading

class AppSettings:
    def __init__(self):

        self._lock = threading.Lock()
        
        # default settings
        self._settings = {

            # which workbenches to display as options for the user to use
            "enabled_workbenches": ["ecosounds", "a2o"], 
            "default_workbench": "ecosounds"
        }
        self.load_config()

    def load_config(self, config_path: str = None):
        path = config_path or os.getenv("RW_APP_SETTINGS")
        
        overrides = {}
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                overrides = json.load(f)

        with self._lock:
            if overrides:
                self._settings.update(overrides)
            baw_settings.update(self._settings)

    def __getattr__(self, name: str) -> Any:
        if name not in self._settings:
            raise AttributeError(f"AppSettings has no setting named '{name}'")
        return self._settings.get(name)

# Singleton instance
app_settings = AppSettings()