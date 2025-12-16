from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import logging
import panel as pn
import param

from recognizer_workshop.ui.panel.help import HelpSystem
from recognizer_workshop.ui.panel.utils.log_handler import PanelLogHandler



# we want some UI elements to be reactive. 
# 1. Generic Observable class to represent a value that ui elements react to
#    when changed
# 2. A hub to hold multiple named observables.
# There is a single observables hub in the view model. Whenever an observable bit of state
# is created or changed, it should also be set in the hub. 
# This works becasue param.Paremeterized objects can be observed with a decorator. 
class Observable(param.Parameterized):
    """A single reactive topic."""
    # value = param.Parameter(default=None, allow_None=True)
    version = param.Integer(default=0)

    def bump(self, v):
        self.version += 1


class ObservablesHub(dict):
    """
    A dict-like container of named Observable objects.
    vm.obs['target_labels'] -> Observable
    """
    def ensure(self, name: str) -> Observable:
        if name not in self:
            self[name] = Observable(name=name)
        return self[name]

    def bump(self, name: str):
        self.ensure(name).bump(None)


# there is nothings stopping new observables from being created on the fly,
# but let's make it clearer by listing them here
_observables_names = [
    "engine_state",  # any change to the engine so that status or other things can update
    "config_state",  # config loaded/changed
    "target_labels", # active label selector updates when this changes
    "active_label", # when the active label is changed, so that the label selector can update the selected item
]


class ViewModel(SimpleNamespace):
    """
    Shared app state:
      eng: agile_state | None
      ui:  PanelUI | None
      status: pn.pane.Markdown
    """

    def __init__(self):
        super().__init__(eng=None, ui=None)
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.log_pane = pn.pane.HTML(
            "",  # Start with empty content
            sizing_mode="stretch_both",
            styles={
                'background-color': '#1e1e1e',  # Dark background
                'color': '#d4d4d4',  # Light text
                'padding': '10px',
                'font-family': '"SF Mono", "Monaco", "Consolas", monospace, "Apple Color Emoji"',
                'font-size': '12px',
                'overflow-y': 'auto',
                'border-radius': '4px'
            }
        )
        self._setup_logging()

        # initialise the observables hub with pre-defined names
        self.obs = ObservablesHub()
        for name in _observables_names:
            self.obs.ensure(name)

    def setup_help_system(self):
        """Initializes the help system with the given template."""
        self.help_system = HelpSystem()




    def _setup_logging(self):
        """Connects the ViewModel's log_view widget to the app's loggers.

        Note: we configure the global loggers to be at INFO level, but we set
        different levels on the handlers to control what gets shown in the UI.
        So that we don't block useful log messages from other handlers.
        """
        
        # Create a single shared handler for both loggers
        shared_panel_handler = PanelLogHandler(pane=self.log_pane, max_lines=2000)
        shared_panel_handler.setLevel(logging.DEBUG)
        
        # Store reference to the handler for easy access
        self.log_handler = shared_panel_handler
        
        # Setup recognizer_workshop logger
        app_logger = logging.getLogger('recognizer_workshop')
        app_logger.setLevel(logging.DEBUG)
        
        if not any(isinstance(h, PanelLogHandler) for h in app_logger.handlers):
            app_logger.addHandler(shared_panel_handler)

        # Setup baw_helpers logger with the same handler
        baw_logger = logging.getLogger('baw_helpers')
        baw_logger.setLevel(logging.INFO)
        
        if not any(isinstance(h, PanelLogHandler) for h in baw_logger.handlers):
            baw_logger.addHandler(shared_panel_handler)




    def _ts(self) -> str:
        # short timestamp for logs
        return datetime.now().strftime("%H:%M:%S")
    

    # unified status helper
    def notify(self, msg: str, kind: str = "info") -> None:
        icon = {"info":"ℹ️","warn":"⚠️","error":"❌","success":"✅"}.get(kind,"ℹ️")
        self.status.object = f"{icon} {msg}"
        log = logging.getLogger(__name__)
        if kind == "error":
            log.error(msg)
        elif kind == "warn":
            log.warning(msg)
        else:
            log.info(msg)

    # mount the adapter view into a container (only when needed)
    def mount_ui_view(self, container: pn.layout.Panel) -> None:

        if not self.ui:
            self.notify("UI adapter not ready. Create engine first.", "warn")
            return
        if getattr(self.ui.view, "parent", None) is not container:
            container.objects = [self.ui.view]

    #TODO: we are not actually calling this anywhere yet
    def reseat_active_label(self):
        """When the target_labels pool changes, ensure eng.active_label is 
        an actual reference to member of the pool of target labels"""

        if not self.eng:
            return
        eng = self.eng
        pool = getattr(eng.config, "target_labels", []) or []
        cur  = getattr(eng, "active_label", None)

        if cur and pool:
            # Find a value-equal label in the new pool & bind to that exact instance
            match = next((ln for ln in pool if ln == cur), None)
            if match is not None and match is not cur:
                eng.active_label = match
                self.obs.bump("active_label")  # notify pickers to update
        elif not cur and pool:
            # Nothing selected yet? default to the first label
            eng.active_label = pool[0]
            self.obs.bump("active_label")
