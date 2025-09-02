from __future__ import annotations
from types import SimpleNamespace
from datetime import datetime
import panel as pn
import logging
from recognizer_workshop.ui.panel.utils import PanelLogHandler

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
        self.log_pane = pn.pane.Markdown(
            "",  # Start with empty content
            height=220,
            sizing_mode="stretch_width",
            css_classes=['log-pane']  # A class for clean CSS styling
        )
        self._setup_logging()

    def _setup_logging(self):
        """Connects the ViewModel's log_view widget to the app's logger."""
        # Get the root logger for our library
        library_logger = logging.getLogger('recognizer_workshop')
        library_logger.setLevel(logging.INFO)

        # Create an instance of our custom handler, pointing it to our widget
        panel_handler = PanelLogHandler(pane=self.log_pane)
        
        # Add the handler if one like it doesn't already exist
        if not any(isinstance(h, PanelLogHandler) for h in library_logger.handlers):
            library_logger.addHandler(panel_handler)

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
