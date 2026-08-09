import logging
from collections import deque
from pathlib import Path
import panel as pn

from recognizer_workshop.ui.panel.utils.file_download import make_file_download_button

log = logging.getLogger(__name__)


def build(vm):
    """Build the log viewer page."""
    
    # Get the log handler to access its buffer
    #app_logger = logging.getLogger('recognizer_workshop')
    handler = getattr(vm, 'log_handler', None)
    
    # Controls
    btn_clear = pn.widgets.Button(name="Clear Logs", button_type="warning", icon="trash")

    _log_file = Path.home() / ".recognizer-workshop" / "logs" / "app.log"
    btn_download = make_file_download_button(
        get_path=lambda: _log_file,
        filename="app.log",
        label="Download logs",
        button_type="primary",
        width=150,
    )
    
    level_selector = pn.widgets.Select(
        name="Minimum Log Level",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
        value="INFO"
    )
    
    auto_scroll = pn.widgets.Checkbox(name="Auto-scroll to bottom", value=True)
    
    max_lines_input = pn.widgets.IntInput(
        name="Max lines to keep",
        value=1000,
        start=100,
        end=10000,
        step=100
    )
    
    # Info panel
    info_pane = pn.pane.Markdown("")
    
    def _update_info():
        """Update info about current log state."""
        if handler:
            buffer_size = len(handler.log_buffer)
            max_size = handler.max_lines
            info_pane.object = f"**Status:** Showing {buffer_size} of max {max_size} lines"
        else:
            info_pane.object = "**Status:** Log handler not initialized"
    
    def _on_clear(_):
        """Clear the logs."""
        if handler:
            handler.clear()
            log.info("Logs cleared by user")
            _update_info()
    
    def _on_level_change(event):
        """Change the log level."""
        level_name = event.new
        level = getattr(logging, level_name)
        
        # Update the shared handler directly
        if handler:
            handler.setLevel(level)
            log.info(f"Log level changed to {level_name}")
        
    
    def _on_max_lines_change(event):
        """Change the buffer size."""
        new_max = event.new
        if handler:
            # Create new buffer with new size, preserving recent entries
            old_buffer = list(handler.log_buffer)
            handler.log_buffer = deque(old_buffer[-new_max:], maxlen=new_max)
            handler.max_lines = new_max
            handler._update_pane()
            log.info(f"Log buffer size changed to {new_max}")
            _update_info()
    
    # Wire up callbacks
    btn_clear.on_click(_on_clear)
    level_selector.param.watch(_on_level_change, 'value')
    max_lines_input.param.watch(_on_max_lines_change, 'value')
    
    # Initial update
    _update_info()
    
    # The actual log display (this is vm.log_pane from ViewModel)
    log_display = vm.log_pane
    
    # Add some CSS for better log display
    # log_display.styles = {
    #     'background-color': '#f5f5f5',
    #     'border': '1px solid #ddd',
    #     'border-radius': '4px',
    #     'padding': '10px',
    #     'font-family': 'monospace',
    #     'font-size': '12px',
    #     'overflow-y': 'auto'
    # }
    
    return pn.Column(
        
        # Controls row
        pn.Card(
            pn.Row(
                level_selector,
                max_lines_input,
                auto_scroll,
                btn_clear,
                btn_download,
                sizing_mode="stretch_width"
            ),
            pn.Row(info_pane),
        ),
        
        # Log display
        pn.Card(
            log_display,
            sizing_mode="stretch_both"
        ),
        
        sizing_mode="stretch_both"
    )
