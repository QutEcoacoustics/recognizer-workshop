import logging
import panel as pn

log = logging.getLogger(__name__)


def build(vm):
    """Build the log viewer page."""
    
    # Get the log handler to access its buffer
    app_logger = logging.getLogger('recognizer_workshop')
    handler = None
    for h in app_logger.handlers:
        if hasattr(h, 'log_buffer'):  # Our custom PanelLogHandler
            handler = h
            break
    
    # Controls
    btn_clear = pn.widgets.Button(name="Clear Logs", button_type="warning", icon="trash")
    btn_download = pn.widgets.Button(name="Download Logs", button_type="primary", icon="download")
    
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
    
    def _on_download(_):
        """Download logs as text file."""
        if handler and handler.log_buffer:
            content = "\n".join(handler.log_buffer)
            # Create a downloadable file
            from datetime import datetime
            filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            # Use Panel's FileDownload widget
            file_download = pn.widgets.FileDownload(
                callback=lambda: content.encode('utf-8'),
                filename=filename,
                button_type="success",
                auto=True  # Auto-trigger download
            )
            # Trigger it
            vm.notify(f"Downloading {filename}", "success")
        else:
            vm.notify("No logs to download", "warn")
    
    def _on_level_change(event):
        """Change the log level."""
        level_name = event.new
        level = getattr(logging, level_name)
        
        # Update all our handlers
        for logger_name in ['recognizer_workshop', 'baw_helpers']:
            logger = logging.getLogger(logger_name)
            for h in logger.handlers:
                if hasattr(h, 'log_buffer'):
                    h.setLevel(level)
        
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
    btn_download.on_click(_on_download)
    level_selector.param.watch(_on_level_change, 'value')
    max_lines_input.param.watch(_on_max_lines_change, 'value')
    
    # Initial update
    _update_info()
    
    # The actual log display (this is vm.log_pane from ViewModel)
    log_display = vm.log_pane
    
    # Add some CSS for better log display
    log_display.styles = {
        'background-color': '#f5f5f5',
        'border': '1px solid #ddd',
        'border-radius': '4px',
        'padding': '10px',
        'font-family': 'monospace',
        'font-size': '12px',
        'overflow-y': 'auto'
    }
    
    return pn.Column(
        "## Application Logs",
        pn.pane.Alert(
            "This page shows real-time logs from the application. "
            "Logs are limited to the most recent entries to prevent performance issues.",
            alert_type="info"
        ),
        
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
            title="Log Controls"
        ),
        
        # Log display
        pn.Card(
            log_display,
            title="Log Output",
            sizing_mode="stretch_both",
            height=600
        ),
        
        sizing_mode="stretch_both"
    )
