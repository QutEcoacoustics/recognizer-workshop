import logging
from collections import deque
from datetime import datetime
import panel as pn


class PanelLogHandler(logging.Handler):
    """
    A logging handler that displays logs in a Panel Markdown pane.
    Uses a circular buffer to limit memory usage and prevent performance issues.
    """

    def __init__(self, pane: pn.pane.HTML, max_lines: int = 1000):
        """
        Args:
            pane: The Panel Markdown pane to write logs to
            max_lines: Maximum number of log lines to keep (default 1000)
        """
        super().__init__()
        self.pane = pane
        self.max_lines = max_lines
        self.log_buffer = deque(maxlen=max_lines)  # Circular buffer

        # Set a formatter with timestamp
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with HTML color formatting."""
        try:
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            level = record.levelname
            logger_name = record.name
            message = record.getMessage()

            # Define colors for each log level
            if record.levelno >= logging.ERROR:
                emoji = "❌"
                level_color = "#f44336"  # Red
                level_bg = "#5a1a1a"
            elif record.levelno >= logging.WARNING:
                emoji = "⚠️"
                level_color = "#ff9800"  # Orange
                level_bg = "#4a3a1a"
            elif record.levelno >= logging.INFO:
                emoji = ""
                level_color = "#2196f3"  # Blue
                level_bg = "#1a2a3a"
            else:  # DEBUG
                emoji = ""
                level_color = "#9e9e9e"  # Gray
                level_bg = "#2a2a2a"

            # Build HTML formatted log line
            html_line = (
                f'<div style="margin: 2px 0; padding: 4px; background-color: {level_bg}; border-left: 3px solid {level_color};">'
                f'<span style="color: #808080;">{timestamp}</span> '
                f'<span>{emoji}</span> '
                f'<span style="color: {level_color}; font-weight: bold;">{level:8s}</span> '
                #f'<span style="color: #4ec9b0;">[{logger_name}]</span> '
                f'<span style="color: #d4d4d4;">{message}</span>'
                f'</div>'
            )
 
            self.log_buffer.append(html_line)
            
 
            self._update_pane()
 


        except Exception:
            self.handleError(record)

    def _update_pane(self):
        """Update the pane with current buffer contents."""
        # Join with newlines and wrap in code block for monospace font
        #content = "```\n" + "\n".join(self.log_buffer) + "\n```"
        content = ''.join(self.log_buffer)
        self.pane.object = content

    def clear(self):
        """Clear all logs."""
        self.log_buffer.clear()
        self.pane.object = ""