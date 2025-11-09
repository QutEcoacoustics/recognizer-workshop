import logging
from collections import deque
from datetime import datetime
import panel as pn


class PanelLogHandler(logging.Handler):
    """
    A logging handler that displays logs in a Panel Markdown pane.
    Uses a circular buffer to limit memory usage and prevent performance issues.
    """

    def __init__(self, pane: pn.pane.Markdown, max_lines: int = 1000):
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
        """Emit a log record to the Panel pane."""
        try:
            msg = self.format(record)

            # Add emoji based on level
            if record.levelno >= logging.ERROR:
                prefix = "❌"
            elif record.levelno >= logging.WARNING:
                prefix = "⚠️"
            elif record.levelno >= logging.INFO:
                prefix = "ℹ️"
            else:
                prefix = "🔍"

            # Add to circular buffer
            self.log_buffer.append(f"{prefix} {msg}")

            # Update the pane with all buffered lines
            self._update_pane()

        except Exception:
            self.handleError(record)

    def _update_pane(self):
        """Update the pane with current buffer contents."""
        # Join with newlines and wrap in code block for monospace font
        content = "```\n" + "\n".join(self.log_buffer) + "\n```"
        self.pane.object = content

    def clear(self):
        """Clear all logs."""
        self.log_buffer.clear()
        self.pane.object = ""