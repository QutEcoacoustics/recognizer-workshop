import panel as pn
import logging

class PanelLogHandler(logging.Handler):
    """A custom logging handler that sends logs to a Panel Markdown pane."""

    def __init__(self, pane: pn.pane.Markdown, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pane = pane
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
        self.setFormatter(formatter)

    def emit(self, record):
        """
        This method is called for every log record. It updates the pane's content.
        """
        msg = self.format(record)
        # For Markdown, we convert newlines to <br> and update the .object
        new_content = msg.replace('\n', '<br>')
        current_content = self.pane.object or ""
        self.pane.object = f"{new_content}<br>{current_content}"