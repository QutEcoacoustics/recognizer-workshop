from pathlib import Path
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files
import logging
import panel as pn


def ppath(p: str) -> Path:
    return Path(p).expanduser().absolute()

def resolve_query_uri(query: str, labeled_dir: Path) -> str:
    #TODO: this is already done somewhere else. replace references to this one, and then delete this one
    q = str(query).strip()
    if q.isdigit():
        files = list_audio_files(labeled_dir, quiet=True)
        idx = int(q)
        if 0 <= idx < len(files):
            return str(files[idx])
        raise ValueError(f"Index {idx} out of range (0..{len(files)-1})")
    p = labeled_dir / q
    if p.exists():
        return str(p)
    return q  # URL / absolute / XC id passthrough

def parse_target_score(s: str):
    s = s.strip()
    if s.lower() == "none":
        return None
    return float(s)




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