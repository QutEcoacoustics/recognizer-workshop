"""Shared helper for creating Panel FileDownload buttons backed by a file path."""
import io
from pathlib import Path
from typing import Callable

import panel as pn


def make_file_download_button(
    get_path: Callable[[], Path | None],
    filename: str = "download",
    label: str = "Download",
    button_type: str = "primary",
    width: int = 130,
) -> pn.widgets.FileDownload:
    """Return a FileDownload widget that reads from the path returned by `get_path` at click time."""

    def _callback():
        path = get_path()
        if path is None or not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return io.BytesIO(path.read_bytes())

    btn = pn.widgets.FileDownload(
        filename=filename,
        label=label,
        button_type=button_type,
        sizing_mode="fixed",
        width=width,
        embed=False,
    )
    btn.callback = _callback
    return btn
