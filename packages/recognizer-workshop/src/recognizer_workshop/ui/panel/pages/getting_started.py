from pathlib import Path
import panel as pn


def build(vm):
    html_path = Path(__file__).parent / "getting_started_content.html"
    try:
        content = html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = "<p>Getting started content not found.</p>"

    return pn.pane.HTML(content, sizing_mode="stretch_width")
