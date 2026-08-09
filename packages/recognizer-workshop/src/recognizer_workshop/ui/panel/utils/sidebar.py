"""Sidebar customisations: ? icon and open/closed state persistence."""
from pathlib import Path

import panel as pn

_ASSETS = Path(__file__).parent.parent / "assets"


def build_sidebar_header_component() -> pn.pane.HTML:
    """Inject sidebar CSS and return a Panel element (for the template header) containing the JS."""
    css = (_ASSETS / "css" / "sidebar.css").read_text(encoding="utf-8")
    js = (_ASSETS / "js" / "sidebar_state.js").read_text(encoding="utf-8")

    pn.config.raw_css.append(css)
    return pn.pane.HTML(f"<script>{js}</script>")
