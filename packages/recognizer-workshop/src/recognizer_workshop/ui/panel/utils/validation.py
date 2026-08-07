import panel as pn


INVALID_FIELD_CLASS = "rw-invalid-field"
_INVALID_FIELD_CSS = (
    ".rw-invalid-field .bk-input { "
    "border-color: #dc2626 !important; "
    "box-shadow: 0 0 0 1px #dc2626 inset !important; "
    "}"
)
_css_registered = False


def ensure_validation_css() -> None:
    global _css_registered

    if _css_registered:
        return

    pn.config.raw_css.append(_INVALID_FIELD_CSS)
    _css_registered = True


def set_valid(widget) -> None:
    css_classes = [css_class for css_class in (widget.css_classes or []) if css_class != INVALID_FIELD_CLASS]
    widget.css_classes = css_classes


def set_invalid(widget) -> None:
    css_classes = list(widget.css_classes or [])
    if INVALID_FIELD_CLASS not in css_classes:
        css_classes.append(INVALID_FIELD_CLASS)
    widget.css_classes = css_classes


def set_validity(widget, *, is_valid: bool) -> None:
    if is_valid:
        set_valid(widget)
        return

    set_invalid(widget)