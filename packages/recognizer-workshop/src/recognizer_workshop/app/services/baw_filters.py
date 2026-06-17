from __future__ import annotations

from typing import Mapping

FILTER_KEYS: tuple[str, ...] = ("projects.id", "regions.id", "sites.id")

DEFAULT_FILTER_DISPLAY_LABELS: dict[str, str] = {
    "projects.id": "Projects",
    "regions.id": "Regions",
    "sites.id": "Sites",
}


def parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    try:
        ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError:
        return None

    return ids or None


def normalize_filter_dict(filter_dict: Mapping | None) -> dict[str, dict[str, list[int]]] | None:
    if not filter_dict:
        return None

    normalized: dict[str, dict[str, list[int]]] = {}
    for key in FILTER_KEYS:
        payload = filter_dict.get(key)
        if not isinstance(payload, Mapping):
            continue

        in_values = payload.get("in")
        if in_values is None:
            continue

        try:
            ids = [int(item) for item in in_values]
        except (TypeError, ValueError):
            continue

        if ids:
            normalized[key] = {"in": ids}

    return normalized or None


def normalize_first_filter(filters: list[Mapping] | None) -> dict[str, dict[str, list[int]]] | None:
    if not filters:
        return None
    first = filters[0]
    if not isinstance(first, Mapping):
        return None
    return normalize_filter_dict(first)


def format_filter_params_for_display(
    filter_params: Mapping | None,
    fallback_name: str = "Filter",
    *,
    key_labels: Mapping[str, str] | None = None,
    delimiter: str = " + ",
) -> str:
    """Convert filter parameters into a human-readable text summary.

    This helper is intentionally UI-agnostic so it can be shared by Panel views,
    CLI output, and logs.
    """
    normalized = normalize_filter_dict(filter_params)
    if not normalized:
        return fallback_name

    labels = key_labels or DEFAULT_FILTER_DISPLAY_LABELS
    parts: list[str] = []

    for key in FILTER_KEYS:
        payload = normalized.get(key)
        if not payload:
            continue
        ids = payload.get("in") or []
        ids_str = ", ".join(map(str, ids))
        parts.append(f"{labels.get(key, key)}: {ids_str}")

    return delimiter.join(parts) if parts else fallback_name
