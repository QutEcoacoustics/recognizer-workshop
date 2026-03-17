from __future__ import annotations

from typing import Mapping

FILTER_KEYS: tuple[str, ...] = ("projects.id", "regions.id", "sites.id")


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
