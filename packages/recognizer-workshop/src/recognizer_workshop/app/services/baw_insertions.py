"""
Service for reading and writing the `baw_insertions` DB metadata key.

Structure stored under key 'baw_insertions':
{
  "datasets": {
    "search_set": {
      "insertions": [
        {"filter": {...}, "embeddings_added": 1234},
        ...
      ]
    },
    "fetched_labels": {
      "insertions": [
        {"filter": {...}, "embeddings_added": 45},
        ...
      ]
    }
  }
}

Each time a batch of embeddings is added for a given filter, the matching insertion
entry's `embeddings_added` count is incremented.  If no entry with an equivalent
filter exists, a new entry is appended.
"""

import logging

from ml_collections import config_dict

log = logging.getLogger(__name__)


def _normalize_filter(filter_dict: dict) -> dict:
    """Normalize a filter dict for semantic comparison.

    Sorts the top-level keys and sorts any ``in`` list values so that
    ``{"projects.id": {"in": [2, 1]}}`` compares equal to
    ``{"projects.id": {"in": [1, 2]}}``.
    """
    return {
        k: {"in": sorted(v["in"])} if isinstance(v, dict) and "in" in v else v
        for k, v in sorted(filter_dict.items())
    }


def get_baw_insertions(db) -> dict:
    """Return the baw_insertions log as a plain dict, or an empty structure."""
    try:
        cd = db.get_metadata('baw_insertions')
        return cd.to_dict()
    except KeyError:
        return {"datasets": {}}


def get_last_filter_for_dataset(db, dataset_name: str, fallback_dataset: str | None = None) -> dict | None:
    """Return the filter dict from the most recent insertion for *dataset_name*.

    If no insertion exists for that dataset, optionally fall back to
    *fallback_dataset*.  Returns ``None`` if nothing is found.
    """
    log_data = get_baw_insertions(db)
    datasets = log_data.get("datasets", {})

    insertions = datasets.get(dataset_name, {}).get("insertions", [])
    if insertions:
        return insertions[-1]["filter"]

    if fallback_dataset:
        fallback_insertions = datasets.get(fallback_dataset, {}).get("insertions", [])
        if fallback_insertions:
            return fallback_insertions[-1]["filter"]

    return None


def record_embeddings_added(db, dataset_name: str, filter_dict: dict, count: int) -> None:
    """Increment the ``embeddings_added`` tally for *filter_dict* in *dataset_name*.

    If an entry with a semantically equivalent filter already exists it is
    updated in place; otherwise a new entry is appended.  The metadata is
    written and committed to the DB before returning.
    """
    log_data = get_baw_insertions(db)
    datasets = log_data.setdefault("datasets", {})
    if dataset_name not in datasets:
        datasets[dataset_name] = {"insertions": []}
    insertions = datasets[dataset_name].setdefault("insertions", [])

    normalized = _normalize_filter(filter_dict)
    matched = False
    for insertion in insertions:
        if _normalize_filter(insertion.get("filter", {})) == normalized:
            insertion["embeddings_added"] = insertion.get("embeddings_added", 0) + count
            matched = True
            break

    if not matched:
        insertions.append({"filter": filter_dict, "embeddings_added": count})

    db.insert_metadata('baw_insertions', config_dict.ConfigDict(log_data))
    db.commit()
    log.debug("Recorded %d embeddings added to dataset '%s'", count, dataset_name)
