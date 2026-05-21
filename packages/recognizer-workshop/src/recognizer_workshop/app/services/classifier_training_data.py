import json
from pathlib import Path

import numpy as np
from perch_hoplite.db import interface
from perch_hoplite.db.sqlite_usearch_impl import deserialize_embedding


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    return value


def _offset_as_key(offset):
    if isinstance(offset, list):
        return tuple(offset)
    return offset


def _finalize_grouped_sources(grouped_sources: dict[str, dict]) -> list[dict]:
    out = []
    for source, source_data in sorted(grouped_sources.items(), key=lambda x: x[0]):
        offset_values = sorted(source_data["offsets"].values(), key=lambda x: repr(x))
        out.append({"source": source, "offsets": offset_values})
    return out


def _build_data_from_sql(db, target_labels: list[str]) -> dict:
    grouped = {
        label: {
            "positive": {},
            "negative": {},
        }
        for label in target_labels
    }

    if not target_labels:
        return {label: {"positive": [], "negative": []} for label in target_labels}

    placeholders = ", ".join(["?"] * len(target_labels))
    query = f"""
        SELECT l.label, l.type, s.source, e.offsets
        FROM hoplite_labels l
        JOIN hoplite_embeddings e ON l.embedding_id = e.id
        JOIN hoplite_sources s ON e.source_idx = s.id
        WHERE l.label IN ({placeholders})
    """
    cursor = db._get_cursor()
    cursor.execute(query, tuple(target_labels))

    for label, label_type, source, offsets_blob in cursor.fetchall():
        if label not in grouped:
            continue
        bucket_key = "positive" if int(label_type) == int(interface.LabelType.POSITIVE.value) else "negative"
        source_groups = grouped[label][bucket_key]
        if source not in source_groups:
            source_groups[source] = {"offsets": {}}

        offsets_array = deserialize_embedding(offsets_blob, db.offset_dtype)
        offset_value = _to_jsonable(offsets_array)
        if isinstance(offset_value, list) and len(offset_value) == 1:
            offset_value = offset_value[0]
        source_groups[source]["offsets"][_offset_as_key(offset_value)] = offset_value

    out = {}
    for label, class_data in grouped.items():
        out[label] = {
            "positive": _finalize_grouped_sources(class_data["positive"]),
            "negative": _finalize_grouped_sources(class_data["negative"]),
        }
    return out


def _build_data_from_interface(db, target_labels: list[str]) -> dict:
    out = {}
    label_types = [
        ("positive", interface.LabelType.POSITIVE),
        ("negative", interface.LabelType.NEGATIVE),
    ]
    for label in target_labels:
        out[label] = {"positive": [], "negative": []}
        for bucket_key, label_type in label_types:
            ids = db.get_embeddings_by_label(label, label_type, None)
            if len(ids) == 0:
                continue

            sources = db.get_embedding_sources(ids)
            grouped_sources: dict[str, dict] = {}
            for src in sources:
                if src.source_id not in grouped_sources:
                    grouped_sources[src.source_id] = {"offsets": {}}
                offset_value = _to_jsonable(src.offsets)
                if isinstance(offset_value, list) and len(offset_value) == 1:
                    offset_value = offset_value[0]
                grouped_sources[src.source_id]["offsets"][_offset_as_key(offset_value)] = offset_value

            out[label][bucket_key] = _finalize_grouped_sources(grouped_sources)

    return out


def build_classifier_data_payload(db, target_labels: list[str], train_params: dict | None) -> dict:
    if db is None:
        return {
            "data": {label: {"positive": [], "negative": []} for label in target_labels},
            "train_params": _to_jsonable(train_params or {}),
            "hoplite_metadata": [],
        }

    if hasattr(db, "_get_cursor") and hasattr(db, "offset_dtype"):
        data = _build_data_from_sql(db, target_labels)
    else:
        data = _build_data_from_interface(db, target_labels)

    hoplite_metadata = []
    if hasattr(db, "_get_cursor"):
        cursor = db._get_cursor()
        cursor.execute("SELECT key, data FROM hoplite_metadata ORDER BY key")
        for key, data_json in cursor.fetchall():
            try:
                data_value = json.loads(data_json)
            except Exception:
                data_value = data_json
            hoplite_metadata.append({"key": key, "data": _to_jsonable(data_value)})
    else:
        metadata = db.get_metadata(None)
        for key, value in dict(metadata).items():
            hoplite_metadata.append({"key": key, "data": _to_jsonable(value)})

    payload = {
        "data": data,
        "train_params": _to_jsonable(train_params or {}),
        "hoplite_metadata": hoplite_metadata,
    }
    return payload


def write_classifier_training_data(
    classifier_path: str | Path,
    *,
    db,
    target_labels: list[str],
    train_params: dict | None,
) -> Path:
    classifier_path = Path(classifier_path)
    training_data_path = classifier_path.with_name(f"{classifier_path.stem}.data.json")
    payload = build_classifier_data_payload(db, target_labels, train_params)
    training_data_path.write_text(json.dumps(payload, indent=4))
    return training_data_path
