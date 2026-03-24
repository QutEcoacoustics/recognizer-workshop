import json
import re
from datetime import datetime
from pathlib import Path


def sanitize_classifier_name(name: str | None, default: str = "classifier", max_length: int = 20) -> str:
    candidate = name or ""
    if candidate == "":
        return default
    candidate = re.sub(r"\\s+", "_", candidate)
    candidate = re.sub(r"[^A-Za-z0-9.\\-_()]", "", candidate)
    candidate = re.sub(r"_+", "_", candidate)
    candidate = candidate[:max_length]
    return candidate or default


def make_created_at() -> str:
    dt = datetime.now().astimezone()
    offset = dt.strftime("%z")
    return dt.strftime("%Y%m%dT%H%M%S") + offset[:-2] + ":" + offset[-2:]


def write_classifier_metadata(
    classifier_path: str | Path,
    *,
    name: str | None,
    description: str | None,
    created_at: str | None = None,
) -> dict:
    path = Path(classifier_path)
    data = json.loads(path.read_text())
    data["created_at"] = created_at or make_created_at()
    data["name"] = sanitize_classifier_name(name)
    data["description"] = description if description is not None else ""
    path.write_text(json.dumps(data, indent=4))
    return data
