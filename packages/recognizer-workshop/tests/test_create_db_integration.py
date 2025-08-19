# tests/test_create_db_integration.py
import json
from pathlib import Path
import pytest

from perch_hoplite.db import sqlite_usearch_impl


from recognizer_workshop.create_database.create_database import create_database


@pytest.mark.integration
def test_create_db_from_project_1189(prod_api_client, tmp_path):
    """End-to-end: pull a few ARIDs from project 1189, build a tiny DB, and check artifacts."""
    db_dir = tmp_path / "db"
    cache_dir = tmp_path / "filelist"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Keep test fast & cheap
    max_files = 3
    max_embeddings = 500

    # Run the pipeline
    create_database(
        db_path=db_dir,
        filelist_path=cache_dir,
        client=prod_api_client,
        project_ids=1189,
        max_embeddings=max_embeddings,
        max_files=max_files,
    )

    # 1) DB was created
    db_file = db_dir / sqlite_usearch_impl.HOPLITE_FILENAME
    assert db_file.exists(), f"Expected DB file at {db_file}"

    # 2) DB has some embeddings
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_dir)
    cnt = db.count_embeddings()
    assert cnt > 0, "Expected >0 embeddings inserted"

    # 3) A cache JSON was written for the audio_recordings filter
    jsons = list(cache_dir.glob("audio_recordings_*.json"))
    assert jsons, f"Expected at least one cached filter result in {cache_dir}"
    # (Optional) sanity: file parses and has 'data' list
    with jsons[0].open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list)
