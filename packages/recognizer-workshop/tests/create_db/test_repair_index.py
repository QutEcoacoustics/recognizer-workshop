"""
Tests for repair_index — rebuilding the usearch index from SQLite + parquet re-download.

The integration test uses the existing test DB (ARID 4207561, 500 embeddings),
saves the original vectors, deletes the index, rebuilds it, and verifies
that every embedding vector matches the original.
"""

import base64
import io
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perch_hoplite.db import sqlite_usearch_impl
import recognizer_workshop.app.create_database.repair_index as repair_index_module

from recognizer_workshop.app.create_database.repair_index import (
    repair_index,
    _get_source_embedding_map,
    _get_arids_from_sources,
)

PROD_URL = os.getenv("PROD_API_BASE_URL", "https://api.ecosounds.org")


# ---------------------------------------------------------------------------
# Unit tests (no API needed)
# ---------------------------------------------------------------------------


class TestGetSourceEmbeddingMap:
    def test_builds_map_from_test_db(self, db_with_path):
        db, _ = db_with_path
        source_map = _get_source_embedding_map(db)

        # Test DB has 1 source with 500 embeddings
        assert len(source_map) == 1
        source_url = list(source_map.keys())[0]
        assert "4207561" in source_url

        offsets = source_map[source_url]
        assert len(offsets) == 500
        # Each offset maps to a unique embedding ID
        assert len(set(offsets.values())) == 500


class TestGetAridsFromSources:
    def test_extracts_arids(self):
        urls = [
            "https://api.ecosounds.org/audio_recordings/12345/original",
            "https://api.ecosounds.org/audio_recordings/67890/original",
        ]
        items = _get_arids_from_sources(urls)
        assert len(items) == 2
        arids = {item['id'] for item in items}
        assert arids == {12345, 67890}

    def test_deduplicates(self):
        urls = [
            "https://api.ecosounds.org/audio_recordings/12345/original",
            "https://api.ecosounds.org/audio_recordings/12345/original",
        ]
        items = _get_arids_from_sources(urls)
        assert len(items) == 1


@pytest.mark.parametrize("embedding_format", ["serialized", "columns"])
def test_repair_index_restores_embedding_formats(
    embedding_format,
    test_db_path,
    mock_api_client,
    monkeypatch,
):
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(test_db_path)
    source_offsets = _get_source_embedding_map(db)
    source_url, offsets = next(iter(source_offsets.items()))
    expected_offset, expected_id = next(iter(offsets.items()))
    expected_embedding = np.linspace(0.01, 0.02, db.embedding_dim, dtype=np.float32)

    parquet_columns = {
        "source": [source_url],
        "channel": [0],
        "offset": [expected_offset],
    }
    if embedding_format == "serialized":
        parquet_columns["embeddings"] = [
            base64.b64encode(expected_embedding.tobytes()).decode("ascii")
        ]
    else:
        parquet_columns.update({
            f"f{feature_index:04d}": [value]
            for feature_index, value in enumerate(expected_embedding)
        })

    parquet_buffer = io.BytesIO()
    pq.write_table(pa.table(parquet_columns), parquet_buffer)
    parquet_buffer.seek(0)
    mock_api_client.analysis_result_file.return_value = parquet_buffer

    def identity_source(source_value, domain):
        _ = domain
        return source_value

    monkeypatch.setattr(
        repair_index_module,
        "convert_source_to_original_url",
        identity_source,
    )

    domain = _get_arids_from_sources([source_url])[0]["domain"]
    index_path = Path(test_db_path) / "usearch.index"
    del db
    index_path.unlink()
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(test_db_path)

    stats = repair_index(db, {domain: mock_api_client})

    assert stats["embeddings_added"] == 1
    assert stats["errors"] == 0
    assert db.ui.contains(expected_id)
    np.testing.assert_array_equal(
        db.ui.get(expected_id),
        expected_embedding.astype(np.float16),
    )


# ---------------------------------------------------------------------------
# Integration test — requires prod API token
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_repair_index_roundtrip(prod_api_client, test_db_path):
    """
    Full roundtrip: open DB, snapshot all vectors, delete the index file,
    rebuild via repair_index, verify all vectors match originals.
    """
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(test_db_path)
    embedding_ids = db.get_embedding_ids()
    assert len(embedding_ids) == 500, f"Expected 500 embeddings, got {len(embedding_ids)}"

    # 1) Snapshot every embedding vector from the original index
    original_vectors = {}
    for eid in embedding_ids:
        original_vectors[int(eid)] = np.array(db.ui.get(eid))

    # 2) Close the index and delete the file
    index_path = Path(test_db_path) / "usearch.index"
    assert index_path.exists()
    # usearch memory-maps the file; delete it and re-open the DB
    del db

    os.remove(index_path)
    assert not index_path.exists()

    # 3) Re-open DB — this will create a fresh empty index
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(test_db_path)
    assert db.ui.size == 0, "Index should be empty after deletion"

    # 4) Repair
    progress_events = []
    def capture_progress(event):
        progress_events.append(event)

    stats = repair_index(db, {PROD_URL: prod_api_client}, progress_callback=capture_progress)

    assert stats["embeddings_added"] == 500
    assert stats["errors"] == 0

    # 5) Verify all vectors match
    rebuilt_ids = db.get_embedding_ids()
    assert len(rebuilt_ids) == 500

    mismatches = 0
    for eid in embedding_ids:
        eid = int(eid)
        assert db.ui.contains(eid), f"Embedding {eid} missing from rebuilt index"
        rebuilt_vec = np.array(db.ui.get(eid))
        original_vec = original_vectors[eid]
        if not np.allclose(rebuilt_vec, original_vec, atol=1e-6):
            mismatches += 1

    assert mismatches == 0, f"{mismatches} embedding vectors differ after rebuild"

    # 6) Verify progress callback was called
    assert len(progress_events) >= 2  # at least "starting" and "complete"
    assert progress_events[-1]["stage"] == "complete"


@pytest.mark.integration
def test_repair_index_idempotent(prod_api_client, test_db_path):
    """Running repair on an already-complete index should skip all embeddings."""
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(test_db_path)
    assert db.ui.size == 500

    stats = repair_index(db, {PROD_URL: prod_api_client})

    # All 500 should be skipped since they already exist in the index
    assert stats["embeddings_skipped"] == 500
    assert stats["embeddings_added"] == 0
    assert stats["errors"] == 0
