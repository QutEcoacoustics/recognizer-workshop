"""
Tests for repair_index — rebuilding the usearch index from SQLite + parquet re-download.

The integration test uses the existing test DB (ARID 4207561, 500 embeddings),
saves the original vectors, deletes the index, rebuilds it, and verifies
that every embedding vector matches the original.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from perch_hoplite.db import sqlite_usearch_impl

from recognizer_workshop.app.create_database.repair_index import (
    repair_index,
    _get_source_embedding_map,
    _get_arids_from_sources,
)


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

    stats = repair_index(db, prod_api_client, progress_callback=capture_progress)

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

    stats = repair_index(db, prod_api_client)

    # All 500 should be skipped since they already exist in the index
    assert stats["embeddings_skipped"] == 500
    assert stats["embeddings_added"] == 0
    assert stats["errors"] == 0
