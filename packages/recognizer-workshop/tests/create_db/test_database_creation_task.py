"""
Tests for DatabaseCreationTask and format_progress_event.

These tests are framework-agnostic (no Panel imports).
"""

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recognizer_workshop.app.create_database.database_creation_task import (
    DatabaseCreationTask,
    format_progress_event,
)
from recognizer_workshop.app.create_database.create_database import (
    add_parquet_to_db,
    create_empty_db,
)


# ---------------------------------------------------------------------------
# Helpers (shared with test_create_db.py but kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _make_df(arid: int, n: int = 3, emb_dim: int = 2) -> pd.DataFrame:
    base = f"https://api.ecosounds.org/audio_recordings/{arid}/original"
    data = {
        "source": [base] * n,
        "channel": [0] * n,
        "offset": [float(i) for i in range(n)],
    }
    for j in range(emb_dim):
        data[f"f{j}"] = [float(arid + i + j / 10.0) for i in range(n)]
    return pd.DataFrame(data)


def _write_parquets(dirpath: Path, arids, rows_per_file: int = 3, emb_dim: int = 2):
    for idx, arid in enumerate(arids):
        df = _make_df(arid, n=rows_per_file, emb_dim=emb_dim)
        p = dirpath / f"file_{idx}_{arid}.parquet"
        pq.write_table(pa.Table.from_pandas(df), p)


# ---------------------------------------------------------------------------
# format_progress_event — pure function, no I/O needed
# ---------------------------------------------------------------------------

class TestFormatProgressEvent:

    def test_unknown_stage_returns_empty(self):
        msgs, pct = format_progress_event({"stage": "nonexistent"})
        assert msgs == []
        assert pct is None

    def test_empty_event_returns_empty(self):
        msgs, pct = format_progress_event({})
        assert msgs == []
        assert pct is None

    def test_selection_start_cache_mode(self):
        msgs, pct = format_progress_event({
            "stage": "selection_start", "max_files": 100, "use_cache": True
        })
        assert len(msgs) == 1
        assert "cache" in msgs[0]
        assert "100" in msgs[0]
        assert pct is None

    def test_selection_start_fresh_mode(self):
        msgs, pct = format_progress_event({
            "stage": "selection_start", "max_files": 50, "use_cache": False
        })
        assert "fresh fetch" in msgs[0]

    def test_selection_complete(self):
        msgs, pct = format_progress_event({"stage": "selection_complete", "selected_recordings": 42})
        assert "42" in msgs[0]
        assert pct is None

    def test_ingest_start_with_total(self):
        msgs, pct = format_progress_event({
            "stage": "ingest_start", "total_candidates": 10,
            "max_count": 200, "current_embeddings": 50,
        })
        assert any("10" in m for m in msgs)
        assert any("50" in m and "200" in m for m in msgs)
        assert pct is None

    def test_ingest_start_without_total(self):
        msgs, pct = format_progress_event({"stage": "ingest_start"})
        assert len(msgs) == 1
        assert "Starting ingestion" in msgs[0]

    def test_ingest_start_omits_limit_line_when_max_count_zero(self):
        msgs, pct = format_progress_event({
            "stage": "ingest_start", "total_candidates": 5,
            "max_count": 0, "current_embeddings": 0,
        })
        assert len(msgs) == 1

    def test_source_progress_with_total_gives_percent(self):
        msgs, pct = format_progress_event({
            "stage": "source_progress",
            "candidates_seen": 5, "total_candidates": 10,
            "skipped_existing": 1, "files_processed": 4,
        })
        assert pct == 50
        assert "5/10" in msgs[0]
        assert "4" in msgs[0]    # files_processed
        assert "1" in msgs[0]    # skipped

    def test_source_progress_without_total_no_percent(self):
        msgs, pct = format_progress_event({
            "stage": "source_progress",
            "candidates_seen": 7, "total_candidates": 0,
            "skipped_existing": 0, "files_processed": 7,
        })
        assert pct is None
        assert "7" in msgs[0]

    def test_source_progress_percent_capped_at_100(self):
        # candidates_seen > total_candidates (shouldn't happen but must not crash)
        msgs, pct = format_progress_event({
            "stage": "source_progress",
            "candidates_seen": 15, "total_candidates": 10,
            "skipped_existing": 0, "files_processed": 10,
        })
        assert pct == 100

    def test_ingest_progress_with_total_gives_percent(self):
        msgs, pct = format_progress_event({
            "stage": "ingest_progress",
            "files_processed": 3, "total_candidates": 6,
            "added_this_file": 5, "total_inserted": 15,
        })
        assert pct == 50
        assert "3/6" in msgs[0]

    def test_ingest_progress_without_total_no_percent(self):
        msgs, pct = format_progress_event({
            "stage": "ingest_progress",
            "files_processed": 3, "total_candidates": None,
            "added_this_file": 5, "total_inserted": 15,
        })
        assert pct is None

    def test_ingest_complete_normal(self):
        msgs, pct = format_progress_event({
            "stage": "ingest_complete",
            "files_processed": 10, "total_inserted": 300,
            "final_embeddings": 300,
        })
        assert pct is None
        assert "10" in msgs[0]
        assert "300" in msgs[0]

    def test_ingest_complete_user_stopped(self):
        msgs, pct = format_progress_event({
            "stage": "ingest_complete",
            "stop_reason": "user_stopped",
            "files_processed": 4, "total_inserted": 100,
        })
        assert pct is None
        assert "Stopped" in msgs[0] or "stopped" in msgs[0]
        assert "4" in msgs[0]
        assert "100" in msgs[0]

    def test_ingest_complete_max_count_already_reached(self):
        msgs, pct = format_progress_event({
            "stage": "ingest_complete",
            "stop_reason": "max_count_already_reached",
            "final_embeddings": 500,
        })
        assert pct is None
        assert "500" in msgs[0]
        assert "limit" in msgs[0].lower() or "Skipped" in msgs[0]


# ---------------------------------------------------------------------------
# add_parquet_to_db — stop_event integration
# ---------------------------------------------------------------------------

class TestAddParquetToDbStopEvent:

    def test_stop_event_halts_ingestion_between_files(self, tmp_path):
        """
        Pre-set the stop_event before calling add_parquet_to_db.
        The loop should see the flag on the first iteration and break
        immediately, inserting 0 rows.
        """
        data_dir = tmp_path / "parquet"
        data_dir.mkdir()
        _write_parquets(data_dir, arids=[111, 222, 333], rows_per_file=3, emb_dim=2)

        db_dir = tmp_path / "db"
        db, _ = create_empty_db(db_path=db_dir, parquet_source=str(data_dir))

        stop = threading.Event()
        stop.set()  # signal stop before any work begins

        add_parquet_to_db(
            db=db,
            parquet_source=str(data_dir),
            dataset_name="search_set",
            source_map_fn=lambda s: s,
            stop_event=stop,
        )

        assert db.count_embeddings() == 0

    def test_stop_event_progress_callback_reports_user_stopped(self, tmp_path):
        """ingest_complete event must carry stop_reason='user_stopped' when halted."""
        data_dir = tmp_path / "parquet"
        data_dir.mkdir()
        _write_parquets(data_dir, arids=[111, 222], rows_per_file=3, emb_dim=2)

        db_dir = tmp_path / "db"
        db, _ = create_empty_db(db_path=db_dir, parquet_source=str(data_dir))

        stop = threading.Event()
        stop.set()

        complete_events = []

        def callback(event):
            if event.get("stage") == "ingest_complete":
                complete_events.append(event)

        add_parquet_to_db(
            db=db,
            parquet_source=str(data_dir),
            dataset_name="search_set",
            source_map_fn=lambda s: s,
            stop_event=stop,
            progress_callback=callback,
        )

        assert len(complete_events) == 1
        assert complete_events[0]["stop_reason"] == "user_stopped"

    def test_no_stop_event_progress_callback_reports_completed(self, tmp_path):
        """Without a stop_event, stop_reason should be 'completed'."""
        data_dir = tmp_path / "parquet"
        data_dir.mkdir()
        _write_parquets(data_dir, arids=[111], rows_per_file=3, emb_dim=2)

        db_dir = tmp_path / "db"
        db, _ = create_empty_db(db_path=db_dir, parquet_source=str(data_dir))

        complete_events = []

        def callback(event):
            if event.get("stage") == "ingest_complete":
                complete_events.append(event)

        add_parquet_to_db(
            db=db,
            parquet_source=str(data_dir),
            dataset_name="search_set",
            source_map_fn=lambda s: s,
            progress_callback=callback,
        )

        assert complete_events[0]["stop_reason"] == "completed"

    def test_db_committed_after_stop(self, tmp_path):
        """Even when stopped, db.commit() must have run (data is durable)."""
        data_dir = tmp_path / "parquet"
        data_dir.mkdir()
        # Only one file so the stop triggers after it's already processed
        _write_parquets(data_dir, arids=[111], rows_per_file=3, emb_dim=2)

        db_dir = tmp_path / "db"
        db, _ = create_empty_db(db_path=db_dir, parquet_source=str(data_dir))

        stop = threading.Event()

        files_seen = []

        original_add_df = __import__(
            "recognizer_workshop.app.create_database.create_database",
            fromlist=["add_df_to_db"],
        ).add_df_to_db

        import recognizer_workshop.app.create_database.create_database as cd_module

        def capturing_add_df(df, db, **kwargs):
            files_seen.append(len(df))
            stop.set()  # request stop after first file is inserted
            return original_add_df(df, db, **kwargs)

        cd_module.add_df_to_db = capturing_add_df
        try:
            add_parquet_to_db(
                db=db,
                parquet_source=str(data_dir),
                dataset_name="search_set",
                source_map_fn=lambda s: s,
                stop_event=stop,
            )
        finally:
            cd_module.add_df_to_db = original_add_df

        # The first file was inserted before stop was honoured
        assert db.count_embeddings() == 3


# ---------------------------------------------------------------------------
# DatabaseCreationTask
# ---------------------------------------------------------------------------

class TestDatabaseCreationTask:

    def test_was_stopped_false_before_run(self, tmp_path):
        task = DatabaseCreationTask(
            db_path=tmp_path / "db",
            filelist_path=tmp_path / "list.json",
        )
        assert task.was_stopped is False

    def test_stop_sets_was_stopped(self, tmp_path):
        task = DatabaseCreationTask(
            db_path=tmp_path / "db",
            filelist_path=tmp_path / "list.json",
        )
        task.stop()
        assert task.was_stopped is True

    def test_run_clears_stop_event_at_start(self, tmp_path, monkeypatch):
        """
        If stop() was called before run(), a fresh run should clear the event
        so was_stopped is False on a successful complete.
        """
        data_dir = tmp_path / "parquet"
        data_dir.mkdir()
        _write_parquets(data_dir, arids=[111], rows_per_file=2, emb_dim=2)

        db_dir = tmp_path / "db"

        import recognizer_workshop.app.create_database.database_creation_task as task_module

        def fake_create(db_path, filelist_path, **kwargs):
            pass  # don't actually hit the network

        monkeypatch.setattr(task_module, "create_database_from_baw", fake_create)

        task = DatabaseCreationTask(
            db_path=db_dir,
            filelist_path=tmp_path / "list.json",
        )
        task.stop()
        assert task.was_stopped is True

        task.run()  # should clear the event at the start
        assert task.was_stopped is False

    def test_stop_event_passed_to_backend(self, tmp_path, monkeypatch):
        """DatabaseCreationTask must forward its internal stop_event to create_database_from_baw."""
        import recognizer_workshop.app.create_database.database_creation_task as task_module

        captured = {}

        def fake_create(db_path, filelist_path, **kwargs):
            captured["stop_event"] = kwargs.get("stop_event")

        monkeypatch.setattr(task_module, "create_database_from_baw", fake_create)

        task = DatabaseCreationTask(
            db_path=tmp_path / "db",
            filelist_path=tmp_path / "list.json",
        )
        task.run()

        assert "stop_event" in captured
        assert isinstance(captured["stop_event"], threading.Event)

    def test_progress_callback_forwarded_to_backend(self, tmp_path, monkeypatch):
        """progress_callback passed to run() must reach create_database_from_baw."""
        import recognizer_workshop.app.create_database.database_creation_task as task_module

        captured = {}

        def fake_create(db_path, filelist_path, **kwargs):
            captured["callback"] = kwargs.get("progress_callback")

        monkeypatch.setattr(task_module, "create_database_from_baw", fake_create)

        sentinel = object()
        task = DatabaseCreationTask(
            db_path=tmp_path / "db",
            filelist_path=tmp_path / "list.json",
        )
        task.run(progress_callback=sentinel)

        assert captured["callback"] is sentinel

    def test_stop_during_run_from_other_thread(self, tmp_path, monkeypatch):
        """
        Simulate stopping from a second thread while run() is executing.
        The task should complete without error and was_stopped should be True.
        """
        import recognizer_workshop.app.create_database.database_creation_task as task_module

        task_ref = []

        def slow_create(db_path, filelist_path, stop_event=None, **kwargs):
            # Simulate work happening; stop triggered from outside
            if task_ref:
                task_ref[0].stop()

        monkeypatch.setattr(task_module, "create_database_from_baw", slow_create)

        task = DatabaseCreationTask(
            db_path=tmp_path / "db",
            filelist_path=tmp_path / "list.json",
        )
        task_ref.append(task)
        task.run()  # stop() called inside fake_create simulating an external thread

        assert task.was_stopped is True
