"""
DatabaseCreationTask and format_progress_event.

These are framework-agnostic and fully testable without Panel or any UI layer.

Usage::

    task = DatabaseCreationTask(db_path=..., filelist_path=..., client=..., ...)
    task.run(progress_callback=my_callback)   # synchronous; call via asyncio.to_thread from a UI
    # or from another thread: task.stop()
    if task.was_stopped:
        print("stopped early")
"""

import threading
from pathlib import Path
from typing import Callable

from recognizer_workshop.app.create_database.create_database import create_database_from_baw


def format_progress_event(event: dict) -> tuple[list[str], int | None]:
    """
    Convert a raw progress-event dict (as emitted by create_database_from_baw) into
    a (messages, progress_percent) tuple.

    Returns:
        messages: list of human-readable strings to display (may be empty).
        progress_percent: 0-100 integer when a progress-bar update is appropriate,
                          otherwise None.
    """
    stage = event.get("stage")

    if stage == "selection_start":
        cache_mode = "cache" if event.get("use_cache") else "fresh fetch"
        return [
            f"Selecting recordings (max_files={event.get('max_files')}, mode={cache_mode})"
        ], None

    if stage == "selection_complete":
        return [f"Selected {event.get('selected_recordings', 0)} recordings"], None

    if stage == "ingest_start":
        total_candidates = event.get("total_candidates")
        max_count = event.get("max_count")
        current_embeddings = event.get("current_embeddings")
        messages = []
        if total_candidates is not None:
            messages.append(f"Starting ingestion for {total_candidates} candidate recordings")
        else:
            messages.append("Starting ingestion")
        if max_count and max_count > 0 and current_embeddings is not None:
            messages.append(
                f"Current DB size is {current_embeddings} embeddings (limit {max_count})"
            )
        return messages, None

    if stage == "source_progress":
        candidates_seen = event.get("candidates_seen", 0)
        total_candidates = event.get("total_candidates", 0)
        skipped_existing = event.get("skipped_existing", 0)
        files_processed = event.get("files_processed", 0)
        pct = (
            int(min(100, (candidates_seen / total_candidates) * 100))
            if total_candidates
            else None
        )
        if total_candidates:
            msg = (
                f"Checked {candidates_seen}/{total_candidates} recordings "
                f"(downloaded {files_processed}, skipped existing {skipped_existing})"
            )
        else:
            msg = (
                f"Checked {candidates_seen} recordings "
                f"(downloaded {files_processed}, skipped existing {skipped_existing})"
            )
        return [msg], pct

    if stage == "ingest_progress":
        files_processed = event.get("files_processed", 0)
        total_candidates = event.get("total_candidates")
        added_this_file = event.get("added_this_file", 0)
        total_inserted = event.get("total_inserted", 0)
        pct = (
            int(min(100, (files_processed / total_candidates) * 100))
            if total_candidates
            else None
        )
        if total_candidates:
            msg = (
                f"Processed {files_processed}/{total_candidates} files "
                f"(+{added_this_file}, total added {total_inserted})"
            )
        else:
            msg = f"Processed {files_processed} files (+{added_this_file}, total added {total_inserted})"
        return [msg], pct

    if stage == "ingest_complete":
        stop_reason = event.get("stop_reason")
        files_processed = event.get("files_processed", 0)
        total_inserted = event.get("total_inserted", 0)
        final_embeddings = event.get("final_embeddings", 0)
        if stop_reason == "max_count_already_reached":
            return [f"Skipped ingestion: DB already at limit ({final_embeddings} embeddings)"], None
        if stop_reason == "user_stopped":
            return [
                f"⏹ Stopped after {files_processed} files, "
                f"{total_inserted} embeddings added. DB committed."
            ], None
        return [
            f"Ingestion complete: {files_processed} files, {total_inserted} embeddings added"
        ], None

    return [], None


class DatabaseCreationTask:
    """
    Wraps create_database_from_baw with a stop mechanism.

    This class is framework-agnostic. The UI layer creates an instance, calls
    ``run()`` via ``asyncio.to_thread``, and calls ``stop()`` from a button handler.
    Scripts can call ``run()`` directly in the main thread or a worker thread.
    """

    def __init__(
        self,
        db_path: str | Path,
        filelist_path: str | Path,
        *,
        client=None,
        filter: dict | None = None,
        max_files: int = -1,
        max_embeddings: int = 1_000_000,
        use_cache: bool = True,
        commit_every_files: int = 50,
        config=None,
    ):
        self.db_path = db_path
        self.filelist_path = filelist_path
        self.client = client
        self.filter = filter
        self.max_files = max_files
        self.max_embeddings = max_embeddings
        self.use_cache = use_cache
        self.commit_every_files = commit_every_files
        self.config = config
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the task to halt after the current file finishes."""
        self._stop_event.set()

    @property
    def was_stopped(self) -> bool:
        """True if stop() was called before the run completed normally."""
        return self._stop_event.is_set()

    def run(self, progress_callback: Callable[[dict], None] | None = None) -> None:
        """
        Run database creation synchronously.

        Safe to call from any thread, including via ``asyncio.to_thread``.
        The stop event is reset at the start of each run so the same task
        object can be reused if needed.
        """
        self._stop_event.clear()
        create_database_from_baw(
            db_path=self.db_path,
            filelist_path=self.filelist_path,
            client=self.client,
            filter=self.filter,
            max_files=self.max_files,
            max_embeddings=self.max_embeddings,
            use_cache=self.use_cache,
            commit_every_files=self.commit_every_files,
            config=self.config,
            progress_callback=progress_callback,
            stop_event=self._stop_event,
        )
