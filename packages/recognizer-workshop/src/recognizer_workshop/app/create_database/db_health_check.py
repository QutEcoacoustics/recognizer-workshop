import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from recognizer_workshop.app.services.db_labels.dataclasses import LabelCounts

log = logging.getLogger(__name__)


@dataclasses.dataclass
class HealthCheckResult:
    total_sqlite_embeddings: int
    total_index_embeddings: int
    missing_from_index: List[int]
    orphaned_labels_count: int

    @property
    def is_healthy(self) -> bool:
        return len(self.missing_from_index) == 0

    def __str__(self) -> str:
        lines = [
            "DB Health Check:",
            f"  SQLite embeddings: {self.total_sqlite_embeddings}",
            f"  Index embeddings:  {self.total_index_embeddings}",
        ]
        if self.is_healthy:
            lines.append("  Status: ✅ Healthy — all embeddings present in index.")
        else:
            lines.append(f"  Status: ⚠️ {len(self.missing_from_index)} embedding(s) missing from index.")
            if self.orphaned_labels_count:
                lines.append(f"  Orphaned labels (referencing missing embeddings): {self.orphaned_labels_count}")
            preview = self.missing_from_index[:10]
            lines.append(f"  Missing IDs (first 10): {preview}")
        return "\n".join(lines)


def run_health_check(db) -> HealthCheckResult:
    """Check consistency between the SQLite embeddings table and the usearch index."""
    sqlite_ids = db.get_embedding_ids()
    total_sqlite = len(sqlite_ids)
    total_index = db.ui.size

    # Check which SQLite IDs are actually in the usearch index
    if total_sqlite == 0:
        missing = []
    else:
        contains = db.ui.contains(sqlite_ids)
        missing = sqlite_ids[~np.asarray(contains)].tolist()

    # Count labels that reference missing embeddings
    orphaned_labels = 0
    if missing:
        cursor = db._get_cursor()
        max_sqlite_variables = 900
        for start in range(0, len(missing), max_sqlite_variables):
            batch = missing[start:start + max_sqlite_variables]
            placeholders = ",".join("?" * len(batch))
            cursor.execute(
                f"SELECT COUNT(*) FROM hoplite_labels WHERE embedding_id IN ({placeholders})",
                batch,
            )
            orphaned_labels += cursor.fetchone()[0]

    return HealthCheckResult(
        total_sqlite_embeddings=total_sqlite,
        total_index_embeddings=total_index,
        missing_from_index=missing,
        orphaned_labels_count=orphaned_labels,
    )


@dataclasses.dataclass(frozen=True)
class IndexInfo:
    """Size information for the usearch index file on disk."""
    path: Path
    exists: bool
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def format_size(self) -> str:
        if not self.exists:
            return "not present"
        mb = self.size_mb
        if mb >= 1:
            return f"{mb:.1f} MB"
        return f"{self.size_bytes / 1024:.1f} KB"


def get_index_info(db) -> IndexInfo:
    """Return the path and disk size of the usearch index file for this database."""
    index_path = Path(db.db_path) / "usearch.index"
    exists = index_path.exists()
    size_bytes = index_path.stat().st_size if exists else 0
    return IndexInfo(path=index_path, exists=exists, size_bytes=size_bytes)


def delete_usearch_index(db) -> IndexInfo:
    """Delete the usearch index file from disk.

    The SQLite tables are untouched. The index can be rebuilt afterwards using
    the repair-index flow.

    Returns the IndexInfo captured *before* deletion.
    """
    info = get_index_info(db)
    if not info.exists:
        raise FileNotFoundError(f"Usearch index not found at {info.path}")
    info.path.unlink()
    log.info("Deleted usearch index at %s (was %s)", info.path, info.format_size())
    return info


def database_exists_at(db_path) -> bool:
    """Return True if a database exists on disk at the given path.

    This is a cheap, path-only check that does NOT require an open database
    connection. A database is considered present if the directory exists and
    contains at least one ``*.sqlite`` or ``*.usearch`` file. Used to decide UI
    state (e.g. button labels) before the engine has loaded a database.
    """
    if not db_path:
        return False
    path = Path(db_path).expanduser()
    return path.exists() and (
        any(path.glob("*.sqlite")) or any(path.glob("*.usearch"))
    )


@dataclasses.dataclass(frozen=True)
class DatabaseStatus:
    """A consolidated snapshot of a database's state.

    This is the single source of truth for "database status" across the app.
    Cheap sections (existence, embedding count, datasets, label counts, and the
    index file size) are always populated. The expensive consistency check
    (``health``) is only populated when ``include_health=True`` is passed to
    :func:`get_database_status`.

    Each section is computed defensively: if one section fails (e.g. a corrupt
    index), the others are still returned and the failure is recorded in
    ``errors`` so the UI can show partial status rather than nothing.
    """

    db_path: Path
    db_exists: bool
    is_loaded: bool
    total_embeddings: Optional[int]
    datasets: List[str]
    label_counts: Dict[str, LabelCounts]
    index: Optional[IndexInfo]
    health: Optional[HealthCheckResult] = None
    errors: Dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def is_healthy(self) -> Optional[bool]:
        """Health status, or None when the health check was not run."""
        return self.health.is_healthy if self.health is not None else None


def get_database_status(db, *, include_health: bool = False) -> DatabaseStatus:
    """Build a consolidated :class:`DatabaseStatus` for a loaded database.

    Args:
        db: An open ThreadedSQLiteUsearchDB instance.
        include_health: When True, also run the (potentially expensive)
            SQLite/usearch consistency check and populate ``status.health``.

    Each section is wrapped in its own try/except so a single failure does not
    prevent the rest of the status from being returned.
    """
    # Imported lazily to avoid any import-ordering concerns between the
    # create_database and db_labels service packages.
    from recognizer_workshop.app.services.db_labels.db_labels_info import get_db_summary

    db_path = Path(db.db_path)
    errors: Dict[str, str] = {}

    total_embeddings: Optional[int] = None
    datasets: List[str] = []
    label_counts: Dict[str, LabelCounts] = {}
    try:
        summary = get_db_summary(db)
        total_embeddings = summary.total_embeddings
        datasets = summary.datasets
        label_counts = summary.label_counts
    except Exception as e:  # noqa: BLE001 - report per-section failure
        log.exception("Failed to compute database summary")
        errors["summary"] = str(e)

    index: Optional[IndexInfo] = None
    try:
        index = get_index_info(db)
    except Exception as e:  # noqa: BLE001 - report per-section failure
        log.exception("Failed to read usearch index info")
        errors["index"] = str(e)

    health: Optional[HealthCheckResult] = None
    if include_health:
        try:
            health = run_health_check(db)
            log.info("Health check result: %s", health)
        except Exception as e:  # noqa: BLE001 - report per-section failure
            log.exception("Health check failed")
            errors["health"] = str(e)

    return DatabaseStatus(
        db_path=db_path,
        db_exists=database_exists_at(db_path),
        is_loaded=True,
        total_embeddings=total_embeddings,
        datasets=datasets,
        label_counts=label_counts,
        index=index,
        health=health,
        errors=errors,
    )
