"""Single source of truth for database status.

Everything about "what state is this database in" lives here:

* :class:`DatabaseStatus` — the one dataclass describing a loaded database.
* :func:`get_database_status` — builds it (cheap fields always; the expensive
  consistency check only when ``include_health=True``).
* :func:`database_exists_at` — a path-only, pre-load existence check.
* :func:`delete_usearch_index` — index-file management action.

This module is consumed by the create-database page, the status page, the
fetch/add-labels flow and the engine. There is intentionally no other
"summary"/"health"/"index info" type — they were consolidated into this.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from recognizer_workshop.app.services.db_labels.dataclasses import LabelCounts
from recognizer_workshop.app.services.db_labels.db_labels_info import get_label_counts

log = logging.getLogger(__name__)

USEARCH_INDEX_FILENAME = "usearch.index"

# SQLite limits the number of host parameters in a single statement.
_MAX_SQLITE_VARIABLES = 900


def format_size(num_bytes: Optional[int]) -> str:
    """Human-readable size, or ``"not present"`` when ``num_bytes`` is None."""
    if num_bytes is None:
        return "not present"
    mb = num_bytes / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.1f} MB"
    return f"{num_bytes / 1024:.1f} KB"


@dataclasses.dataclass(frozen=True)
class DatabaseStatus:
    """A consolidated snapshot of a loaded database's state.

    The first group of fields is always populated and cheap to compute. The
    health fields (``missing_from_index``, ``orphaned_labels``) are only filled
    in when :func:`get_database_status` is called with ``include_health=True``;
    otherwise they are ``None``.
    """

    db_embeddings: int                      # rows in the SQLite embeddings table
    index_embeddings: int                   # entries in the usearch index (ui.size)
    datasets: List[str]
    label_counts: Dict[str, LabelCounts]
    index_size_bytes: Optional[int]         # None => index file is missing

    # Populated only when a health check was requested.
    missing_from_index: Optional[int] = None    # db rows absent from the index
    orphaned_labels: Optional[int] = None       # labels referencing those rows

    @property
    def index_present(self) -> bool:
        """True when the usearch index file exists on disk."""
        return self.index_size_bytes is not None

    @property
    def health_checked(self) -> bool:
        """True when this snapshot includes the consistency check."""
        return self.missing_from_index is not None

    @property
    def is_healthy(self) -> Optional[bool]:
        """Tri-state index health.

        * ``False`` — definitely unhealthy: the index file is missing, the
          database and index hold different numbers of embeddings, or a health
          check found database rows absent from the index.
        * ``True`` — a health check confirmed every database embedding is in the
          index.
        * ``None`` — unknown: the index exists and its size matches the
          database, but the (slower) per-id health check has not been run.
        """
        if not self.index_present:
            return False
        if self.db_embeddings != self.index_embeddings:
            return False
        if self.health_checked:
            return self.missing_from_index == 0
        return None

    def index_size_str(self) -> str:
        """Human-readable index size (``"not present"`` when missing)."""
        return format_size(self.index_size_bytes)

    def __str__(self) -> str:
        lines = [
            "Database Status:",
            f"  Embeddings (database): {self.db_embeddings}",
            f"  Embeddings (index):    {self.index_embeddings}",
            f"  Index size: {self.index_size_str()}",
        ]
        if self.datasets:
            lines.append(f"  {len(self.datasets)} Dataset(s): {', '.join(self.datasets)}")
        else:
            lines.append("  Datasets: none")
        if self.label_counts:
            lines.append("  Labels:")
            for label_name in sorted(self.label_counts.keys()):
                lines.append(f"    {self.label_counts[label_name]}")
        else:
            lines.append("  Labels: none")
        healthy = self.is_healthy
        if healthy is None:
            lines.append("")
            lines.append("  Health: counts match; run a health check to confirm.")
        elif healthy:
            lines.append("")
            lines.append("  Health: ✅ all database embeddings are present in the index.")
        else:
            lines.append("")
            if not self.index_present:
                lines.append("  Health: ⚠️ the usearch index file is missing.")
            elif self.health_checked:
                lines.append(f"  Health: ⚠️ {self.missing_from_index} embedding(s) missing from index.")
                if self.orphaned_labels:
                    lines.append(f"  Orphaned labels (referencing missing embeddings): {self.orphaned_labels}")
            else:
                lines.append(
                    f"  Health: ⚠️ database has {self.db_embeddings} embedding(s) "
                    f"but the index has {self.index_embeddings}."
                )
        return "\n".join(lines)


def _count_db_embeddings(db) -> int:
    """Count rows in the SQLite embeddings table (independent of the index)."""
    cursor = db._get_cursor()
    cursor.execute("SELECT COUNT(*) FROM hoplite_embeddings;")
    return int(cursor.fetchone()[0])


def _index_size_bytes(db) -> Optional[int]:
    """Disk size of the usearch index file, or None when it is absent."""
    index_path = Path(db.db_path) / USEARCH_INDEX_FILENAME
    return index_path.stat().st_size if index_path.exists() else None


def _missing_index_ids(db) -> List[int]:
    """SQLite embedding IDs that are not present in the usearch index."""
    sqlite_ids = db.get_embedding_ids()
    if len(sqlite_ids) == 0:
        return []
    contains = db.ui.contains(sqlite_ids)
    return sqlite_ids[~np.asarray(contains)].tolist()


def _count_orphaned_labels(db, missing_ids: List[int]) -> int:
    """Count labels that reference embedding IDs missing from the index."""
    if not missing_ids:
        return 0
    cursor = db._get_cursor()
    total = 0
    for start in range(0, len(missing_ids), _MAX_SQLITE_VARIABLES):
        batch = missing_ids[start:start + _MAX_SQLITE_VARIABLES]
        placeholders = ",".join("?" * len(batch))
        cursor.execute(
            f"SELECT COUNT(*) FROM hoplite_labels WHERE embedding_id IN ({placeholders})",
            batch,
        )
        total += cursor.fetchone()[0]
    return total


def get_database_status(db, *, include_health: bool = False) -> DatabaseStatus:
    """Build a :class:`DatabaseStatus` for an open database.

    Args:
        db: An open ThreadedSQLiteUsearchDB instance.
        include_health: When True, also run the (potentially expensive)
            SQLite/usearch consistency check and populate the health fields.
    """
    missing_from_index: Optional[int] = None
    orphaned_labels: Optional[int] = None
    if include_health:
        missing_ids = _missing_index_ids(db)
        missing_from_index = len(missing_ids)
        orphaned_labels = _count_orphaned_labels(db, missing_ids)

    return DatabaseStatus(
        db_embeddings=_count_db_embeddings(db),
        index_embeddings=int(db.ui.size),
        datasets=list(db.get_dataset_names()),
        label_counts=get_label_counts(db),
        index_size_bytes=_index_size_bytes(db),
        missing_from_index=missing_from_index,
        orphaned_labels=orphaned_labels,
    )


def database_exists_at(db_path) -> bool:
    """Return True if a database exists on disk at the given path.

    A cheap, path-only check that does NOT require an open database connection.
    A database is considered present if the directory exists and contains at
    least one ``*.sqlite`` or ``usearch.index`` file. Used to decide UI state
    (e.g. button labels) before the engine has loaded a database.
    """
    if not db_path:
        return False
    path = Path(db_path).expanduser()
    return path.exists() and (
        any(path.glob("*.sqlite")) or (path / USEARCH_INDEX_FILENAME).exists()
    )


def delete_usearch_index(db) -> int:
    """Delete the usearch index file from disk; return the bytes freed.

    The SQLite tables are untouched; the index can be rebuilt afterwards using
    the repair-index flow.
    """
    index_path = Path(db.db_path) / USEARCH_INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"Usearch index not found at {index_path}")
    size_bytes = index_path.stat().st_size
    index_path.unlink()
    log.info("Deleted usearch index at %s (was %s)", index_path, format_size(size_bytes))
    return size_bytes
