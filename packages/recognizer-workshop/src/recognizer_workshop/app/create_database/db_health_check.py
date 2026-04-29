import dataclasses
import logging
from typing import List

import numpy as np

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
        missing_set = set(missing)
        cursor = db._get_cursor()
        placeholders = ','.join('?' * len(missing))
        cursor.execute(
            f"SELECT COUNT(*) FROM hoplite_labels WHERE embedding_id IN ({placeholders})",
            missing,
        )
        orphaned_labels = cursor.fetchone()[0]

    return HealthCheckResult(
        total_sqlite_embeddings=total_sqlite,
        total_index_embeddings=total_index,
        missing_from_index=missing,
        orphaned_labels_count=orphaned_labels,
    )
