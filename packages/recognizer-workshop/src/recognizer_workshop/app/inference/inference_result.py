"""
Service for indexing and retrieving inference results stored on disk.

This service scans a predictions folder recursively for CSV files, parses
accompanying metadata, and provides a structured list of results. It is designed
to be backward-compatible, handling older CSV files that may not have the
newer JSON metadata files.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import logging
from typing import Any
import pandas as pd
import random

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResultItem:
    """A single inference result found on disk."""
    csv_path: Path
    inference_date: datetime
    recognizer_name: str
    recognizer_display_name: str
    metadata: dict[str, Any]

    @property
    def display_name(self) -> str:
        """Formatted string for UI dropdowns."""
        date_str = self.inference_date.strftime('%Y-%m-%d %H:%M')
        return f"{self.csv_path.name} ({date_str}) - Recognizer: {self.recognizer_display_name}"

    def get_num_rows(self) -> int:
        """Returns the number of rows in the CSV file."""
        try:
            with self.csv_path.open('rb') as f:
                lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
            # Exclude header line; return 0 if file is empty
            return max(0, lines - 1) if lines > 0 else 0
        except Exception as e:
            log.warning(f"Could not count rows in '{self.csv_path}': {e}")
            return 0

    def get_preview(self, preview_limit = 1000):
        """Returns a random subset of the csv (order preserved) efficiently without loading the entire file into memory."""
        
        total_rows = self.get_num_rows()
        if total_rows == 0:
            return pd.DataFrame(), 0
        keep_indices = set(random.sample(range(1, total_rows + 1), min(preview_limit, total_rows)))
        df_preview = pd.read_csv(self.csv_path, skiprows=lambda x: x > 0 and x not in keep_indices)
        return df_preview, total_rows

    def get_info(self, preview_limit = 1000):
        """Get information about a CSV file including row count and preview"""
        try:
            
            df_preview, row_count = self.get_preview(preview_limit)
            return {
                'total_row_count': row_count,
                'columns': df_preview.columns.tolist() or [],
                'preview_df': df_preview,
                'unique_labels': df_preview['label'].unique()
            }
        except Exception as e:
            return {'error': str(e)}
    

class InferenceResultIndex:
    """Scans and provides an index of inference results."""

    def __init__(self, predictions_folder: Path | str | None):
        self.predictions_folder = Path(predictions_folder) if predictions_folder else None

    def scan_results(self) -> list[InferenceResultItem]:
        """
        Recursively scans the predictions folder for inference CSVs.

        For each CSV, it looks for a sibling `.metadata.json` file. If found,
        it parses it for recognizer details. If not found, it falls back to
        using the CSV filename and file modification time.

        Returns:
            A list of InferenceResultItem objects, sorted with the most recent first.
        """
        if not self.predictions_folder or not self.predictions_folder.exists():
            return []

        results = []
        for csv_path in self.predictions_folder.rglob("*.csv"):
            if not csv_path.is_file():
                continue

            try:
                # Always use file modification time for the inference date.
                inference_date = datetime.fromtimestamp(csv_path.stat().st_mtime)

                # Defaults for fallback case
                metadata = {}
                recognizer_name = csv_path.stem
                recognizer_display_name = "Unknown"

                # Check for and parse the metadata file
                metadata_path = csv_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    with metadata_path.open('r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    recognizer_name = metadata.get("recognizer-name", recognizer_name)
                    recognizer_date = metadata.get("recognizer-date", "Unknown")
                    recognizer_display_name = f"{recognizer_name} ({recognizer_date})"

                results.append(InferenceResultItem(
                    csv_path=csv_path,
                    inference_date=inference_date,
                    recognizer_name=recognizer_name,
                    recognizer_display_name=recognizer_display_name,
                    metadata=metadata,
                ))
            except Exception as e:
                log.warning(f"Could not process potential inference result '{csv_path}': {e}")

        # Sort by inference date, newest first
        return sorted(results, key=lambda item: item.inference_date, reverse=True)