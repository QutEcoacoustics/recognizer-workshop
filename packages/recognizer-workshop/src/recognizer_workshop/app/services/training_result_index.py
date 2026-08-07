"""Index of saved trained classifiers on disk, with an in-memory active selection.

Directory layout expected:
    <models_folder>/<classifier_name>/<version_name>/recognizer.json
    <models_folder>/<classifier_name>/<version_name>/metrics.json (optional)

Ordering is always by directory modified time (newest first) for both classifiers
and versions. Version folder names are not required to be parseable dates;
when they are (the `%Y%m%d_%H%M%S` format written by save_classifier_versioned),
they're shown as a human-readable date. Otherwise the raw folder name is used
as the display name.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Optional

from perch_hoplite.agile.classifier import LinearClassifier

from recognizer_workshop.app.services import training as training_service

RECOGNIZER_FILENAME = "recognizer.json"
METRICS_FILENAME = "metrics.json"


def _parse_version_datetime(version_name: str) -> Optional[datetime]:
    try:
        return datetime.strptime(version_name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


@dataclass(frozen=True)
class ClassifierVersionEntry:
    """A single saved classifier version on disk."""
    classifier_name: str
    version_name: str
    version_dir: Path
    modified_time: datetime

    @property
    def recognizer_path(self) -> Path:
        return self.version_dir / RECOGNIZER_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self.version_dir / METRICS_FILENAME

    @property
    def has_metrics(self) -> bool:
        return self.metrics_path.exists()

    @property
    def display_name(self) -> str:
        """Human-readable version label; falls back to the raw folder name."""
        parsed = _parse_version_datetime(self.version_name)
        if parsed is not None:
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        return self.version_name


class ClassifierResultIndex:
    """Scans the saved-classifier folder and tracks the active selection."""

    def __init__(self, models_folder: Path | str | None):
        self.models_folder = Path(models_folder) if models_folder else None
        self.entries: dict[str, dict[str, ClassifierVersionEntry]] = {}
        self.active_classifier: str | None = None
        self.active_version: str | None = None

    def initialize_index(self) -> None:
        """Rescans the models folder and resets the active selection to the newest entry."""
        self.entries = self._scan()
        self._select_newest_globally()

    def refresh(self) -> None:
        """Rescans the models folder and resets the active selection to the newest entry.

        Used after training or deleting a version, since either can change what
        the newest entry is.
        """
        self.initialize_index()

    def _scan(self) -> dict[str, dict[str, ClassifierVersionEntry]]:
        if not self.models_folder or not self.models_folder.exists():
            return {}

        found: dict[str, dict[str, ClassifierVersionEntry]] = {}
        for classifier_dir in self.models_folder.iterdir():
            if not classifier_dir.is_dir():
                continue
            versions: dict[str, ClassifierVersionEntry] = {}
            for version_dir in classifier_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                if not (version_dir / RECOGNIZER_FILENAME).exists():
                    continue
                modified_time = datetime.fromtimestamp(version_dir.stat().st_mtime)
                versions[version_dir.name] = ClassifierVersionEntry(
                    classifier_name=classifier_dir.name,
                    version_name=version_dir.name,
                    version_dir=version_dir,
                    modified_time=modified_time,
                )
            if versions:
                found[classifier_dir.name] = versions
        return found

    def _newest_entry_overall(self) -> ClassifierVersionEntry | None:
        all_entries = [entry for versions in self.entries.values() for entry in versions.values()]
        if not all_entries:
            return None
        return max(all_entries, key=lambda e: e.modified_time)

    def _select_newest_globally(self) -> None:
        newest_entry = self._newest_entry_overall()
        if newest_entry is None:
            self.active_classifier = None
            self.active_version = None
            return
        self.active_classifier = newest_entry.classifier_name
        self.active_version = newest_entry.version_name

    def get_classifier_list(self) -> list[str]:
        """Classifier names, ordered by their newest version's modified time, newest first."""
        def newest_modified_time(classifier_name: str) -> datetime:
            return max(v.modified_time for v in self.entries[classifier_name].values())

        return sorted(self.entries.keys(), key=newest_modified_time, reverse=True)

    def get_versions_for_classifier(self, classifier_name: str) -> list[ClassifierVersionEntry]:
        """Versions for a classifier, newest first by modified time."""
        versions = self.entries.get(classifier_name, {})
        return sorted(versions.values(), key=lambda e: e.modified_time, reverse=True)

    def get_classifier_entry(self, classifier_name: str, version_name: str) -> ClassifierVersionEntry | None:
        return self.entries.get(classifier_name, {}).get(version_name)

    def get_active_classifier_entry(self) -> ClassifierVersionEntry | None:
        if self.active_classifier is None or self.active_version is None:
            return None
        return self.get_classifier_entry(self.active_classifier, self.active_version)

    def set_active_classifier(self, classifier_name: str, version_name: str) -> ClassifierVersionEntry:
        entry = self.get_classifier_entry(classifier_name, version_name)
        if entry is None:
            raise ValueError(f"No saved classifier found for '{classifier_name}' version '{version_name}'.")
        self.active_classifier = classifier_name
        self.active_version = version_name
        return entry

    def delete_classifier_version(self, classifier_name: str, version_name: str) -> None:
        """Deletes a single version directory, then refreshes and reselects the newest entry."""
        entry = self.get_classifier_entry(classifier_name, version_name)
        if entry is None:
            raise ValueError(f"No saved classifier found for '{classifier_name}' version '{version_name}'.")
        shutil.rmtree(entry.version_dir)
        self.refresh()

    @property
    def is_empty(self) -> bool:
        return not self.entries


def load_training_result_from_classifier_entry(entry: ClassifierVersionEntry) -> training_service.TrainingResult:
    """Load a TrainingResult from a saved on-disk classifier entry."""
    classifier_model = LinearClassifier.load(str(entry.recognizer_path))
    eval_scores = None
    training_info = None
    if entry.has_metrics:
        with entry.metrics_path.open('r', encoding='utf-8') as f:
            metrics_json = json.load(f)
        eval_scores = metrics_json.get('metrics')
        training_info = metrics_json.get('info')

    return training_service.TrainingResult(
        classifier_model=classifier_model,
        eval_scores=eval_scores,
        training_info=training_info,
        classifier_name=entry.classifier_name,
        version_name=entry.version_name,
    )
