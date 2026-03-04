from __future__ import annotations
from typing import Protocol, Iterable, Literal, runtime_checkable
from pathlib import Path
import numpy as np
from perch_hoplite.db import interface as hoplite_interface

@runtime_checkable
class AgileUI(Protocol):
    """A stateful UI adapter: retains last results so the controller stays simple."""

    # Label selection (updates engine.label in-place)
    def pick_label(self, eng, initial_species: str | None = None,
                   initial_call_types: list[str] | None = None) -> None: ...
    
    # A component to define the list of target labels
    def view_target_labels(self, eng): ...

    # a component to select one of the target labels for e.g. searching
    def select_active_label(self, eng): ...

    # Interactively pick a 5s window and return raw audio samples (32 kHz).
    def select_query_window(self, uri: str, window_size_s: float = 5.0) -> np.ndarray: ...

    # Render TopK results and keep internal handle for later harvesting.
    def show_results(self, eng, results, *, positive_labels: list[str]) -> None: ...

    # Show arbitrary examples (for your sanity-check view)
    def show_examples(self, examples: list[dict], eng) -> None: ...

    # Return hoplite Label objects collected from the last UI render
    def harvest_labels(self) -> Iterable[hoplite_interface.Label]: ...
     
    # Histogram of scores for a label (reads engine-stored scores)
    def plot_scores_histogram(self, eng, label: str) -> None: ...

    # Utilities
    def set_active_label(self, label: str | None) -> None: ...
    def clear_results(self) -> None: ...
    def confirm_overwrite(self, db_path: Path) -> bool: ...
    def notify(self, msg: str, level: Literal["info","warn","error"] = "info") -> None: ...
