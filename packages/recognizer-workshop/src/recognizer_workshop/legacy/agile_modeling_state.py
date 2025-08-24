# coding=utf-8
# Copyright 2024 The Perch Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
UI-free refactor of the agile modeling state/engine.

This module intentionally contains **no ipywidgets / display() / matplotlib UI**.
It exposes pure functions and data structures that can be used from either a
Jupyter notebook, a Panel app, or a FastAPI backend.

Highlights:
- No UI dependencies
- Structured return types for searches
- Bulk label-saving API
- Optional logging instead of print() (still falls back to print)
- No interactive prompts in create_database(); caller passes overwrite=True

Downstream UIs (Notebook/Panel) are expected to:
- Render spectrograms/audio (from returned `source_id` & `offset_s`)
- Plot histograms using the returned score arrays
- Confirm overwrite before calling create_database(..., overwrite=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import numpy as np
import pandas as pd

from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
from perch_hoplite.agile import source_info
from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from perch_hoplite.db import score_functions
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db import interface as hoplite_interface
from perch_hoplite.zoo import model_configs

from recognizer_workshop.create_database.create_database import convert_parquet
from recognizer_workshop.agile_modeling_helpers import list_audio_files

import baw_helpers.baw_utils as baw_utils
from baw_helpers.baw_api import baw_api

from recognizer_workshop.db_labels.db_labels import db_summary as _db_summary_print
from recognizer_workshop.inference.inference import run_inference as _run_inference


# =====================
# Data structures
# =====================
@dataclass
class AgileConfig:
    """Configuration for the agile modeling engine."""
    db_path: Union[str, Path, None] = None
    annotator_id: Optional[str] = None
    baw_config: dict = field(default_factory=dict)
    search_dataset_name: str = "search_set"
    embeddings_folder: Optional[Union[str, Path]] = None
    labeled_examples_folder: Optional[Union[str, Path]] = None
    models_folder: Optional[Union[str, Path]] = None
    predictions_folder: Optional[Union[str, Path]] = None
    max_embeddings_count: int = -1

    def load_json(self, json_path: Union[str, Path] = "./agile_config.json") -> None:
        p = Path(json_path)
        if not p.exists():
            return
        with p.open("r") as f:
            data = json.load(f)
        def resolve(pathlike):
            if pathlike is None:
                return None
            q = Path(pathlike)
            if not q.is_absolute():
                q = p.parent / q
            return q
        for k, v in data.items():
            if k in {"db_path", "embeddings_folder", "labeled_examples_folder",
                     "models_folder", "predictions_folder"}:
                v = resolve(v)
            setattr(self, k, v)


@dataclass
class Hit:
    embedding_id: int
    source_id: str
    offset_s: float
    score: float


@dataclass
class SearchOutcome:
    label: str
    hits: List[Hit]
    all_scores: np.ndarray  # distribution for plotting/thresholding


# =====================
# Engine
# =====================
class AgileEngine:
    def __init__(self, config: AgileConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.db: Optional[sqlite_usearch_impl.SQLiteUsearchDB] = None
        self.db_model_config = None
        self.model_class = None
        self.embedding_model = None
        self.audio_filepath_loader = None
        self.baw_api = None
        self.classifier = None
        self.label: Dict[str, object] = {}  # { 'species': str, 'call_types': [str] }
        self.log = logger or logging.getLogger("agile")

    # --------------
    # Lifecycle
    # --------------
    def initialize(self) -> None:
        self.init_db()

    def init_db(self) -> None:
        if not self.config.db_path:
            raise ValueError("DB path not set in config.")
        db_path = Path(self.config.db_path)
        if not db_path.exists():
            raise ValueError(f"DB path {db_path} does not exist.")
        self.log.info("Initializing DB at %s", db_path)
        self.db = sqlite_usearch_impl.SQLiteUsearchDB.create(db_path)
        self.db_model_config = self.db.get_metadata("model_config")
        self.model_class = model_configs.get_model_class(self.db_model_config.model_key)
        self.embedding_model = self.model_class.from_config(self.db_model_config.model_config)
        embed_cfg = self.db.get_metadata("audio_sources")
        self.audio_filepath_loader = audio_loader.make_filepath_loader(
            audio_sources=source_info.AudioSources.from_config_dict(embed_cfg),
            window_size_s=self.embedding_model.window_size_s,
            sample_rate_hz=self.embedding_model.sample_rate,
        )
        n = self.db.count_embeddings()
        self.log.info("DB ready with %d embeddings", n)

    # --------------
    # Label selection (UI provides values; engine stores & derives)
    # --------------
    def set_label(self, species: str, call_types: Optional[List[str]] = None) -> None:
        self.label = {
            "species": species or "",
            "call_types": list(call_types or []),
        }

    def get_target_labels(self) -> List[str]:
        species = self.label.get("species", "")
        call_types = self.label.get("call_types") or []
        return [species] if not call_types else [f"{species} {ct}" for ct in call_types]

    def get_target_label_description(self) -> str:
        species = self.label.get("species", "")
        call_types = self.label.get("call_types") or []
        return species if not call_types else f"{species} ({', '.join(call_types)})"

    # --------------
    # Training
    # --------------
    def train_classifier(
        self,
        learning_rate: float = 1e-3,
        weak_neg_weight: float = 0.05,
        num_steps: int = 128,
        train_ratio: float = 0.9,
        batch_size: int = 128,
        weak_negatives_batch_size: int = 128,
    ) -> Dict[str, float]:
        if not self.db:
            raise RuntimeError("DB not initialized")
        target_labels = self.get_target_labels()
        self.log.info("Training for target labels: %s", target_labels)
        dm = classifier_data.AgileDataManager(
            target_labels=target_labels,
            db=self.db,
            train_ratio=train_ratio,
            min_eval_examples=1,
            batch_size=batch_size,
            weak_negatives_batch_size=weak_negatives_batch_size,
            rng=np.random.default_rng(seed=5),
        )
        linear_classifier, eval_scores = classifier.train_linear_classifier(
            data_manager=dm,
            learning_rate=learning_rate,
            weak_neg_weight=weak_neg_weight,
            num_train_steps=num_steps,
        )
        self.classifier = linear_classifier
        # eval_scores has keys like 'top1_acc', 'roc_auc', 'cmap'
        self.log.info("Training done: top1=%.3f roc_auc=%.3f cmap=%.3f",
                      eval_scores.get("top1_acc", float("nan")),
                      eval_scores.get("roc_auc", float("nan")),
                      eval_scores.get("cmap", float("nan")))
        return eval_scores

    # --------------
    # Searching
    # --------------
    def _search_core(
        self,
        *,
        query_vec: np.ndarray,
        target_label: str,
        bias: float = 0.0,
        num_results: int = 50,
        sample_size: Optional[int] = 1_000_000,
        target_score: Optional[float] = None,
        dataset: Optional[str] = None,
        exclude_labeled: bool = True,
    ) -> SearchOutcome:
        if not self.db:
            raise RuntimeError("DB not initialized")
        score_fn = score_functions.get_score_fn("dot", bias=bias, target_score=target_score)
        if dataset is not None:
            all_datasets = self.db.get_dataset_names()
            if dataset not in all_datasets:
                raise ValueError(f"Dataset {dataset} not found; available: {all_datasets}")
        results, all_scores = brutalism.threaded_brute_search(
            self.db, query_vec, num_results, score_fn=score_fn,
            sample_size=sample_size, exclude_labeled=exclude_labeled,
        )
        hits: List[Hit] = []
        for r in results.search_results:
            eid = int(r.embedding_id)
            src = self.db.get_embedding_source(eid)
            hits.append(Hit(
                embedding_id=eid,
                source_id=src.source_id,
                offset_s=float(src.offsets[0]),
                score=float(r.sort_score),
            ))
        return SearchOutcome(label=target_label, hits=hits, all_scores=np.asarray(all_scores))

    def search_with_query(
        self,
        *,
        query_embedding: np.ndarray,
        num_results: int = 50,
        sample_size: Optional[int] = 1_000_000,
        target_score: Optional[float] = None,
        target_label: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> SearchOutcome:
        if dataset == "from_config":
            dataset = self.config.search_dataset_name
        if target_label is None:
            labels = self.get_target_labels()
            if len(labels) == 1:
                target_label = labels[0]
            elif len(labels) == 0:
                raise ValueError("No target label set; call set_label() first")
            else:
                raise ValueError(f"Multiple labels available; please pick one: {labels}")
        return self._search_core(
            query_vec=query_embedding,
            target_label=target_label,
            num_results=num_results,
            sample_size=sample_size,
            target_score=target_score,
            dataset=dataset,
        )

    def search_with_classifier(
        self,
        *,
        num_results: int = 50,
        sample_size: Optional[int] = 1_000_000,
        target_score: Optional[float] = None,
        dataset: Optional[str] = None,
    ) -> Dict[str, SearchOutcome]:
        if self.classifier is None:
            raise RuntimeError("Classifier not trained")
        outcomes: Dict[str, SearchOutcome] = {}
        labels = list(self.classifier.classes)
        self.log.info("Searching for labels: %s", labels)
        for idx, label in enumerate(labels):
            class_query = self.classifier.beta[:, idx]
            bias = float(self.classifier.beta_bias[idx])
            outcomes[label] = self._search_core(
                query_vec=class_query,
                target_label=label,
                bias=bias,
                num_results=num_results,
                sample_size=sample_size,
                target_score=target_score,
                dataset=dataset,
            )
        return outcomes

    # --------------
    # Label persistence
    # --------------
    def save_labels_bulk(
        self,
        updates: List[Tuple[int, Optional[int], str]],
    ) -> Tuple[int, int]:
        """
        Save labels to the DB.
        Args:
            updates: list of (embedding_id, label_type_or_None, label_value)
                     label_type is hoplite_interface.LabelType.POSITIVE/NEGATIVE or None
                     if None, the entry is skipped
        Returns:
            (new_lbls, prev_lbls) counts
        """
        if not self.db:
            raise RuntimeError("DB not initialized")
        new_cnt = prev_cnt = 0
        provenance = self.config.annotator_id or "unknown"
        for eid, lt, label_value in updates:
            if lt is None:
                continue
            lbl = hoplite_interface.Label(
                embedding_id=int(eid),
                value=str(label_value),
                type=lt,
                provenance=provenance,
            )
            check = self.db.insert_label(lbl, skip_duplicates=True)
            new_cnt += check
            prev_cnt += (1 - check)
        self.db.commit()
        self.log.info("Saved labels: new=%d prev=%d", new_cnt, prev_cnt)
        return new_cnt, prev_cnt

    # --------------
    # Exports / reporting
    # --------------
    def save_hits_csv(self, hits: List[Hit], path: Union[str, Path], append: bool = False) -> None:
        rows = []
        for h in hits:
            domain, arid = baw_utils.extract_arid_and_domain(h.source_id)
            audio_url = baw_utils.make_baw_audio_url_from_arid(arid, h.offset_s, 5.0, domain)
            rows.append({
                "embedding_id": h.embedding_id,
                "arid": arid,
                "offset": h.offset_s,
                "score": h.score,
                "link": audio_url,
            })
        df = pd.DataFrame(rows)
        mode = "a" if append else "w"
        header = not append
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, mode=mode, header=header, index=False)

    def run_inference(
        self,
        output_filepath: Optional[Union[str, Path]] = None,
        threshold: float = 0.0,
        labels: Optional[List[str]] = None,
        dataset: Optional[str] = None,
        subset: float = 1.0,
        exclude_labeled: bool = False,
    ) -> None:
        if output_filepath is None:
            output_filepath = Path(self.config.predictions_folder or Path.cwd()) / "inference.csv"
        output_filepath = Path(output_filepath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        _run_inference(
            self.classifier,
            self.db,
            output_filepath,
            self.baw_api,
            threshold=threshold,
            labels=labels,
            subset=subset,
            exclude_labeled=exclude_labeled,
        )

    # --------------
    # DB utilities (no prints; return data or call provided printer)
    # --------------
    def db_summary(self) -> None:
        if not self.db:
            raise RuntimeError("DB not initialized")
        _db_summary_print(self.db)

    # --------------
    # DB creation (no buttons/UI)
    # --------------
    def create_database(self, embeddings_files: Union[str, Path], *, overwrite: bool = False) -> None:
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists() and not overwrite:
            raise FileExistsError(
                f"DB path already exists at {db_path.resolve()}. Pass overwrite=True to recreate."
            )
        # Delete existing if overwriting (wal/shm too)
        for f in [db_path, db_path.parent / f"{db_path.name}-shm", db_path.parent / f"{db_path.name}-wal"]:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass
        if (Path(embeddings_files) / "filelist.json").exists():
            parquet_filepaths = [
                Path(embeddings_files)
                / "embeddings"
                / Path(str(f["site_id"]))
                / Path(str(f["id"]))
                / "embeddings.parquet"
                for f in json.load(open(Path(embeddings_files) / "filelist.json"))
            ]
        else:
            parquet_filepaths = None
        max_count = self.config.max_embeddings_count if self.config.max_embeddings_count else -1
        if not self.config.search_dataset_name:
            raise ValueError("search_dataset_name not set in config")
        self.log.info("Creating DB at %s", db_path.resolve())
        db = convert_parquet(
            parquet_folder=Path(embeddings_files),
            parquet_filepaths=parquet_filepaths,
            db_type="sqlite_usearch",
            dataset_name=self.config.search_dataset_name,
            max_count=max_count,
            db_path=db_path,
        )
        self.log.info("DB created at %s with %d embeddings", db_path.resolve(), db.count_embeddings())

    # --------------
    # Ecosounds API
    # --------------
    def init_workbench(self, token: Optional[str] = None) -> None:
        self.baw_api = baw_api(token=token)
        if self.baw_api.token is not None:
            self.config.baw_config["auth_token"] = self.baw_api.token
            self.log.info("Loaded BAW token")
        elif self.config.baw_config.get("auth_token"):
            self.log.info("Using existing BAW token from config")
        else:
            self.log.warning("No BAW token set")

    def upload_detections(
        self,
        csv_path: Union[str, Path],
        project_id: Optional[Union[str, int]] = None,
        limit: Optional[int] = None,
    ) -> None:
        from recognizer_workshop.events.upload_detections import upload_detections as _upload
        upload_name = f"Agile Modelling: {self.get_target_label_description()}"
        _upload(csv_path=csv_path, upload_name=upload_name, project_id=project_id, limit=limit, baw_api=self.baw_api)

    # --------------
    # Embedding helpers (engine-level; UI decides how to pick the window)
    # --------------
    def embed_window(self, audio_samples_32khz: np.ndarray) -> np.ndarray:
        if self.embedding_model is None:
            raise RuntimeError("DB not initialized (embedding model missing)")
        emb = self.embedding_model.embed(audio_samples_32khz).embeddings[0, 0]
        return emb

    def embed_from_source(self, source_id: str, start_s: float, window_s: float = 5.0) -> np.ndarray:
        if self.embedding_model is None:
            raise RuntimeError("DB not initialized (embedding model missing)")
        sr = self.embedding_model.sample_rate
        audio = self._load_audio(source_id, sr)
        s = max(0, int(start_s * sr))
        e = s + int(window_s * sr)
        return self.embed_window(audio[s:e])

    # Separate helper so UIs can mock/override for remote media
    def _load_audio(self, source_id: str, sample_rate_hz: int) -> np.ndarray:
        from perch_hoplite.agile import audio_io  # local import to avoid heavy import at module load
        return audio_io.load_audio(source_id, sample_rate_hz)

    # --------------
    # Convenience for UIs
    # --------------
    @staticmethod
    def choose_query_uri(query: Union[str, int, Path], labeled_examples_folder: Union[str, Path]) -> Union[str, Path]:
        """Resolve a query reference (index or relative path) into a concrete URI/path.
        - An integer refers to index inside labeled_examples_folder
        - A string path is resolved against labeled_examples_folder if it exists there
        - XenoCanto IDs 'XC123' are normalized to lowercase 'xc123'
        Returns the resolved string/Path; UI can then decide how to display/pick windows.
        """
        if isinstance(query, int) or (isinstance(query, str) and str(query).isdigit()):
            idx = int(query)
            files = list_audio_files(labeled_examples_folder, quiet=True)
            if idx < 0 or idx >= len(files):
                raise IndexError(f"Query index {idx} out of range (0..{len(files)-1})")
            return files[idx]
        q = str(query)
        if q.startswith("XC"):
            q = "xc" + q[2:]
        p = Path(labeled_examples_folder) / q
        return p if p.exists() else q
