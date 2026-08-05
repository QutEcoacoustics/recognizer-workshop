import json
import functools
from pathlib import Path
import numpy as np
import random
from typing import Optional, Callable, Any
from ml_collections import config_dict
import multiprocessing
import logging
from urllib.parse import urlparse

from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import source_info
from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from perch_hoplite.db import score_functions
from perch_hoplite.db import interface as hoplite_interface
from perch_hoplite.zoo import model_configs
from perch_hoplite import audio_io

import baw_helpers.baw_utils as baw_utils
from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_mod
from baw_helpers.baw_api import baw_api

from recognizer_workshop.app.agile_config import AgileConfig
from recognizer_workshop.app.app_settings import app_settings  # noqa: F401 — triggers baw_settings override
from recognizer_workshop.app.services.events.upload_detections import upload_detections
from recognizer_workshop.app.services.db_labels.fetch_baw_labels import labels_from_event_taggings
from recognizer_workshop.app.services.db_labels.add_db_labels import add_labels_to_db
from recognizer_workshop.app.services.db_labels.db_labels_info import get_label_counts
from recognizer_workshop.app.services.db_labels.dataclasses import LabelName
from recognizer_workshop.app.services import training as training_service
from recognizer_workshop.app.create_database.repair_index import repair_index
from recognizer_workshop.app.services.database_status import (
    get_database_status,
    database_exists_at,
    delete_usearch_index as _delete_usearch_index,
)
from recognizer_workshop.app.services.baw_insertions import get_last_filter_for_dataset
from recognizer_workshop.app.services.inference.inference import run_inference as inference_service_run_inference

from recognizer_workshop.app.services.embed import embed_worker
from concurrent.futures import ProcessPoolExecutor

log = logging.getLogger(__name__)



class agile_state:

    config: AgileConfig = None
    db: sqlite_usearch_impl_mod.ThreadedSQLiteUsearchDB = None
    db_model_config = None
    model_class = None
    _embedding_model = None
    audio_filepath_loader = None
    baw_api_clients = {}
    active_client = None
    _active_label: Optional[LabelName] = None  

    # search results from the last search, whether it's a single example query
    # or a classifier-based search.
    search_results = None

    # an object representing the visual display of search_results
    # including result of interactive user-labeling of these results. 
    # this is overwritten each time a search results are displayed. 
    display_group = None

    # list of all ecosounds tags to use for selecting target labels
    workbench_tags = []
    fetched_label_result_sets = None
    fetched_label_outcomes = None
    labels_fetch_state = 'idle'
    labels_fetch_message = ''
    labels_fetch_error = None
    labels_commit_state = 'idle'
    labels_commit_message = ''
    labels_commit_error = None
    train_result = None

    def __init__(self, config):
        self.config = config
        self.baw_api_clients = {}
        self.active_client = None
        self.fetched_label_result_sets = None
        self.fetched_label_outcomes = None
        self.labels_fetch_state = 'idle'
        self.labels_fetch_message = ''
        self.labels_fetch_error = None
        self.labels_commit_state = 'idle'
        self.labels_commit_message = ''
        self.labels_commit_error = None
        mp_context = multiprocessing.get_context('spawn')
        self._embed_runner = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp_context
        )

    @staticmethod
    def count_fetched_labels(result_sets) -> int:
        """Count actual labeled segments across fetched result sets."""
        if not result_sets or not isinstance(result_sets, list):
            return 0
        total = 0
        for result_set in result_sets:
            counts = result_set.get("counts", {}) if isinstance(result_set, dict) else {}
            if isinstance(counts, dict):
                # Sum positive and negative counts across all labels
                for label_counts in counts.values():
                    if isinstance(label_counts, dict):
                        total += label_counts.get('positive', 0)
                        total += label_counts.get('negative', 0)
        return total

    @property
    def is_db_loaded(self) -> bool:
        """True when the database has been successfully initialised via init_db()."""
        return self.db is not None

    @property
    def has_fetched_labels_ready(self) -> bool:
        """True when there are fetched labels available to add to DB."""
        return self.count_fetched_labels(self.fetched_label_result_sets) > 0

    @property
    def fetched_labels_count(self) -> int:
        """Number of fetched labels currently held in engine state."""
        return self.count_fetched_labels(self.fetched_label_result_sets)


    def initialize(self):
        self.init_db()


    def init_db(self):
        log.info('Initializing DB...')

        if not self.config.db_path:
            raise ValueError('DB path not set in config. Please set it before initializing the DB.')
        if not Path(self.config.db_path).exists():
            raise ValueError(f'DB path {self.config.db_path} does not exist.')
        # if no sqlite file or usearch index file exists inside the db_path folder
        if not any(Path(self.config.db_path).glob("*.sqlite")) and not any(Path(self.config.db_path).glob("*.usearch")):
            raise ValueError(f'No valid DB files found in {self.config.db_path}. Please check the path.')
        
        try:
            self.db = sqlite_usearch_impl_mod.ThreadedSQLiteUsearchDB.create(self.config.db_path)
        except Exception as e:
            log.exception(f"Failed to initialize DB at {self.config.db_path}")
            raise RuntimeError(f"DB initialization failed: {e}") from e

        self.db_model_config = self.db.get_metadata('model_config')
        
        self.model_class = model_configs.get_model_class(self.db_model_config.model_key)
        # self.embedding_model = self.model_class.from_config(self.db_model_config.model_config)


        num_embeddings = self.db.count_embeddings()
        log.info(f'DB initialized with {num_embeddings} embeddings.')
        log.info('DB status: %s', get_database_status(self.db))

        #TODO: This is a hack to get the audio loader to be created even when it's not used.  
        # seems that embed config metadata is not inserted during legacy conversion. 
        # it's wrapped in the legacy_config object instead. 
        # In the case of ecosounds recordings, this filepath loader is not really used, but
        # it needs to exist to pass to the display_group object.

        # embed_config = self.db.get_metadata('audio_sources')
        # self.audio_filepath_loader = audio_loader.make_filepath_loader(
        #   audio_sources=source_info.AudioSources.from_config_dict(embed_config),
        #   window_size_s=self.db_model_config.model_config.window_size_s,
        #   sample_rate_hz=self.db_model_config.model_config.sample_rate,
        # )

        self.audio_filepath_loader = self._create_audio_loader()

        return True

    # @property
    # def embedding_model(self):
    #     """
    #     Lazy-loads the embedding model on first access and caches it.
    #     Tfhub attempts to cache downloaded models. This function handles cache clearing and retries,
    #     in the event that the cache is corrupted.
    #     """
    #     if self._embedding_model is None:
    #         log.info("Loading embedding model for the first time...")

    #         def clear_tfhub_cache():
    #             cache_dir = Path(tempfile.gettempdir()) / "tfhub_modules"
    #             if cache_dir.exists():
    #                 log.info(f"Corrupted cache detected. Deleting {cache_dir}...")
    #                 try:
    #                     shutil.rmtree(cache_dir)
    #                     log.info("Cache cleared successfully.")
    #                 except OSError as e:
    #                     log.info(f"Error clearing cache: {e}")

    #         max_retries = 2
    #         for attempt in range(max_retries):
    #             try:
    #                 #TODO: load the model based on db config data rather than hardcoding "perch_8"
    #                 self._embedding_model = model_configs.load_model_by_name("perch_8")
    #                 #self._embedding_model = self.model_class.from_config(self.db_model_config.model_config)
    #                 log.info("Model loaded successfully.")
    #                 break
    #             except ValueError as e:
    #                 is_cache_error = 'saved_model.pb' in str(e) or 'saved_model.pbtxt' in str(e)
    #                 if is_cache_error and attempt < max_retries - 1:
    #                     log.info(f"Attempt {attempt + 1} failed. Looks like a cache issue.")
    #                     clear_tfhub_cache()
    #                     log.info("Retrying model load...")
    #                 else:
    #                     log.warning("Failed to load model after retries.")
    #                     raise e
    #     return self._embedding_model

    @property
    def active_label(self) -> Optional[LabelName]:
        """
        Gets the active label, ensuring it is an identical object from the
        config.target_labels list. Includes a fast-path identity check.
        """
        pool = getattr(self.config, "target_labels", []) or []
        current_intent = self._active_label

        if not pool:
            return None

        if current_intent in pool:
            return current_intent

        if current_intent:
            match = next((ln for ln in pool if ln == current_intent), None)
            if match is not None:
                # We found the equivalent object, return it.
                return match

        # We always return an active label, defaulting to the first item if necessary.
        return pool[0]


    @active_label.setter
    def active_label(self, value: Optional[LabelName]):
        """Sets the user's intended active label."""
        self._active_label = value

    
    # def embed_query_old(self, query_audio_samples):
    #     """
    #     Embeds a single query audio clip allowing the user to select the 5s window within the specified source
    #     """

    #     self.query_embedding = self.embedding_model.embed(query_audio_samples).embeddings[0, 0]
    #     log.info('Query audio embedded.')
    #     return self.query_embedding


    def embed_query(self, query_audio_samples):
        """
        Embeds a single query audio clip using the worker process.
        """
        model_name = "perch_8" 
        
        future = self._embed_runner.submit(
            embed_worker.run_embed_task, 
            model_name, 
            query_audio_samples
        )
        
        embedding_numpy = future.result()
        self.query_embedding = embedding_numpy
        log.info('Query audio embedded (via worker).')
        
        return self.query_embedding
    

    async def embed_query_async(self, query_audio_samples):
        """
        Async version. ESSENTIAL for Panel to avoid deadlocking the event loop.
        """
        import asyncio
        model_name = "perch_8"
        
        # Get the Panel event loop
        loop = asyncio.get_running_loop()

        # "await" the result without freezing the main thread
        # This allows the main thread to receive signals from the worker
        embedding_numpy = await loop.run_in_executor(
            self._embed_runner, 
            embed_worker.run_embed_task, 
            model_name, 
            query_audio_samples
        )
        
        self.query_embedding = embedding_numpy
        log.info('Query audio embedded (via worker, async).')
        return self.query_embedding
    

    def shutdown(self):
            self._embed_runner.shutdown()

    def get_random_labeled_examples(self, label, label_type=1, num=5, dataset_name=None):

        label_type = [
            hoplite_interface.LabelType.NEGATIVE,
            hoplite_interface.LabelType.POSITIVE
        ][label_type]
    
        embedding_ids = self.db.get_embeddings_by_label(label, label_type)
        
        indices = sorted(random.sample(range(len(embedding_ids)), min(num, len(embedding_ids))))
        embedding_ids = [embedding_ids[i] for i in indices]

        examples = []

        for eid in embedding_ids:
            source = self.db.get_embedding_source(eid)

            examples.append({
                "eid": eid,
                "dataset_name": dataset_name,
                "source_id": source.source_id,
                "offset": source.offsets[0],
                "score": 0 # we don't have a score, but currently EmbeddingDisplay widget requires it.
            })
        return examples



    def search_with_query(self, num_results=50, sample_size=1_000_000, target_score=None, target_label=None, dataset='from_config'):
        if dataset == 'from_config':
            dataset = self.config.search_dataset_name

        if getattr(self, "query_embedding", None) is None:
            raise RuntimeError("query_embedding is not set. Call embed_query(audio_window) first.")
        
        if target_label is None:
            labels = self.get_target_labels()
            if len(labels) == 1:
                target_label = labels[0]
                log.info(f'No target label specified. Using the only available label: {target_label}')
            elif len(labels) > 1:
                log.info(f'No target label specified. Available labels: {labels}')
                return
            else:
                log.warning('No labels set for the classifier. Please set a label first.')
                return
            
        self.search(
            query=self.query_embedding,
            target_label=target_label,
            num_results=num_results,
            sample_size=sample_size,
            target_score=target_score,
            dataset=dataset,
            exclude_labeled=True)
    



    def search(self, query, target_label, bias=0.0, num_results=50, sample_size=1_000_000, 
              target_score=None,  dataset=None, exclude_labeled=True):
        """
        Searches the db using a query or model
        query: np.array of shape (embedding_dim,) (the result of self.embedding_model.embed())
              OR the result of classifier.train_linear_classifier() (kind of)
        """

        log.info(f'Searching for label {target_label} with num_results={num_results}, sample_size={sample_size}, target_score={target_score}, dataset={dataset}, exclude_labeled={exclude_labeled}')

        score_fn = score_functions.get_score_fn('dot', bias=bias, target_score=target_score)
        all_datasets = self.db.get_dataset_names()
        if dataset is not None and dataset not in all_datasets:
            log.warning(f'Dataset {dataset} not found in DB. Available datasets are {all_datasets}')
            return
        # results, all_scores = brutalism.threaded_brute_search(
        #     self.db, query, num_results, score_fn=score_fn,
        #     sample_size=sample_size, dataset=dataset, exclude_labeled=exclude_labeled)

        # TODO: exclude labelled, specifiy dataset
        results, all_scores = brutalism.threaded_brute_search(
            self.db, query, num_results, score_fn=score_fn,
            sample_size=sample_size, exclude_labeled=exclude_labeled)
        
        log.info(f'Search complete. Found {len(results.search_results)} results.')

        if self.search_results is None:
            self.search_results = {} # dict for each target label

        self.attach_metadata_to_search_results(results)

        self.search_results[str(target_label)] = results
        if getattr(self, "search_scores", None) is None:
            self.search_scores = {}

        self.search_scores[str(target_label)] = all_scores


    def attach_metadata_to_search_results(self, search_results):
        """
        fetches audio recording info for each search result source
        attaches a "source" and "metadata" to each result item. 
        """
        results_list = list(search_results)

        sources = {}
        domain_arids = {}
        for result in results_list:
            source = self.db.get_embedding_source(result.embedding_id)
            source_id = source.source_id
            result.source_id = source_id
            result.metadata: dict[str, Any] = {
                "source": source
            }
            if source_id not in sources:
                domain, arid = baw_utils.extract_domain_and_arid(source_id)
                sources[source_id] = {
                    "domain": domain,
                    "arid": arid,
                }
                if domain and arid is not None:
                    domain_arids.setdefault(domain, set()).add(arid)
                else:
                    log.warning(f"Could not extract domain and arid from source_id {source_id}")

            cached = sources.get(source_id, {})
            result.metadata["domain"] = cached.get("domain")
            result.metadata["arid"] = cached.get("arid")

        # for each domain, fetch the recording info for all arids in that domain in a single filter request
        domain_recording_info = {}
        for domain, arids in domain_arids.items():
            client = self.baw_api_clients.get(domain)

            if client is None:
                log.warning("No BAW API client found for domain %s; skipping metadata fetch", domain)
                continue

            response = client.post(
                "/audio_recordings/filter/?disable_paging=true",
                data={
                    "filter": {"id": {"in": sorted(arids)}},
                    "projection": {
                        "include": [
                            "id",
                            "site_id",
                            "canonical_file_name",
                            "sites.name",
                            "regions.name",
                            "regions.id",
                            "duration_seconds",
                            "recorded_date"
                        ]
                    },
                },
            )

            if response is None:
                log.warning("BAW API returned no response for domain %s", domain)
                continue

            records = response.get("data", [])
            # dict of arid to recording info for this each domain
            domain_recording_info[domain] = {
                rec.get("id"): rec for rec in records if rec.get("id") is not None
            }

        for result in results_list:
            recording_info = domain_recording_info.get(result.metadata.get("domain"), {}).get(result.metadata.get("arid"))
            result.metadata["recording_info"] = recording_info

        return search_results





    def _create_audio_loader(self) -> Callable[[str], np.ndarray]:
        """
        Returns a single, unified audio loader function.
        This function can handle both local file paths and remote BAW URLs.
        """
        # baw_config is expected to be a dict of {domain: token}
        window_size_s = self.db_model_config.model_config.window_size_s
        sample_rate_hz = self.db_model_config.model_config.sample_rate

        # untested attempt to support non-baw-url sources (local audio paths)
        filepath_loader = None
        if self.db_model_config:
            embed_config = self.db.get_metadata('audio_sources')

            # make_filepath_loader returns a callable that accepts a uri and offset
            filepath_loader = audio_loader.make_filepath_loader(
                        audio_sources=source_info.AudioSources.from_config_dict(embed_config),
                        window_size_s=window_size_s,
                        sample_rate_hz=sample_rate_hz,
                    )

        # This is the actual loader function that will be returned.
        # It has access to 'self' from the outer scope, so it can use stuff like self.config 
        def _loader(uri: str, offset: float | int) -> np.ndarray:

            baw_token = None
            baw_domain = None
            if self.config.baw_config:
                for domain, token in self.config.baw_config.items():
                    if domain and uri.startswith(domain):
                        baw_domain = domain
                        baw_token = token or ''
                        break

            if baw_domain:
                # It's a URL, so use the baw utils to download and load the audio data.
                # TODO: We need to support public projects that don't need a token.
                try:

                    segment_url = baw_utils.make_baw_audio_url_from_file_id(uri, offset, window_size_s, baw_domain)
                    # we can just use the load_baw_audio function directly since we are not doing multi loading anymore
                    audio_segment = baw_utils.load_baw_audio(segment_url, sample_rate=sample_rate_hz, auth_token=baw_token)
                    return audio_segment
                except Exception as e:
                    log.error(f"Failed to load BAW audio from {uri}: {e}")
                    # Return an empty array on failure so the UI doesn't crash.
                    return np.array([])
            else:

                # It's not a url belonging to our baw config domain, so treat it as a local file path.
                # If the db metadata allowed us to create a filepath_loader, use that (it prepends base paths, etc)
                # Otherwise, fallback to just loading the path directly. 
                if filepath_loader:
                    return filepath_loader(uri, offset)
                else:
                    return np.array(
                        audio_io.load_audio_window(
                            uri,
                            offset_s=offset,
                            sample_rate=sample_rate_hz,
                            window_size_s=window_size_s,
                        )
                    )


        return _loader


    def save_labels(self, verified_labels):
    
        labels_before = get_label_counts(self.db)
        success_count = 0
        failed_count = 0
        for lbl in verified_labels:
            check = self.db.insert_label(lbl, skip_duplicates=True)
            success_count += int(check)
            failed_count += int(not check)
        self.db.commit()
        labels_after = get_label_counts(self.db)
        return labels_before, labels_after, success_count, failed_count

    def upsert_label(self, embedding_id: int, label_name: str, label_type, provenance: str) -> None:
        """Insert or update a single label, then commit."""
        existing = [l for l in self.db.get_labels(embedding_id) if l.label == label_name]
        if existing:
            self.db.update_label_type(embedding_id, label_name, label_type)
        else:
            lbl = hoplite_interface.Label(
                embedding_id=embedding_id,
                label=label_name,
                type=label_type,
                provenance=provenance,
            )
            self.db.insert_label(lbl, skip_duplicates=False)
        self.db.commit()

    def delete_label(self, embedding_id: int, label_name: str) -> None:
        """Remove a label for an embedding, then commit."""
        self.db.delete_label(embedding_id, label_name)
        self.db.commit()


    def train_classifier(self,
                        learning_rate = 1e-3,
                        weak_neg_weight = 0.05,
                        num_steps = 128,
                        train_ratio = 0.9,
                        batch_size = 128,
                        weak_negatives_batch_size = 128,
                        balance_dataset: bool = True):
        target_labels = self.get_target_labels()
        resolved_weak_negatives_batch_size = max(int(weak_negatives_batch_size), 0)

        params = training_service.TrainClassifierParams(
            learning_rate=learning_rate,
            weak_neg_weight=weak_neg_weight,
            num_steps=num_steps,
            train_ratio=train_ratio,
            batch_size=batch_size,
            weak_negatives_batch_size=resolved_weak_negatives_batch_size,
        )
        log.info('Training for target labels: %s', target_labels)

        result = training_service.train_classifier(
            db=self.db,
            target_labels=target_labels,
            params=params,
            balance_dataset=balance_dataset,
        )

        # Keep a single source of truth in state: model + metrics + training_info.
        self.train_result = result
        return training_service.training_result_to_json(result)

    def has_trained_classifier(self) -> bool:
        return bool(
            self.train_result is not None
            and getattr(self.train_result, 'classifier_model', None) is not None
        )

    def get_trained_classifier(self):
        if not self.has_trained_classifier():
            raise ValueError('No classifier available. Train a classifier first.')
        return self.train_result.classifier_model

    def calculate_balanced_weak_negatives_batch_size(self,
                        weak_neg_weight = 0.05,
                        batch_size = 128) -> tuple[int, float, str]:
        target_labels = self.get_target_labels()

        return training_service.calculate_balanced_weak_negatives_batch_size(
            db=self.db,
            target_labels=target_labels,
            weak_neg_weight=weak_neg_weight,
            batch_size=batch_size,
        )







    def search_with_classifier(self,
          num_results=50, sample_size=1_000_000, target_score=None, dataset=None, target_labels=None):
        """
        For each class in the classifier, 
        Prepares the query and bias from the classifier params and searches with these
        """
        
        # unless we specify target labels, search for everything we trained on
        classifier = self.get_trained_classifier()
        if target_labels is None:
            target_labels = classifier.classes
            log.info("target_labels: %s", ','.join(target_labels))
        
        for target_label_idx, target_label in enumerate(target_labels):

            log.info('Searching for %s ...', target_label)

            class_query = classifier.beta[:, target_label_idx]
            bias = classifier.beta_bias[target_label_idx]

            # don't return anything, it saves to the self.search_results dict    
            self.search(
                query=class_query,
                target_label=target_label,
                bias=bias,
                num_results=num_results,
                sample_size=sample_size,
                target_score=target_score, 
                dataset=dataset,
                exclude_labeled=True)
      
  
    def get_status(self) -> dict:
        """
        Returns a dict with the status of the agile state
        - db_loaded: Whether the database is loaded
        - embedding_runtime_ready: Whether the worker-based embedding runtime is ready
        - embedding_model_config: The configuration of the embedding model
        - db_metadata: Metadata stored in the hoplite DB
        - database: A DatabaseStatus snapshot (or None when no DB is loaded)
        """

        embedding_runtime_ready = (
            getattr(self, "_embed_runner", None) is not None
        )

        status = {
            "config": self.config,
            "db_loaded": self.db is not None,
            "embedding_runtime_ready": embedding_runtime_ready,
            # Backwards-compatible key used by some UI code; now reflects worker runtime readiness.
            "embedding_model_loaded": embedding_runtime_ready,
            "embedding_model_config": self.db_model_config,
            "db_metadata": self.db.get_metadata(None) if self.db else None,
            "database": self.get_database_status() if self.db else None,
        }

        return status

    def get_database_status(self, include_health: bool = False):
        """Return a consolidated DatabaseStatus snapshot for the loaded database.

        This is the single entry point for database status. Cheap sections
        (existence, counts, datasets, label counts, index size) are always
        populated; the expensive consistency check is only included when
        ``include_health=True``.
        """
        if not self.db:
            raise ValueError("Database not initialized.")
        return get_database_status(self.db, include_health=include_health)

    def database_exists(self) -> bool:
        """True if a database exists on disk at the configured path.

        Path-only check that does not require the database to be loaded; used to
        drive UI state before the engine has initialised a database.
        """
        return database_exists_at(self.config.db_path)

    def repair_db_index(self, progress_callback=None, stop_event=None):
        """Rebuild the usearch index by re-downloading parquet files."""
        if not self.db:
            raise ValueError("Database not initialized.")
        if not self.baw_api_clients:
            raise ValueError("No BAW API clients available. Configure authentication first.")
        try:
            stats = repair_index(
                self.db,
                self.baw_api_clients,
                progress_callback=progress_callback,
                stop_event=stop_event,
            )
            log.info("Index repair complete: %s", stats)
            return stats
        except Exception:
            log.exception("Index repair failed")
            raise

    def delete_usearch_index(self):
        """Delete the usearch index file from disk (SQLite tables are untouched).

        After unlinking the file we reload the database so the in-memory usearch
        index reflects the on-disk state. The loaded ``db.ui`` is a live
        (memory-mapped) index; deleting the file alone leaves it populated, so
        the health check would keep reporting the deleted embeddings as present
        and a later commit could write the stale index back to disk.
        """
        if not self.db:
            raise ValueError("Database not initialized.")
        freed_bytes = _delete_usearch_index(self.db)
        # Reload via the single DB-loading path so the stale in-memory index is
        # replaced by a fresh empty one. The SQLite file still exists, so this
        # rebuilds an empty index ready to be repopulated by repair_db_index().
        self.init_db()
        return freed_bytes

    def get_baw_filters(self, dataset_priority=None, fallback_to_legacy=True):
        """Return the recording filter(s) stored in the database metadata.

        Looks up the most recent insertion filter for each dataset in
        *dataset_priority* order (defaults to ``fetched_labels`` then
        ``search_set``), falling back to the legacy ``baw_filters`` metadata key
        when *fallback_to_legacy* is true. Returns ``None`` when nothing is found.

        This is the single non-UI resolver for "what filter is associated with
        this database"; the filter editor widget reads through it.
        """
        dataset_priority = dataset_priority or ['fetched_labels', 'search_set']

        # Check baw_insertions first (last filter used per dataset, in priority order)
        for dataset_name in dataset_priority:
            last_filter = get_last_filter_for_dataset(self.db, dataset_name)
            if last_filter is not None:
                return [last_filter]

        # Legacy fallback: read from old baw_filters metadata key
        if fallback_to_legacy:
            try:
                baw_filters = self.db.get_metadata('baw_filters')
            except KeyError:
                baw_filters = None
            if baw_filters:
                return baw_filters.get('filters', None)

        log.warning("No baw_filters found in db metadata")
        return None
    

    def set_baw_filters(self, baw_filters: list):
        """returns a list of audio_recording filter critera used to create the database
        This is saved as hoplite db metadata and is used to get events from the workbench for
        adding labels to embeddings
        """

        # db metadata needs to be a config dict
        metadata = config_dict.ConfigDict({'filters': baw_filters})
        self.db.insert_metadata('baw_filters', metadata)
        self.db.commit()


    def fetch_labels(self, baw_filters=None):
        """
        For the configured target labels, fetches labels from the workbench using the
        supplied *baw_filters*. When *baw_filters* is ``None`` it falls back to the
        filter(s) stored in the database metadata.
        """

        self.labels_fetch_state = 'running'
        self.labels_fetch_error = None
        self.labels_fetch_message = 'Fetching labels from workbench...'

        try:
            if baw_filters is None:
                baw_filters = self.get_baw_filters()
            if not baw_filters:
                log.warning("No baw_filters found in db metadata. Cannot update labels.")
                self.fetched_label_result_sets = []
                self.fetched_label_outcomes = None
                self.labels_fetch_state = 'success'
                self.labels_fetch_message = 'No filters available to fetch labels.'
                return []
            
            all_results = []

            if self.config.target_labels is None or len(self.config.target_labels) == 0:
                raise ValueError("No target labels set in config. Cannot fetch labels.")

            # baw_filters is a list, so repeat the process for each one
            for filter_params in baw_filters:
                labels, counts = labels_from_event_taggings(self.active_client, filter_params, self.config.target_labels)
                all_results.append({
                    "filter_params": filter_params,
                    "counts": counts,
                    "labels": labels
                })

            self.fetched_label_result_sets = all_results
            self.fetched_label_outcomes = None
            self.labels_fetch_state = 'success'
            self.labels_fetch_message = (
                f'Fetched {self.fetched_labels_count} labels across {len(all_results)} result sets.'
            )

            return all_results
        except Exception as e:
            self.fetched_label_result_sets = None
            self.fetched_label_outcomes = None
            self.labels_fetch_state = 'error'
            self.labels_fetch_error = str(e)
            self.labels_fetch_message = f'Fetch failed: {e}'
            raise


    def commit_fetched_labels(self, progress_callback=None):
        """Commit labels currently stored in engine fetched-label state."""
        self.labels_commit_state = 'running'
        self.labels_commit_error = None
        self.labels_commit_message = 'Adding fetched labels to database...'

        try:
            if not self.has_fetched_labels_ready:
                raise ValueError("No fetched labels are ready to commit.")

            outcomes = self.commit_labels(self.fetched_label_result_sets, progress_callback=progress_callback)
            self.fetched_label_outcomes = outcomes

            added_count = sum(
                1
                for filter_outcomes in outcomes
                for source_data in filter_outcomes.get('outcomes', {}).values()
                for label_data in source_data.values()
                for type_data in label_data.values()
                for item in type_data
                if item.get('outcome') == 'added'
            )
            self.labels_commit_state = 'success'
            self.labels_commit_message = f'Added {added_count} labels to the database.'
            return outcomes
        except Exception as e:
            self.labels_commit_state = 'error'
            self.labels_commit_error = str(e)
            self.labels_commit_message = f'Add failed: {e}'
            raise


    def commit_labels(self, result_sets, progress_callback=None):
        """
        Commits the provided result_sets to the DB.
        The Engine injects the DB and API client automatically.
        @param result_sets: List of dicts, returned from self.fetch_labels()
        """
        if not result_sets:
            raise ValueError("No result sets provided to commit.")

        if not self.active_client:
             raise ValueError("BAW API client not initialized.")
        
        if not self.db:
            raise ValueError("Database not initialized.")

        all_outcomes_by_filter = []

        total_filters = len(result_sets)

        def _emit_filter_progress(event, *, filter_index, filter_params):
            if not progress_callback:
                return
            progress_callback({
                **event,
                'filter_index': filter_index,
                'total_filters': total_filters,
                'filter_params': filter_params,
            })

        for i, result_set in enumerate(result_sets):
            # The Engine does the heavy lifting of wiring up the dependencies
            filter_params = result_set.get('filter_params', {})
            cur_filter_index = i + 1

            if progress_callback:
                progress_callback({
                    'stage': 'filter_start',
                    'filter_index': cur_filter_index,
                    'total_filters': total_filters,
                    'filter_params': filter_params,
                    'message': f'Adding labels for filter {cur_filter_index}/{total_filters}...'
                })

            _progress_for_filter = functools.partial(
                _emit_filter_progress,
                filter_index=cur_filter_index,
                filter_params=filter_params,
            )

            outcomes = add_labels_to_db(
                db=self.db, 
                config=self.config, 
                items=result_set["labels"], 
                api_client=self.active_client,
                filter_dict=filter_params or None,
                progress_callback=_progress_for_filter,
            )

            if progress_callback:
                progress_callback({
                    'stage': 'filter_complete',
                    'filter_index': cur_filter_index,
                    'total_filters': total_filters,
                    'filter_params': filter_params,
                    'message': f'Finished filter {cur_filter_index}/{total_filters}.'
                })

            all_outcomes_by_filter.append({
                'filter_desc': f"Filter {i+1}",
                'filter_params': filter_params,
                'outcomes': outcomes
            })

        return all_outcomes_by_filter


    def save_classifier(self, path=None):
        classifier = self.get_trained_classifier()
        if path is None:
            path = Path(self.config.models_folder) / 'classifier.json'
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # I prefer it to save pretty. Perch hoplite saves compact. 
        # This hack just opens it, prettifies it, and saves it again.
        classifier.save(output_path)
        try:
            with output_path.open('r', encoding='utf-8') as f:
                classifier_json = json.load(f)
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(classifier_json, f, indent=2)
                f.write('\n')
        except (OSError, ValueError) as exc:
            log.warning('Could not prettify classifier JSON at %s: %s', output_path, exc)
      

    def run_inference(self, output_filepath=None, threshold=0.0, labels=None, dataset=None, subset=1, exclude_labeled=False):
        """
        Calls the classifier.write_inference_csv() function to write the inference results to a csv file
        injecting the site_id and link columns via a row_func that uses baw_utils and 
        """

        classifier = self.get_trained_classifier()
        if output_filepath is None:
            output_filepath = Path(self.config.predictions_folder) / 'inference.csv'
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        # not self.run_inference
        inference_service_run_inference(classifier,
                        self.db,
                        output_filepath,
                        self.active_client,
                        threshold=threshold,
                        labels=labels,
                        subset=subset,
                        exclude_labeled=exclude_labeled)
        log.info(f'Inference results written to {output_filepath}')
    

    
    
    def upload_detections(self, 
                          csv_path, 
                          project_id=None, 
                          limit=None):
        """
        Uploads the detections from a csv file to the BAW API.
        cvs_path: path to the csv file containing the detections
        project_id: the ID of the project. This is only used for the verify URL.
                    if not provided, it will just show a link with a placeholder. 

        """
        #todo: maybe project id or current date too
        upload_name = f"Agile Modelling: {self.get_target_label_description()}"
        success_status, link = upload_detections(csv_path=csv_path, upload_name=upload_name, 
                          project_id=project_id, limit=limit, baw_api=self.active_client)
        return success_status, link
        

    def set_active_client_by_domain(self, domain):
        """
        Sets the active BAW API client based on the domain.
        This is useful when we have multiple clients for different domains.
        """
        if domain in self.baw_api_clients:
            self.active_client = self.baw_api_clients[domain]
            log.info(f"Active client set to domain {domain}")
        else:
            log.warning(f"No BAW API client found for domain {domain}. Active client not changed.")


    def init_workbench(self, allow_prompt=True):
        """
        Inits our workbench client for interacting with Ecosounds. 
        """
        log.debug(f"Initializing workbench")

        if self.config.baw_config is None:
            log.warning("No BAW config found in config. Workbench features will be disabled.")
            return
        
        self.baw_api_clients = {}
        self.active_client = None

        for domain, token in self.config.baw_config.items():

            parsed_url = urlparse(domain)
            use_anonymous = token is None

            cur_client = baw_api(
                token=token,
                base_url=domain,
                allow_prompt=allow_prompt,
                anonymous=use_anonymous,
            )

            self.baw_api_clients[domain] = cur_client

            if cur_client and not self.active_client:
                self.active_client = cur_client
                log.info(f"Set active workbench client to domain {domain}")

            cache_path = Path(self.config.config_path).parent / f"tags_cache_{parsed_url.netloc}.json"
            try:
                self.baw_api_clients[domain].enable_tag_cache(cache_path)
            except Exception as e:
                log.warning(f"Failed to enable tag cache for domain {domain} at {cache_path}: {e}")

    def get_workbench_auth_statuses(self) -> dict[str, dict]:
        statuses = {}
        for domain, client in self.baw_api_clients.items():
            status = getattr(client, "auth_status", None)
            if isinstance(status, dict):
                statuses[domain] = status
            else:
                statuses[domain] = {
                    "state": "unknown",
                    "message": None,
                    "http_status": None,
                    "checked_at": None,
                    "base_url": domain,
                    "has_token": bool(getattr(client, "token", None)),
                }
        return statuses
            
        
            


    def fetch_workbench_tags(self):
        """
        """
        self.workbench_tags = []
        workbench_tags = []

        for domain, client in self.baw_api_clients.items():

            # get all taxonomic tags from the workbench
            if client is None:
                log.warning(f"Tags not fetched for {domain}, BAW API is not initialized")
                return False
            else:
            
                response = client.baw_filter('/tags', {'is_taxonomic': {'eq': True}}, qsp={"disable_paging":"true"})
                
                if response is None:
                    log.warning("Tags not fetched, BAW API request failed")
                else:
                    workbench_tags.extend([l['text'] for l in response.get('data', [])])
                    log.info(f"Fetched {len(self.workbench_tags)} workbench tags")

        self.workbench_tags = sorted(list(set(workbench_tags)))
        return self.workbench_tags

        
                

    # def fetch_events(self):
    #     """
    #     Fetches labels based on current filters. 
    #     Returns the data directly to the caller.
    #     """
    #     if not self.db: raise ValueError("Database not initialized")
        
    #     # 1. Get filters
    #     filters = self.get_baw_filters()
        
    #     # 2. Call your internal logic/helper to get the labels
    #     # (Assuming you have this logic ready to return 'all_results')
    #     # all_results = ... 
        
    #     return all_results


    def get_target_labels(self):
        """
        """
        return [str(lbl) for lbl in self.config.target_labels]
    

    def get_target_label_description(self):
        """
        Returns a string describing the target labels, 
        e.g. "species (call_type1, call_type2); species2"
        """

        description_dict = {}
        for lbl in self.config.target_labels:

            if lbl.species not in description_dict:
                description_dict[lbl.species] = []
            if lbl.call_type:
                description_dict[lbl.species].append(lbl.call_type)

        descriptions = []
        for species, call_types in description_dict.items():
            desc = species
            if call_types:
                desc += f" ({', '.join(call_types)})"
            descriptions.append(desc)

        return '; '.join(descriptions)
    









