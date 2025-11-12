"""Wrapper class for agile modeling steps."""



import dacite
import dataclasses
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random
import tempfile
import shutil

from tqdm import tqdm
from typing import Union, Dict, Any, List, Optional, Callable


from ml_collections import config_dict


import logging



from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
#from perch_hoplite.agile import embedding_display
#from recognizer_workshop.perch_hoplite_custom_modifications import embedding_display_group as embedding_display_mod
from perch_hoplite.agile import source_info
from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from perch_hoplite.db import score_functions
#from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db import interface as hoplite_interface
# from chirp.projects.zoo import models
from perch_hoplite.zoo import model_configs
from perch_hoplite import audio_io

import baw_helpers.baw_utils as baw_utils
from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_mod
from recognizer_workshop.perch_hoplite_custom_modifications import classifier as classifier_mod
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files, get_or_create_workshop_provenance
from baw_helpers.baw_api import baw_api
from baw_helpers.baw_helpers import annotation_upload_csv


from recognizer_workshop.app.services.events.upload_detections import upload_detections
from recognizer_workshop.app.services.inference.helpers import make_site_map
from recognizer_workshop.app.services.db_labels.db_labels import get_db_summary, get_label_counts, labels_from_event_taggings, add_labels_to_db
from recognizer_workshop.app.services.db_labels.dataclasses import LabelName
from recognizer_workshop.app.services.inference.inference import run_inference as inference_service_run_inference


log = logging.getLogger(__name__)

_APP_DATA_FILE = Path.home() / ".recognizer_workshop_data.json"
_MAX_RECENT_FILES = 10


@dataclasses.dataclass
class AgileConfig:
    """
    Configuration class for the Agile Recognizer workshop.
    Manages loading, saving, and validating configuration settings, especially file paths.
    """

    # Note: type hinting is important here so dacite can serialize and deserialize things correctly

    # --- Paths ---
    db_path: Path | None = None
    embeddings_folder: Path | None = None
    labeled_examples_folder: Path | None = None
    models_folder: Path | None = None
    predictions_folder: Path | None = None

    # --- Configuration ---
    annotator_id: str = "default_annotator"
    search_dataset_name: str = 'search_set'
    max_embeddings_count: int = -1
    baw_config: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # list of species labels and optional call types
    target_labels: List[LabelName] = dataclasses.field(default_factory=list) 

    # Path to the current config file after a load or save_as, not part of the JSON serialization.
    config_path: Path | None = dataclasses.field(default=None, init=False, repr=False)

    _APP_DATA_FILE = Path.home() / ".recognizer_workshop_data.json"
    _MAX_RECENT_FILES = 10
    
    # List of fields that should be treated as paths
    _PATH_FIELDS: dataclasses.InitVar[List[str]] = [
        'db_path', 'embeddings_folder', 'labeled_examples_folder', 
        'models_folder', 'predictions_folder'
    ]

    def __post_init__(self, *args, **kwargs):
        """
        Called after initialization, for validation.
        """
        log.info("Running validation...")

        # convert all path strings to Path objects
        for file_name in self._PATH_FIELDS:
            setattr(self, file_name, Path(getattr(self, file_name)) if getattr(self, file_name) is not None else None)

        # Check if essential paths exist
        # Note: We check the PARENT of db_path, as the file may not exist yet.
        if self.db_path and not self.db_path.parent.exists():
            log.warning(f"Database directory does not exist: {self.db_path}. You may need to create it.")
        
        # we automatically create all folders if they don't exist, except for the db one, 
        # because it needs to exist before we start (this might change)
        for field_name in self._PATH_FIELDS:
            if 'db_path' in field_name: continue # already checked
            path_value = getattr(self, field_name)
            if path_value and not path_value.exists():
                log.info(f"Path for '{field_name}' does not exist: {path_value}. Creating it.")
                path_value.mkdir(parents=True, exist_ok=True)


    def reset(self):
        """Resets the config to its default state, clearing the current path."""
        # Create a new default instance and copy its values to this one.
        default_instance = AgileConfig()
        for field in dataclasses.fields(self):
            if field.init: # Don't overwrite fields like 'config_path'
                setattr(self, field.name, getattr(default_instance, field.name))
        self.config_path = None
        log.info("New configuration created.")


    def save(self):
        if not self.config_path:
            raise ValueError(
                "No file path is set for this configuration. "
                "Use 'save_as(path)' to save it for the first time."
            )
        self._to_json(self.config_path)
        log.info(f"Configuration saved to {self.config_path}")


    def save_as(self, path):
        self.config_path = Path(path).resolve()
        self._to_json(self.config_path)
        _add_to_recent_files(self.config_path)
        log.info(f"Configuration saved to new location: {self.config_path}")

    def __str__(self) -> str:
        """Readable string representation of the config."""
        lines = ["Agile Recognizer Config:"]
        for field in dataclasses.fields(self):
            if field.name.startswith('_'): continue # Skip private fields
            
            value = getattr(self, field.name)
            
            # Mask sensitive information like auth tokens if they exist
            if 'auth_token' in str(field.name) and isinstance(value, str) and len(value) > 2:
                value = f"{value[0]}**********{value[-1]}"
            
            lines.append(f"  - {field.name}: {value}")
        return "\n".join(lines)


    @classmethod
    def load(cls, path: str | Path) -> 'AgileConfig':
        """Loads a config from a path and updates the recent files list."""
        load_path = Path(path).resolve()

        if not load_path.exists():
            raise FileNotFoundError(f"Config file not found: {load_path}")

        instance = cls.from_json(load_path)
        instance.config_path = load_path
        _add_to_recent_files(load_path)
        return instance




    @classmethod
    def from_json(cls, config_path: str | Path, relative_to: str | Path | None = None) -> 'AgileConfig':
        """
        Loads configuration from a JSON file and returns a new AgileConfig instance.
        Paths within the JSON are resolved relative to a specified base directory.

        Args:
            config_path: The path to the JSON configuration file.
            relative_to: The base directory for resolving relative paths.
                         - If None (default), paths are relative to the config file's location.
                         - If ".", paths are relative to the current working directory.
                         - If a specific path is given, paths are relative to that directory.
        """
        config_path = Path(config_path).resolve()
        
        if not config_path.exists():
            log.warning(f"Config file not found at {config_path}. Creating a default one.")
            # Create a default instance and save it so it exists for next time
            default_config = cls()
            default_config._to_json(config_path)
            return default_config

        # Determine the base directory for path resolution
        if relative_to is None:
            base_dir = config_path.parent
        else:
            base_dir = Path(relative_to).resolve()

        with open(config_path, 'r') as f:
            data = json.load(f)

        # Resolve paths: if a path in the JSON is not absolute, join it with the base_dir
        for field_name in cls._PATH_FIELDS:
            if field_name in data and data[field_name] is not None:
                path_from_json = Path(data[field_name])
                if not path_from_json.is_absolute():
                    data[field_name] = base_dir / path_from_json

        instance = dacite.from_dict(
            data_class=cls,
            data=data,
            config=dacite.Config(type_hooks={Path: Path}),
        )

        return instance
    

 

    def _to_json(self, config_path: str | Path, relative_to: str | Path | None = None) -> None:
        """
        Saves the current configuration to a JSON file.
        Path objects are converted to strings, relative to a specified base directory.

        Args:
            config_path: The path where the JSON file will be saved.
            relative_to: The base directory for making paths relative.
                         - If None (default), paths are relative to the saved config file's location.
                         - If ".", paths are relative to the current working directory.
                         - If "/", paths are saved as absolute paths.
                         - If a specific path is given, paths are relative to that directory.
        """
        config_path = Path(config_path).resolve()
        
        # Determine the base directory for creating relative paths
        if relative_to is None:
            base_dir = config_path.parent
        elif str(relative_to) == "/":
            # TODO: why can't we just make the base dir "/"
            base_dir = None 
        else:
            base_dir = Path(relative_to).resolve()
            
        config_path.parent.mkdir(parents=True, exist_ok=True)

        
        data_to_save = dataclasses.asdict(self)
        del data_to_save['config_path'] # don't include the config path
      
        for field_name in self._PATH_FIELDS:
            if field_name in data_to_save and data_to_save[field_name] is not None:
                absolute_path = Path(data_to_save[field_name])
                if base_dir is None: 
                    # Save as absolute path
                    saved_path = absolute_path.as_posix()
                else: 
                    # Save as relative path
                    saved_path = os.path.relpath(absolute_path, base_dir)
                
                data_to_save[field_name] = saved_path.replace("\\", "/")

        with open(config_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        log.info(f"Configuration saved to: {config_path}")



class agile_state:

    config: AgileConfig = None
    db: sqlite_usearch_impl_mod.ThreadedSQLiteUsearchDB = None
    db_model_config = None
    model_class = None
    _embedding_model = None
    audio_filepath_loader = None
    baw_api = None
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

    def __init__(self, config):
        self.config = config


    def initialize(self):
        self.init_db()


    def init_db(self):
        log.info('Initializing DB...')

        if not self.config.db_path:
            raise ValueError('DB path not set in config. Please set it before initializing the DB.')
        if not Path(self.config.db_path).exists():
            raise ValueError(f'DB path {self.config.db_path} does not exist.')
        
        self.db = sqlite_usearch_impl_mod.ThreadedSQLiteUsearchDB.create(self.config.db_path)
        self.db_model_config = self.db.get_metadata('model_config')
        
        self.model_class = model_configs.get_model_class(self.db_model_config.model_key)
        # self.embedding_model = self.model_class.from_config(self.db_model_config.model_config)


        num_embeddings = self.db.count_embeddings()
        log.info(f'DB initialized with {num_embeddings} embeddings.')
        self.db_summary()

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

    @property
    def embedding_model(self):
        """
        Lazy-loads the embedding model on first access and caches it.
        Tfhub attempts to cache downloaded models. This function handles cache clearing and retries,
        in the event that the cache is corrupted.
        """
        if self._embedding_model is None:
            log.info("Loading embedding model for the first time...")

            def clear_tfhub_cache():
                cache_dir = Path(tempfile.gettempdir()) / "tfhub_modules"
                if cache_dir.exists():
                    log.info(f"Corrupted cache detected. Deleting {cache_dir}...")
                    try:
                        shutil.rmtree(cache_dir)
                        log.info("Cache cleared successfully.")
                    except OSError as e:
                        log.info(f"Error clearing cache: {e}")

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    #TODO: load the model based on db config data rather than hardcoding "perch_8"
                    self._embedding_model = model_configs.load_model_by_name("perch_8")
                    #self._embedding_model = self.model_class.from_config(self.db_model_config.model_config)
                    log.info("Model loaded successfully.")
                    break
                except ValueError as e:
                    is_cache_error = 'saved_model.pb' in str(e) or 'saved_model.pbtxt' in str(e)
                    if is_cache_error and attempt < max_retries - 1:
                        log.info(f"Attempt {attempt + 1} failed. Looks like a cache issue.")
                        clear_tfhub_cache()
                        log.info("Retrying model load...")
                    else:
                        log.warning("Failed to load model after retries.")
                        raise e
        return self._embedding_model

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

    
    def embed_query(self, query_audio_samples):
        """
        Embeds a single query audio clip allowing the user to select the 5s window within the specified source
        """

        self.query_embedding = self.embedding_model.embed(query_audio_samples).embeddings[0, 0]
        log.info('Query audio embedded.')
        return self.query_embedding
    

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

        self.search_results[str(target_label)] = results
        if getattr(self, "search_scores", None) is None:
            self.search_scores = {}

        self.search_scores[str(target_label)] = all_scores



    def _create_audio_loader(self) -> Callable[[str], np.ndarray]:
        """
        Returns a single, unified audio loader function.
        This function can handle both local file paths and remote BAW URLs.
        """
        baw_domain = self.config.baw_config.get('domain')
        baw_token = self.config.baw_config.get('auth_token', '')

        window_size_s=self.db_model_config.model_config.window_size_s
        sample_rate_hz=self.db_model_config.model_config.sample_rate

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
        # It has access to 'self' from the outer scope, so it can use self.baw_api.
        def _loader(uri: str, offset: float | int) -> np.ndarray:
            if baw_domain and uri.startswith((baw_domain)):
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


    def train_classifier(self,
                        learning_rate = 1e-3,
                        weak_neg_weight = 0.05,
                        num_steps = 128,
                        train_ratio = 0.9,
                        batch_size = 128,
                        weak_negatives_batch_size = 128):
        
        target_labels = self.get_target_labels()

        data_manager = classifier_data.AgileDataManager(
            target_labels=target_labels,
            db=self.db,
            train_ratio=train_ratio,
            min_eval_examples=1,
            batch_size=batch_size,
            weak_negatives_batch_size=weak_negatives_batch_size,
            rng=np.random.default_rng(seed=5))
        log.info('Training for target labels : ')
        log.info(data_manager.get_target_labels())
        linear_classifier, eval_scores = classifier.train_linear_classifier(
            data_manager=data_manager,
            learning_rate=learning_rate,
            weak_neg_weight=weak_neg_weight,
            num_train_steps=num_steps,
        )
        log.info('\n' + '-' * 80)
        top1 = eval_scores['top1_acc']
        log.info(f'top-1      {top1:.3f}')
        rocauc = eval_scores['roc_auc']
        log.info(f'roc_auc    {rocauc:.3f}')
        cmap = eval_scores['cmap']
        log.info(f'cmap       {cmap:.3f}')
        # self.classifier_params = params
        # self.classifier_eval_scores = eval_scores

        self.classifier = linear_classifier

        return eval_scores







    def search_with_classifier(self,
          num_results=50, sample_size=1_000_000, target_score=None, dataset=None, target_labels=None):
        """
        For each class in the classifier, 
        Prepares the query and bias from the classifier params and searches with these
        """
        
        # unless we specify target labels, search for everything we trained on
        if target_labels is None:
            target_labels = self.classifier.classes
            log.info("target_labels: %s", ','.join(target_labels))
        
        for target_label_idx, target_label in enumerate(target_labels):

            log.info('Searching for %s ...', target_label)

            class_query = self.classifier.beta[:, target_label_idx]
            bias = self.classifier.beta_bias[target_label_idx]

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
        - embedding_model_loaded: Whether the embedding model is loaded
        - embedding_model_config: The configuration of the embedding model
        - db_summary: A summary of the database contents (self.db.summary())
        """

        status = {
            "config": self.config,
            "db_loaded": self.db is not None,
            "embedding_model_loaded": self._embedding_model is not None,
            "embedding_model_config": self.db_model_config,
            "db_summary": self.db_summary() if self.db else None,
        }

        return status

    def db_summary(self):
        """Debugging/sanity checking to see what's in the database"""
        return get_db_summary(self.db)
    

    def get_baw_filters(self):
        """returns a list of audio_recording filter critera used to create the database
        This is saved as hoplite db metadata and is used to get events from the workbench for
        adding labels to embeddings
        """
        try:
            baw_filters = self.db.get_metadata('baw_filters')
        except KeyError:
            baw_filters = None

        if not baw_filters:
            log.warning("No baw_filters found in db metadata")
            return None
        
        return baw_filters.get('filters', None)
    

    def set_baw_filters(self, baw_filters: list):
        """returns a list of audio_recording filter critera used to create the database
        This is saved as hoplite db metadata and is used to get events from the workbench for
        adding labels to embeddings
        """

        # db metadata needs to be a config dict
        metadata = config_dict.ConfigDict({'filters': baw_filters})

        self.db.insert_metadata('baw_filters', metadata)
        self.db.commit()


    def fetch_labels(self):

        baw_filters = self.get_baw_filters()
        if not baw_filters:
            log.warning("No baw_filters found in db metadata. Cannot update labels.")
            return "no_baw_filters"
        
        all_results = []

        # baw_filters is a list, so repeat the process for each one
        for filter_params in baw_filters:
            labels, counts = labels_from_event_taggings(self.baw_api, filter_params, self.config.target_labels)
            all_results.append({
                "filter_params": filter_params,
                "counts": counts,
                "labels": labels
            })

        return all_results

        

    def save_classifier(self, path=None):
        if path is None:
            path = Path(self.config.models_folder) / 'classifier.json'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save(path)
      

    def run_inference(self, output_filepath=None, threshold=0.0, labels=None, dataset=None, subset=1, exclude_labeled=False):
        """
        Calls the classifier.write_inference_csv() function to write the inference results to a csv file
        injecting the site_id and link columns via a row_func that uses baw_utils and 
        """

        if output_filepath is None:
            output_filepath = Path(self.config.predictions_folder) / 'inference.csv'
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        # not self.run_inference
        inference_service_run_inference(self.classifier,
                        self.db,
                        output_filepath,
                        self.baw_api,
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
                          project_id=project_id, limit=limit, baw_api=self.baw_api)
        return success_status, link
        


    def init_workbench(self, token=None, allow_prompt=True):
        """
        Inits our workbench client for interacting with Ecosounds. 
        """
        log.debug(f"Initializing workbench")

        if token is None:
            token = self.config.baw_config.get('auth_token', None)


        baw_config = getattr(self.config, 'baw_config', {}) or {}
        log.debug(f"BAW config: {baw_config}")


        self.baw_api = baw_api(token=token,
                               base_url=self.config.baw_config.get('domain', None),
                               allow_prompt=allow_prompt)

        if self.baw_api.token is not None:
            # token supplied was valid OR was loaded from env and valid OR was prompted for and valid
            # update our config to sync with the baw_api loaded token
            log.debug("Auth token set")
            self.config.baw_config['auth_token'] = self.baw_api.token
            return True
        else:
            log.warning("No auth token set")
            return False


    def fetch_workbench_tags(self):
        """
        """
        self.workbench_tags = []


        # get all taxonomic tags from the workbench
        if self.baw_api is None:
            log.warning("Tags not fetched, BAW API is not initialized")
            return True
        else:
        
            response = self.baw_api.baw_filter('/tags', {'is_taxonomic': {'eq': True}}, qsp={"disable_paging":"true"})
            
            if response is None:
                log.warning("Tags not fetched, BAW API request failed")
                return False
            else:
                self.workbench_tags = sorted([l['text'] for l in response.get('data', [])])
                log.info(f"Fetched {len(self.workbench_tags)} workbench tags")
                return True



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



# private module functions for saving/loading app data

def _get_app_data_file() -> Path:
    """Returns the path to the central app data file."""
    return AgileConfig._APP_DATA_FILE

def _read_app_data() -> dict:
    """Reads the central app data file."""
    if _get_app_data_file().exists():
        with open(_get_app_data_file(), 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def _write_app_data(data: dict):
    """Writes to the central app data file."""
    with open(_get_app_data_file(), 'w') as f:
        json.dump(data, f, indent=4)


def _add_to_recent_files(path: Path):
    """Adds a file path to the top of the recent files list."""
    data = _read_app_data()
    recent_files = data.get("recent_files", [])
    str_path = str(path)
    if str_path in recent_files:
        recent_files.remove(str_path)
    recent_files.insert(0, str_path)
    data["recent_files"] = recent_files[:AgileConfig._MAX_RECENT_FILES]
    _write_app_data(data)


def get_recent_files(num: int = AgileConfig._MAX_RECENT_FILES) -> list[str]:
    data = _read_app_data()
    return data.get("recent_files", [])[:num]