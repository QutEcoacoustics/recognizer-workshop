import json
from pathlib import Path
import numpy as np
import random
from typing import Optional, Callable, Any
from ml_collections import config_dict
import multiprocessing
import logging
from urllib.parse import urlparse

from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
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
from recognizer_workshop.app.services.events.upload_detections import upload_detections
from recognizer_workshop.app.services.db_labels.fetch_baw_labels import labels_from_event_taggings
from recognizer_workshop.app.services.db_labels.add_db_labels import add_labels_to_db
from recognizer_workshop.app.services.db_labels.db_labels_info import get_db_summary, get_label_counts
from recognizer_workshop.app.services.db_labels.dataclasses import LabelName
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

    def __init__(self, config):
        self.config = config
        mp_context = multiprocessing.get_context('spawn')
        self._embed_runner = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp_context
        )


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
        """
        For the configured target labels, fetches labels from the workbench based on the baw_filters 
        configured in the database metadata. 
        """

        baw_filters = self.get_baw_filters()
        if not baw_filters:
            log.warning("No baw_filters found in db metadata. Cannot update labels.")
            return "no_baw_filters"
        
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

        return all_results


    def commit_labels(self, result_sets):
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

        for i, result_set in enumerate(result_sets):
            # The Engine does the heavy lifting of wiring up the dependencies
            outcomes = add_labels_to_db(
                db=self.db, 
                config=self.config, 
                items=result_set["labels"], 
                api_client=self.active_client 
            )
            
            filter_params = result_set.get('filter_params', {})
            all_outcomes_by_filter.append({
                'filter_desc': f"Filter {i+1}",
                'filter_params': filter_params,
                'outcomes': outcomes
            })

        return all_outcomes_by_filter


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

        for domain, token in self.config.baw_config.items():

            parsed_url = urlparse(domain)

            cur_client = baw_api(
                token=token,
                base_url=domain,
                allow_prompt=allow_prompt
            )

            self.baw_api_clients[domain] = cur_client

            if cur_client and not self.active_client:
                self.active_client = cur_client
                log.info(f"Set active workbench client to domain {domain}")

            cache_path = Path(self.config.config_path).parent / f"tags_cache_{parsed_url.netloc}.json"
            self.baw_api_clients[domain].enable_tag_cache(cache_path)
            
        
            


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








