"""Wrapper class for agile modeling steps."""


from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import pandas as pd
import random

from tqdm import tqdm
from typing import Union


from ml_collections import config_dict


from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
#from perch_hoplite.agile import embedding_display
#from recognizer_workshop.perch_hoplite_custom_modifications import embedding_display_group as embedding_display_mod
from perch_hoplite.agile import source_info
from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from perch_hoplite.db import score_functions
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db import interface as hoplite_interface
# from chirp.projects.zoo import models
from perch_hoplite.zoo import model_configs


import baw_helpers.baw_utils as baw_utils
from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_mod
from recognizer_workshop.perch_hoplite_custom_modifications import classifier as classifier_mod
from recognizer_workshop.agile_modeling_helpers import list_audio_files, get_or_create_workshop_provenance
from baw_helpers.baw_api import baw_api
from baw_helpers.baw_helpers import annotation_upload_csv



from recognizer_workshop.events.upload_detections import upload_detections
from recognizer_workshop.inference.helpers import make_site_map
from recognizer_workshop.db_labels.db_labels import print_label_counts, db_summary
from recognizer_workshop.inference.inference import run_inference




@dataclass
class agile_config:

    # path to the sqlite db containing the embeddings
    db_path: Union[str, Path] = None

    # annotator id to attach to labels
    annotator_id: str = None

    # config for the baw api 
    baw_config: dict = field(default_factory=dict)

    # name of the dataset in the database we are working with
    search_dataset_name: str = 'search_set'

    # path to the embeddings files to create the database from
    embeddings_folder: str = None

    # path to labeled examples
    labeled_examples_folder: str = None
    
    # path to the folder where we will save the classifiers
    models_folder: str = None

    # path to the folder where we will save the inference results
    predictions_folder: str = None

    # max number of embeddings to load from the embeddings_folder
    max_embeddings_count: int = -1

    def from_json(self, json_path = "./agile_config.json"):
        # read a json file and populate the dataclass properties from that

        if not Path(json_path).exists():
            Warning(f'Config file {json_path} does not exist. Config not loaded.')
            return

        def resolve_path(path):
            if not Path(path).is_absolute():
                path = Path(json_path).parent / Path(path)
            return path
          
        with open(json_path, 'r') as f:
            # paths in config is relative to the working directory, 
            # paths in json config is relative to the json file
            data = json.load(f)
            paths_to_resolve = ['db_path', 'embeddings_folder', 'labeled_examples_folder']
            for key, value in data.items():
                if key in paths_to_resolve:
                    value = resolve_path(value)
                setattr(self, key, value)

        self.check()
        self.show_config()
    

    def show_config(self, conf_dict=None, depth=0):
        if depth == 0:
            print(f'Config:')
        if conf_dict is None:
            conf_dict = self.__dict__
        
        for key, value in conf_dict.items():
            # if value is a dict, recurse
            if isinstance(value, dict):
                print(f'{key}:')
                self.show_config(value, depth+1)
            else:
                if key in ['auth_token']:
                    # replace all but the 1st and last characters with *
                    value = value[0] + '*' * (len(value) - 2) + value[-1]
                print(f'{key}: {value}')


    def check(self):
        # check if the paths in the config exist
        if not Path(self.db_path).exists():
            Warning(f'DB path {self.db_path} does not exist.')


class agile_state:

    config: agile_config = None
    db: sqlite_usearch_impl.SQLiteUsearchDB = None
    db_model_config = None
    model_class = None
    embedding_model = None
    audio_filepath_loader = None
    baw_api = None
    label = {"value": None}  # this is used to store the label selected by the user in the label selector widget

    # search results from the last search, whether it's a single example query
    # or a classifier-based search.
    search_results = None

    # an object representing the visual display of search_results
    # including result of interactive user-labeling of these results. 
    # this is overwritten each time a search results are displayed. 
    display_group = None

    def __init__(self, config):
        self.config = config


    def initialize(self):
        self.init_db()


    def init_db(self):
        print('Initializing DB...')

        if not self.config.db_path:
            raise ValueError('DB path not set in config. Please set it before initializing the DB.')
        if not Path(self.config.db_path).exists():
            raise ValueError(f'DB path {self.config.db_path} does not exist.')
        
        self.db = sqlite_usearch_impl.SQLiteUsearchDB.create(self.config.db_path)
        self.db_model_config = self.db.get_metadata('model_config')
        
        self.model_class = model_configs.get_model_class(self.db_model_config.model_key)
        self.embedding_model = self.model_class.from_config(self.db_model_config.model_config)
        num_embeddings = self.db.count_embeddings()
        #print(f'DB initialized with {num_embeddings} embeddings.')
        self.db_summary()

        #TODO: This is a hack to get the audio loader to be created even when it's not used.  
        # seems that embed config metadata is not inserted during legacy conversion. 
        # it's wrapped in the legacy_config object instead. 
        # In the case of ecosounds recordings, this filepath loader is not really used, but
        # it needs to exist to pass to the display_group object.

        embed_config = self.db.get_metadata('audio_sources')
        self.audio_filepath_loader = audio_loader.make_filepath_loader(
          audio_sources=source_info.AudioSources.from_config_dict(embed_config),
          window_size_s=self.embedding_model.window_size_s,
          sample_rate_hz=self.embedding_model.sample_rate,
        )



    
    def embed_query(self, query_audio_samples):
        """
        Embeds a single query audio clip allowing the user to select the 5s window within the specified source
        """

        self.query_embedding = self.embedding_model.embed(query_audio_samples).embeddings[0, 0]
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
                print(f'No target label specified. Using the only available label: {target_label}')
            elif len(labels) > 1:
                print(f'No target label specified. Available labels: {labels}')
                return
            else:
                print('No labels set for the classifier. Please set a label first.')
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

        score_fn = score_functions.get_score_fn('dot', bias=bias, target_score=target_score)
        all_datasets = self.db.get_dataset_names()
        if dataset is not None and dataset not in all_datasets:
            print(f'Dataset {dataset} not found in DB. Available datasets are {all_datasets}')
            return
        # results, all_scores = brutalism.threaded_brute_search(
        #     self.db, query, num_results, score_fn=score_fn,
        #     sample_size=sample_size, dataset=dataset, exclude_labeled=exclude_labeled)

        # TODO: exclude labelled, specifiy dataset
        results, all_scores = brutalism.threaded_brute_search(
            self.db, query, num_results, score_fn=score_fn,
            sample_size=sample_size, exclude_labeled=exclude_labeled)

        if self.search_results is None:
            self.search_results = {} # dict for each target label

        self.search_results[target_label] = results
        if getattr(self, "search_scores", None) is None:
            self.search_scores = {}
        self.search_scores[target_label] = all_scores





    def save_labels(self, verified_labels):
        prev_lbls, new_lbls = 0, 0

        for lbl in verified_labels:
            check = self.db.insert_label(lbl, skip_duplicates=True)
            new_lbls += check
            prev_lbls += (1 - check)
        self.db.commit()
        print('\nnew_lbls: ', new_lbls)
        print('\nprev_lbls: ', prev_lbls)
        print_label_counts(self.db)
        return prev_lbls, new_lbls
    
    
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
        print('Training for target labels : ')
        print(data_manager.get_target_labels())
        linear_classifier, eval_scores = classifier.train_linear_classifier(
            data_manager=data_manager,
            learning_rate=learning_rate,
            weak_neg_weight=weak_neg_weight,
            num_train_steps=num_steps,
        )
        print('\n' + '-' * 80)
        top1 = eval_scores['top1_acc']
        print(f'top-1      {top1:.3f}')
        rocauc = eval_scores['roc_auc']
        print(f'roc_auc    {rocauc:.3f}')
        cmap = eval_scores['cmap']
        print(f'cmap       {cmap:.3f}')
        # self.classifier_params = params
        # self.classifier_eval_scores = eval_scores

        self.classifier = linear_classifier

        return eval_scores







    def search_with_classifier(self,
          num_results=50, sample_size=1_000_000, target_score=None, dataset=None):
        """
        For each class in the classifier, 
        Prepares the query and bias from the classifier params and searches with these
        """

        target_labels = self.classifier.classes
        print('target_labels: ', target_labels)
        
        for target_label_idx, target_label in enumerate(target_labels):

            print(f'Searching for {target_label}...')

            class_query = self.classifier.beta[:, target_label_idx]
            bias = self.classifier.beta_bias[target_label_idx]
                
            self.search(
                query=class_query,
                target_label=target_label,
                bias=bias,
                num_results=num_results,
                sample_size=sample_size,
                target_score=target_score, 
                dataset=dataset,
                exclude_labeled=True)
      
  


    def db_summary(self):
        """Debugging/sanity checking to see what's in the database"""
        if not self.db:
            print('DB not initialized.')
            return
        db_summary(self.db)


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

        run_inference(self.classifier,
                        self.db,
                        output_filepath,
                        self.baw_api,
                        threshold=threshold,
                        labels=labels,
                        subset=subset,
                        exclude_labeled=exclude_labeled)
    

    
    
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
        
        upload_name = f"Agile Modelling: {self.get_target_label_description()}"
        upload_detections(csv_path=csv_path, upload_name=upload_name, 
                          project_id=project_id, limit=limit, baw_api=self.baw_api)
        

    def init_workbench(self, token=None):
                
        # this will prompt for a token if not set
        self.baw_api = baw_api(token=token)

        if self.baw_api.token is not None:
            if self.config.baw_config.get('auth_token'):
                print("Overwriting config auth token with new value")
            self.config.baw_config['auth_token'] = self.baw_api.token
        elif self.config.baw_config.get('auth_token'):
            print("Auth token not loaded, but already in config")
        else:
            print("No auth token set")




    def get_target_labels(self):
        """
        """
        species = self.label.get('species', '')
        call_types = self.label.get('call_types') or []

        if not call_types:
            return [species]
        else:
            return [f"{species} {ct}" for ct in call_types]
        

    def get_target_label_description(self):
        """
        Returns a string describing the target label, e.g. "species (call_type1, call_type2)"
        """
        species = self.label.get('species', '')
        call_types = self.label.get('call_types') or []

        if not call_types:
            return species
        else:
            return f"{species} ({', '.join(call_types)})"



