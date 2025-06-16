# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper class for agile modeling steps."""


from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import pandas as pd
import random

from tqdm import tqdm

from matplotlib import pyplot as plt
from ml_collections import config_dict

from perch_hoplite import audio_io
from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
#from perch_hoplite.agile import embedding_display
from recognizer_workshop.perch_hoplite_custom_modifications import embedding_display
#from recognizer_workshop.perch_hoplite_custom_modifications import embedding_display_group as embedding_display_mod
from perch_hoplite.agile import source_info
from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from perch_hoplite.db import score_functions
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db import interface as hoplite_interface
# from chirp.projects.zoo import models
from perch_hoplite.zoo import model_configs
from recognizer_workshop.create_database import convert_parquet

import recognizer_workshop.baw_utils as baw_utils
from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_mod
from recognizer_workshop.perch_hoplite_custom_modifications import classifier as classifier_mod
from recognizer_workshop.agile_modeling_helpers import list_audio_files
from recognizer_workshop.baw_api import baw_api

from recognizer_workshop import agile_widgets

import ipywidgets as widgets
from IPython.display import display

@dataclass
class agile_config:

    # path to the sqlite db containing the embeddings
    db_path: str = None

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


    def display_query(self, query_uri: str | int): 
        """
        Displays the query audio so that the user can select the 5s window to embed
        """

        query_uri = str(query_uri)

        #xeno canto ids are uppercase on the website, audio loader requires lowercase
        if query_uri.startswith('XC'):
            query_uri = 'xc' + query_uri[2:]

        # if the query is a path relative to the self.conifg.labeled_examples_folder, resolve its full path
        if (self.config.labeled_examples_folder / Path(query_uri)).exists():
            query_uri = self.config.labeled_examples_folder / Path(query_uri)

        # if the query is an integer it's an index into the list of audio files in the labeled_examples_folder
        elif str(query_uri).isdigit():
            file_list =  list_audio_files(self.config.labeled_examples_folder, quiet=True)
            if int(query_uri) < len(file_list):
                query_uri = file_list[int(query_uri)]
            else:
                raise ValueError(f'Index {query_uri} is out of range of the labeled examples folder')
          
        self.query_display = embedding_display.QueryDisplay(
            uri=query_uri, offset_s=0.0, window_size_s=5.0, sample_rate_hz=32000)
        
        self.query_display.display_interactive()

    
    def embed_query(self):
        """
        Embeds a single query audio clip allowing the user to select the 5s window within the specified source
        """

        self.query_embedding = self.embedding_model.embed(self.query_display.get_audio_window()).embeddings[0, 0]


    def add_labeled_examples(self, labeled_examples_folder, label, dataset_name, sample_rate_hz = 32000):
        """
        Adds labeled examples to the database from a folder containing audio files
        """

        window_size_s = 5.0

        labeled_examples_folder = Path(labeled_examples_folder)
        labeled_examples = list_audio_files(labeled_examples_folder)

        for example_source in labeled_examples:

            print(f'Adding {label} example to db:{dataset_name}: {example_source}')

            audio = audio_io.load_audio(example_source, sample_rate_hz)
            # choose the middle 5s of the audio
            start = int(max(len(audio)//2 - sample_rate_hz * window_size_s // 2, 0))
            end = int(start + sample_rate_hz * window_size_s) # python handles index past the end
            audio = audio[start: end]
        
            embedding = self.embedding_model.embed(audio).embeddings[0, 0]
            self.add_query_to_db(dataset_name, embedding, example_source, label)
            print('done')
        

    def add_query_to_db(self, dataset_name, embedding, source, query_label):
      
        #TOOD: do we also need to add the config, so that it can construct a full path from config
        # base path and file id?
        source = str(source)
        source = hoplite_interface.EmbeddingSource(dataset_name=dataset_name, source_id=source, offsets=np.array(0.0))
        embedding_id = self.db.insert_embedding(embedding, source)
        label = hoplite_interface.Label(embedding_id, query_label, type = hoplite_interface.LabelType.POSITIVE, provenance=self.config.annotator_id)
        self.db.insert_label(label, skip_duplicates=True)
        self.db.commit()


    def search_with_query(self, num_results=50, sample_size=1_000_000, target_score=None, dataset='from_config'):
        if dataset == 'from_config':
          dataset = self.config.search_dataset_name
        self.search(
            query=self.query_embedding,
            num_results=num_results,
            sample_size=sample_size,
            target_score=target_score,
            dataset=dataset,
            exclude_labeled=True)
    

    def show_labeled_examples(self, label, label_type=1, num=5, dataset_name=None):
        """
        Show a few random spectrograms of labeled examples, largely for sanity-checking. 
        """
        
        label_type = [
          hoplite_interface.LabelType.NEGATIVE,
          hoplite_interface.LabelType.POSITIVE
        ][label_type]
        
        embedding_ids = self.db.get_embeddings_by_label(label, label_type)
      
        indices = sorted(random.sample(range(len(embedding_ids)), min(num, len(embedding_ids))))
        embedding_ids = [embedding_ids[i] for i in indices]

        display_group_members = []
        for eid in embedding_ids:
            source = self.db.get_embedding_source(eid)
            emb_disp = embedding_display.EmbeddingDisplay(eid, 
                                                          dataset_name, 
                                                          source.source_id, 
                                                          source.offsets[0], 0)
            display_group_members.append(emb_disp)
            edg = embedding_display.EmbeddingDisplayGroup.create(display_group_members, 
                                                                  sample_rate_hz=32000, 
                                                                  audio_loader=self.audio_filepath_loader)
            
            # TODO: in the old perch repo, i modified the embeddingDisplayGroup to have a baw_config
            # and use baw to load audio if the baw-config is set
            # this is not implemented in the new perch-hoplite. Need to find another way around this. 
            print("Warning: baw_config is not set in the display group - so this is probably going to fail")
            edg.baw_config = self.config.baw_config
            edg.display(positive_labels=[])


    def search(self, query, bias=0.0, num_results=50, sample_size=1_000_000, 
              target_score=None, dataset=None, exclude_labeled=True):
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
            sample_size=sample_size)

        self.search_results = results
        self.all_search_scores = all_scores

        # TODO(tomdenton): Better histogram when target sampling.
        _ = plt.hist(all_scores, bins=100)
        hit_scores = [r.sort_score for r in results.search_results]
        plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|',
                    color='r', alpha=0.5)
        plt.show()


    def display_search_results(self):
        """
        Displays the results in self.search_results (which come from either classifier or single query searching)
        """

        query_label = self.label['value']

        if not query_label:
            print('No label set. Please select a label before displaying search results.')
            return
        
        plt.figure() 
        self.display_group = embedding_display.EmbeddingDisplayGroup.from_search_results(
            self.search_results, self.db, sample_rate_hz=32000, frame_rate=100,
            audio_loader=self.audio_filepath_loader)
        
        # injecting this into the embedding_display_group causes it to load audio from baw
        self.display_group.baw_config = self.config.baw_config
        self.display_group.display(positive_labels=[query_label])


    def save_labels(self):
        prev_lbls, new_lbls = 0, 0
        if not self.display_group:
            print('No display group found. Please run display_search_results() first.')
            return
        for lbl in self.display_group.harvest_labels(self.config.annotator_id):
            check = self.db.insert_label(lbl, skip_duplicates=True)
            new_lbls += check
            prev_lbls += (1 - check)
        self.db.commit()
        print('\nnew_lbls: ', new_lbls)
        print('\nprev_lbls: ', prev_lbls)
        self.print_label_counts()
        return prev_lbls, new_lbls
    
    
    def train_classifier(self, 
                        target_labels=None,
                        learning_rate = 1e-3,
                        weak_neg_weight = 0.05,
                        num_steps = 128,
                        train_ratio = 0.9,
                        batch_size = 128,
                        weak_negatives_batch_size = 128):

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


    def save_search_results(self, path, append=False):
        """
        saves the search results to a csv file
        """
        rows = []
        for res in self.search_results.search_results:
            source = self.db.get_embedding_source(res.embedding_id)
            domain, arid = baw_utils.extract_arid_and_domain(source.source_id)
            offset = float(source.offsets[0])
            audio_url = baw_utils.make_baw_audio_url_from_arid(arid, offset, 5.0, domain)
            #source = self.db.get_source(embedding.source_id)
            row = {
              'embedding_id': res.embedding_id,
              'arid': arid,
              'offset': offset,
              'score': res.sort_score,
              'link': audio_url
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if append:
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, index=False)


    def print_label_counts(self, label: str=None):
        """Prints the number of positive and negative of each label in the db"""
        
        positive = self.db.get_class_counts(hoplite_interface.LabelType.POSITIVE)
        negative = self.db.get_class_counts(hoplite_interface.LabelType.NEGATIVE)
        if (len(positive.items()) > 0):
            print('Label counts:')
            for lbl, count in positive.items():
                if label is None or lbl == label:
                    print(f'{lbl}: {count} positive, {negative[lbl]} negative')
        else:
            print('No labeled embeddings in db.')


    def search_with_classifier(self,
          num_results=50, sample_size=1_000_000, target_score=None, dataset=None):
        """Prepares the query and bias from the classifier params and searches with these"""

        target_label= self.label['value']
        if not target_label:
            print('No label set. Please select a label before searching with classifier.')
            return

        target_labels = self.classifier.classes
        print('target_labels: ', target_labels)
        target_label_idx = target_labels.index(target_label)
        if target_label_idx < 0:
            print(f'Label {target_label} not found in classifier. Available labels: {target_labels}')
            return
        class_query = self.classifier.beta[:, target_label_idx]
        bias = self.classifier.beta_bias[target_label_idx]
        
        self.search(
            query=class_query,
            bias=bias,
            num_results=num_results,
            sample_size=sample_size,
            target_score=target_score, 
            dataset=dataset,
            exclude_labeled=True)
      
  
    def create_database(self, embeddings_files, overwrite=False):
        """
        Checks if a database already exists at the specified path, and if not, creates one from the files at the location
        specified. If it does allows the user to delete and then creates the database. 
        """

        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        def create_db():

            to_delete = [
              db_path,
              (db_path.parent / Path(db_path.name + "-shm")),
              (db_path.parent / Path(db_path.name + "-wal"))
            ]

            for f in to_delete:
                if f.exists():
                    f.unlink()
            
            if (Path(embeddings_files) / 'filelist.json').exists():
                parquet_filepaths = [Path(embeddings_files) / 'embeddings' / Path(str(f['site_id'])) / Path(str(f['id'])) / 'embeddings.parquet' 
                                     for f in json.load(open(Path(embeddings_files) / 'filelist.json'))]
            else:
                parquet_filepaths = None

            if self.config.max_embeddings_count:
                max_count = self.config.max_embeddings_count
            else:
                max_count = -1

            if not self.config.search_dataset_name:
                print('search_dataset_name not set in config. Please set it before creating the database.')
                return

            print(f'creating db at {db_path.resolve()}')
            db = convert_parquet(parquet_folder = Path(embeddings_files),
                                parquet_filepaths = parquet_filepaths, 
                                db_type = "sqlite_usearch", 
                                dataset_name = self.config.search_dataset_name, 
                                max_count=max_count, 
                                db_path=db_path)
            print(f'db created at {db_path.resolve()} with {db.count_embeddings()} embeddings.')

        if db_path.exists() and not overwrite:

            print(f"DB path already exists at {db_path.resolve()}. (Initialize the DB to see details)")

            button = widgets.Button(description="Delete Existing Database and create it again?",
                                    layout=widgets.Layout(width='auto'))
            def on_button_click(b):
                create_db()
            button.on_click(on_button_click)
            display(button)

        else:
            create_db()


    def db_summary(self):
        """Debugging/sanity checking to see what's in the database"""
        if not self.db:
            print('DB not initialized.')
            return
        print('DB Summary:')
        print(f'Number of embeddings: {self.db.count_embeddings()}')
        datasets = self.db.get_dataset_names()
        print(f'{len(datasets)} Dataset(s):')
        for dataset in datasets:
            # This is slow, so commented out for now
            #print(f'{dataset}: {len(self.db.get_embeddings_by_source(dataset_name=dataset, source_id=None))} embeddings')
            print(f'{dataset}')
        self.print_label_counts()


    def save_classifier(self, path=None):
        if path is None:
            path = Path(self.config.models_folder) / 'classifier.json'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save(path)
      

    def run_inference(self, output_filepath=None, threshold=0.0, labels=None, dataset=None, subset=1):
        """
        Calls the classifier.write_inference_csv() function to write the inference results to a csv file
        injecting the site_id and link columns via a row_func that uses baw_utils and 
        """

        if subset is None:
            subset = 1.0

        if subset <= 0 or subset > 1:
          raise ValueError('Subset must be between 0 and 1')
        
        # I think i originally edited the db implementation to add the dataaset. 
        # TODO: copy this function to our own package and implement it there
        #embedding_ids = self.db.get_embedding_ids(dataset=dataset)
        embedding_ids = sqlite_usearch_impl_mod.get_embedding_ids(self.db, dataset=dataset)

        if subset < 1:
            n_samples = int(subset * len(embedding_ids))
            indices = np.arange(len(embedding_ids))
            np.random.shuffle(indices)
            selected_indices = sorted(indices[:n_samples])
            embedding_ids = embedding_ids[selected_indices]

        if output_filepath is None:
            output_filepath = Path(self.config.predictions_folder) / 'inference.csv'
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)

        site_lookup = self.make_site_map()

        def row_func(row=None):
            if row is None:
                return ['link', 'site_id', 'site_name']
            baw_domain, arid  = baw_utils.extract_arid_and_domain(row[2])
            link = baw_utils.make_baw_listen_link_from_arid(arid, row[3], 5.0, baw_domain)
            site_info = site_lookup.get(int(arid), '')
            return [link, site_info['site_id'], site_info['site_name']]
            
        classifier_mod.write_inference_csv(self.classifier,
                                       self.db,
                                       output_filepath,
                                       threshold,
                                       labels,
                                       embedding_ids,
                                       row_func=row_func)
    
    def make_site_map(self):
        """
        Creates a mapping source_id from the hoplite db to sites on ecoeounds
        """

        all_sources = sqlite_usearch_impl_mod.get_all_sources(self.db)
        # initialize empty site map
        site_map = {baw_utils.extract_arid_and_domain(src.source_id)[1]: {
            'source': src.source_id,
            'site_id': None,
            'site_name': None,
        } for src in all_sources}

        arids = [arid for arid in site_map.keys()]

        response = self.baw_api.post(
            '/audio_recordings/filter/?disable_paging=true',
            data={
                'filter': {
                    'id': {
                        'in': arids
                    }
                },
                'projection': { "include":['id', 'site_id', 'sites.name'] }
            }
        )

        recording_records = response.get('data', [])
        recording_records = {r['id']: r for r in recording_records}
        for arid, src in site_map.items():
            if arid in recording_records:
                rec = recording_records[arid]
                src['site_id'] = rec['site_id']
                src['site_name'] = rec['sites.name']
            else:
                print(f'No recording found for arid {arid} in the BAW API')

        return site_map

        

        
        



    def init_workbench(self, token=None):
        
        
        # this will prompt for a token if not set
        self.baw_api = baw_api(token=token, debug=True)

        if self.baw_api.token is not None:
            if self.config.baw_config.get('auth_token'):
                print("Overwriting config auth token with new value")
            self.config.baw_config['auth_token'] = self.baw_api.token
        elif self.config.baw_config.get('auth_token'):
            print("Auth token not loaded, but already in config")
        else:
            print("No auth token set")

    def show_label_selector(self, initial_value=None):
        """
        Displays a label selector widget that allows the user to select a label from the list of available labels.
        If a label is provided, it will be used to initialize the widget (if it is valid)"""

        agile_widgets.label_selector(
            baw_api=self.baw_api, binding=self.label, initial_value=initial_value
        )

