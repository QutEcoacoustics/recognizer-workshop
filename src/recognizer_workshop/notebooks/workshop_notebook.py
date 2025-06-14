# code from the notebook for debugging




# CELL ONE
from recognizer_workshop.agile_modeling_state import agile_config, agile_state

from recognizer_workshop.agile_modeling_helpers import list_audio_files, labels_from_event_taggings, add_labels_to_db

import ast
import os
from pathlib import Path
import sys

import dotenv

find_dotenv = dotenv.find_dotenv
load_dotenv = dotenv.load_dotenv


# shortcuts to the state variables
#agile_config = agile.agile_config
#agile_state = agile.agile_state
import dotenv

# CELL TWO (colab import - not needed)

# CELL TWO

annotator_id = "phil"
working_folder = Path('/Users/eichinsp/Documents/github/peichins/recognizer-workshop/local_scripts/test_folder')

config = agile_config(
    db_path=working_folder / 'db',
    annotator_id=annotator_id,
    # For this tutorial we arbitrarily choose to name our search dataset
    # "search_set", however if you manage multiple search datasets you will have
    # to assign each a unique name via this configuration.
    search_dataset_name="search_set",
    embeddings_folder=working_folder / 'embeddings',
    labeled_examples_folder=working_folder / 'labeled_examples',
    models_folder=working_folder / 'models',
    predictions_folder=working_folder / 'predictions',
)

Path(config.labeled_examples_folder).mkdir(exist_ok=True, parents=True)
Path(config.embeddings_folder).mkdir(exist_ok=True, parents=True)

config.baw_config['domain'] = 'api.ecosounds.org'

agile = agile_state(config)

#print(agile)


# CELL THREE

env_file = find_dotenv()
if not env_file:
    print(f"No .env file found in the working directory {os.getcwd()}. \nFollow the local Jupyter instructions above to create one.")
else:
    load_dotenv(override=True)
    auth_token = os.getenv('BAW_AUTH_TOKEN')
    if auth_token:
        print(f"Got auth token from .env file {env_file}")
    else:
        print("BAW_AUTH_TOKEN env variable not found in your .env file. Follow the local Jupyter instructions above to set it")
        import getpass
        auth_token = getpass.getpass("Enter your BAW_AUTH_TOKEN: ")
        print("Token entered.")


if auth_token:
    if config.baw_config.get('auth_token'):
        print("Overwriting config auth token with new value")
    config.baw_config['auth_token'] = auth_token
elif config.baw_config.get('auth_token'):
    print("Auth token not loaded, but already in config")
else:
    print("No auth token set")

# CELL FOUR (download embeddings - not needed)


dataset_name = "yellow_bellied_glider"

# CELL FIVE (create database - not needed)

# CELL SIX (initialize agile)
agile.initialize()



# BONUS CELL - pull all labels from the website and insert into the database
from recognizer_workshop.agile_modeling_helpers import labels_from_event_taggings
from recognizer_workshop.baw_api import baw_api

api = baw_api()
labels = labels_from_event_taggings(api, filter_params={"region_id": [85]})
add_labels_to_db(agile, config, labels)


# CELL SEVEN 

audio_files = list_audio_files(config.labeled_examples_folder)


# TODO: display labelled embeddings from the database to use as search examples

query = '0'
agile.display_query(query)


# CELL EIGHT (search with query)

num_results = 10
target_score = None
agile.embed_query()
agile.search_with_query(
    num_results=num_results,
    # When working with really large datasets it may be necessary for
    # performance reasons to look at a smaller (random) subset of the database
    # entries to perform the search. Replacing None with an integer argument
    # would limit the search to a random subset of that size.
    sample_size=None,
    target_score=target_score,
)


# CELL NINE (display results)
# to be replaced with verification page on ecosounds 

query_label = 'ybg' 
agile.display_search_results(query_label)

# CELL TEN (save labels to the database)

agile.save_labels()


# TODO: update labels in the db from the website

# CELL ELEVEN train classifier

target_labels = None
learning_rate = 1e-3
num_steps = 101
batch_size = 32  #@param
train_ratio = 0.9  #@param
weak_neg_weight = 0.05
weak_negatives_batch_size = 16
agile.train_classifier(target_labels, learning_rate, weak_neg_weight, num_steps, train_ratio, batch_size, weak_negatives_batch_size)