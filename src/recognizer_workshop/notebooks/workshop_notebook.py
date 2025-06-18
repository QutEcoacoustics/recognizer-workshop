# code from the notebook for debugging




# CELL ONE

import os
import dotenv
from pathlib import Path

from recognizer_workshop.agile_modeling_state import agile_config, agile_state
from recognizer_workshop.agile_modeling_helpers import list_audio_files, labels_from_event_taggings, add_labels_to_db

find_dotenv = dotenv.find_dotenv
load_dotenv = dotenv.load_dotenv

# shortcuts to the state variables
#agile_config = agile.agile_config
#agile_state = agile.agile_state


# CELL TWO (colab import - not needed)

# CELL TWO

annotator_id = "phil"
working_folder = Path('/Users/eichinsp/Documents/github/peichins/recognizer-workshop/local_scripts/test_folder')

config = agile_config(
    db_path=Path('/Users/eichinsp/Documents/github/peichins/recognizer-workshop/local_scripts/test_folder/db/85_bak'),
    annotator_id=annotator_id,
    # For this tutorial we arbitrarily choose to name our search dataset
    # "search_set", however if you manage multiple search datasets you will have
    # to assign each a unique name via this configuration.
    search_dataset_name="search_set",
    embeddings_folder=None,
    labeled_examples_folder=working_folder / 'labeled_examples',
    models_folder=working_folder / 'models',
    predictions_folder=working_folder / 'predictions',
)

Path(config.labeled_examples_folder).mkdir(exist_ok=True, parents=True)

config.baw_config['domain'] = 'api.ecosounds.org'

agile = agile_state(config)

#print(agile)


# CELL THREE

agile.init_workbench()

# CELL FOUR (download embeddings - not needed)




# CELL FIVE (create database - not needed)

# CELL SIX (initialize agile)
agile.initialize()

# query_label = 'Yellow-bellied Glider' 
# agile.show_label_selector(query_label)

# BONUS CELL - pull all labels from the website and insert into the database
# from recognizer_workshop.agile_modeling_helpers import labels_from_event_taggings
# from recognizer_workshop.baw_api import baw_api

#from recognizer_workshop.agile_modeling_helpers import fix_source_format
#fix_source_format(agile.db)


# labels = labels_from_event_taggings(agile.baw_api, filter_params={"region_id": [116]})
# add_labels_to_db(agile, config, labels)


# CELL SEVEN 

audio_files = list_audio_files(config.labeled_examples_folder)


# TODO: display labelled embeddings from the database to use as search examples

# query = '0'
# agile.display_query(query)


# CELL EIGHT (search with query)

# num_results = 10
# target_score = None
# agile.embed_query()
# agile.search_with_query(
#     num_results=num_results,
#     # When working with really large datasets it may be necessary for
#     # performance reasons to look at a smaller (random) subset of the database
#     # entries to perform the search. Replacing None with an integer argument
#     # would limit the search to a random subset of that size.
#     sample_size=None,
#     target_score=target_score,
# )


# CELL NINE (display results)
# to be replaced with verification page on ecosounds 

#agile.show_label_selector()
#agile.display_search_results()

# CELL TEN (save labels to the database)

#agile.save_labels()


# TODO: update labels in the db from the website

# CELL ELEVEN train classifier

target_labels = "Yellow-bellied glider"  #@param {type:'string'}
learning_rate = 1e-3
num_steps = 101
batch_size = 32  #@param
train_ratio = 0.9  #@param
weak_neg_weight = 0.05
weak_negatives_batch_size = 16
agile.train_classifier(target_labels, learning_rate, weak_neg_weight, num_steps, train_ratio, batch_size, weak_negatives_batch_size)



# CELL TWELVE search with classifier

# num_results = 10  #@param
# target_score = None  #@param

# agile.search_with_classifier(
#     num_results=num_results,
#     # When working with really large datasets it may be necessary for
#     # performance reasons to look at a smaller (random) subset of the database
#     # entries to perform the search. Replacing None with an integer argument
#     # would limit the search to a random subset of that size.
#     sample_size=None,
#     target_score=target_score,
# )

# CELL THIRTEEN display classifier results

#agile.display_search_results()

# CELL FOURTEEN save classifier labels
#agile.save_labels()


# CEL FIFTEEN save classifier model

classifier_name = 'ybg_01'  #@param {type:'string'}

classifier_path = config.models_folder / f'{classifier_name}.json'

classifier_path.parent.mkdir(exist_ok=True, parents=True)

#agile.classifier.save(classifier_path)


output_filename = 'recognizer_output_01.csv'  #@param {type:'string'}
#@markdown The `threshold` determines what constitutes a detection. The classifier
#@markdown outputs a score for every embedding in the search set; anything above
#@markdown the threshold is considered a detection. A higher threshold value
#@markdown results in fewer but more confident detections, at the cost of missing
#@markdown out on relevant events in the search set. Conversely, a lower threshold
#@markdown value results in more but less confident detections, at the cost of
#@markdown producing more false positive detections.
threshold = 1 # @param {"type":"number"}
#@markdown You can also specify a random subset of the dataset to run inference
#@markdown on, e.g. 0.5 for 50%.
subset = 0.0003 #@param {type:'string'}

# Which labels to include in the output file. If None, all labels in your database are included.
labels = None

output_filepath = config.predictions_folder / output_filename
output_filepath.parent.mkdir(parents=True, exist_ok=True)
agile.run_inference(
    output_filepath,
    threshold=threshold,
    dataset=config.search_dataset_name,
    subset=subset,
)




