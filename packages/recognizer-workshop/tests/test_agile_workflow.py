
from pathlib import Path

from recognizer_workshop.app.agile_engine import agile_config, agile_state

from recognizer_workshop.app.agile_modeling_helpers import list_audio_files, labels_from_event_taggings, add_labels_to_db

def test_full_workflow (db_with_path, working_folder, audio_files_folder):

    db, path_to_db = db_with_path



    annotator_id = "phil"


    config = agile_config(
        db_path=path_to_db,
        annotator_id=annotator_id,
        # For this tutorial we arbitrarily choose to name our search dataset
        # "search_set", however if you manage multiple search datasets you will have
        # to assign each a unique name via this configuration.
        search_dataset_name="search_set",
        embeddings_folder=None,
        labeled_examples_folder=audio_files_folder,
        models_folder=working_folder / 'models',
        predictions_folder=working_folder / 'predictions',
    )

    agile = agile_state(config)

    assert agile is not None

    agile.init_workbench()

    agile.initialize()


    labels = labels_from_event_taggings(agile.baw_api, filter_params={"project_id": [1]}, target_labels=agile.get_target_labels())
    add_labels_to_db(agile, config, labels)

    audio_files = list_audio_files(config.labeled_examples_folder)