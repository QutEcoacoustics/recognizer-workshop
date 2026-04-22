
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from recognizer_workshop.app.agile_engine import agile_state
from recognizer_workshop.app.agile_config import AgileConfig

from recognizer_workshop.app.agile_modeling_helpers import list_audio_files

from recognizer_workshop.app.services.db_labels.add_db_labels import add_labels_to_db 
from recognizer_workshop.app.services.db_labels.fetch_baw_labels import labels_from_event_taggings


@pytest.fixture
def minimal_config(tmp_path):
    """An AgileConfig pointing at a non-existent db_path — enough to construct agile_state."""
    return AgileConfig(db_path=tmp_path / "db")


def test_engine_is_db_loaded_false_before_init_db(minimal_config):
    """
    agile_state.is_db_loaded must be False immediately after construction,
    before init_db() is called.

    This guards against the bug where vm.obs.bump('engine_state') fired
    before vm.eng was None-ed out, causing UI callbacks to see a truthy
    engine with db=None and crash on db access.
    """
    eng = agile_state(minimal_config)
    assert eng.is_db_loaded is False


def test_engine_db_is_none_before_init_db(minimal_config):
    """
    Verifies the underlying db attribute is None before init_db(),
    consistent with is_db_loaded returning False.
    """
    eng = agile_state(minimal_config)
    assert eng.db is None


def test_engine_get_status_db_loaded_false_before_init(minimal_config):
    """
    get_status() should report db_loaded=False before init_db() is called,
    and must not raise even though db is None.
    """
    eng = agile_state(minimal_config)
    status = eng.get_status()
    assert status["db_loaded"] is False


def test_engine_is_db_loaded_reflects_db_assignment(minimal_config):
    """
    If db is set directly (simulating a successful init_db), is_db_loaded
    should return True. This keeps the property honest when the underlying
    attribute changes.
    """
    eng = agile_state(minimal_config)
    assert eng.is_db_loaded is False
    eng.db = MagicMock()  # simulate a successfully loaded db
    assert eng.is_db_loaded is True


@pytest.mark.skip(reason="WIP: Full workflow test not complete")
def test_full_workflow (db_with_path, working_folder, audio_files_folder):

    db, path_to_db = db_with_path

    annotator_id = "phil"

    config = AgileConfig(
        db_path=path_to_db,
        annotator_id=annotator_id,

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