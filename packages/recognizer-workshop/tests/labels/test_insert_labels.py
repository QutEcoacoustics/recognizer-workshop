import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

# Adjust import to match your structure
from recognizer_workshop.app.services.db_labels import add_db_labels as target_module
from perch_hoplite.db import interface as hoplite_interface

# --- Local Mocks & Fixtures ---

@pytest.fixture
def mock_db():
    """
    Local mock for the database interface. 
    We use this instead of the real 'db_with_path' fixture when we want 
    to force specific edge cases (like DB errors or specific existing labels) 
    without setting up a complex SQLite state.
    """
    db = MagicMock()
    return db

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.search_dataset_name = "test_dataset"
    config.annotator_id = "test_user"
    return config

# --- Unit Tests ---

def test_build_parquet_source_list():
    """Test the pure logic function that converts the items dict to a list."""
    items = {
        "some_source": {"Magpie": {"pos": [10.0, 20.0]}},
        "some_other_source": {"Crow": {"neg": [5.0]}}
    }
    
    result = target_module.build_parquet_source_list(items)
    
    assert len(result) == 2
    # Verify contents (order doesn't matter, so we look for matching url)
    s1 = next(x for x in result if x['source_url'] == "some_source")
    assert set(s1['offsets']) == {10.0, 20.0}
    
    s2 = next(x for x in result if x['source_url'] == "some_other_source")
    assert set(s2['offsets']) == {5.0}


def test_ensure_embeddings_exist(mock_db, mock_config, mock_api_client):
    """
    Tests that the stuff that downloads embedddings based on the source and offsets works
    Almost everything is mocked, but we just check that the right functions are called with the right arguments.
    """
    items = {"s1.wav": {"Magpie": {"pos": [10.0]}}}

    # Patch the external function calls
    with patch.object(target_module, 'build_parquet_source_list', return_value=[{'fake': 'data'}]) as mock_build, \
         patch.object(target_module, 'add_parquet_to_db') as mock_add_parquet:
        
        target_module.ensure_embeddings_exist(mock_db, mock_config, items, mock_api_client)
        
        mock_build.assert_called_once_with(items)
        mock_add_parquet.assert_called_once()
        
        # Verify arguments passed to the heavy lifter
        kwargs = mock_add_parquet.call_args.kwargs
        assert kwargs['db'] == mock_db
        assert kwargs['api_client'] == mock_api_client
        assert kwargs['dataset_name'] == 'fetched_labels'
        assert kwargs['filter_dict'] is None


def test_ensure_embeddings_exist_passes_filter_dict(mock_db, mock_config, mock_api_client):
    """filter_dict passed to ensure_embeddings_exist must be forwarded to add_parquet_to_db."""
    items = {"s1.wav": {"Magpie": {"pos": [10.0]}}}
    my_filter = {'site_id': 42}

    with patch.object(target_module, 'build_parquet_source_list', return_value=[{'fake': 'data'}]),\
         patch.object(target_module, 'add_parquet_to_db') as mock_add_parquet:

        target_module.ensure_embeddings_exist(mock_db, mock_config, items, mock_api_client, filter_dict=my_filter)

        kwargs = mock_add_parquet.call_args.kwargs
        assert kwargs['filter_dict'] == my_filter


def test_add_labels_to_db_finds_embedding_in_search_set(mock_db, mock_config):
    """
    When the embedding is in the search_set dataset, it should be found and the label added.
    """
    source_url = "http://test.wav"
    mock_config.search_dataset_name = "search_set"

    def get_embeddings_side_effect(dataset_name, source_id):
        if dataset_name == 'search_set':
            return np.array([101])
        return np.array([])  # not in fetched_labels

    mock_db.get_embeddings_by_source.side_effect = get_embeddings_side_effect

    mock_src = MagicMock()
    mock_src.offsets = [10.0]
    mock_db.get_embedding_source.return_value = mock_src
    mock_db.get_labels.return_value = []

    items = {source_url: {"Magpie": {"pos": [10.0]}}}
    with patch.object(target_module, 'ensure_embeddings_exist'):
        results = target_module.add_labels_to_db(mock_db, mock_config, items)

    outcome = results[source_url]["Magpie"]["pos"][0]
    assert outcome['outcome'] == 'added'
    args, _ = mock_db.insert_label.call_args
    assert args[0].embedding_id == 101


def test_add_labels_to_db_finds_embedding_in_fetched_labels(mock_db, mock_config):
    """
    When the embedding is in the fetched_labels dataset (not search_set), it should still be found.
    """
    source_url = "http://test.wav"
    mock_config.search_dataset_name = "search_set"

    def get_embeddings_side_effect(dataset_name, source_id):
        if dataset_name == 'fetched_labels':
            return np.array([202])
        return np.array([])  # not in search_set

    mock_db.get_embeddings_by_source.side_effect = get_embeddings_side_effect

    mock_src = MagicMock()
    mock_src.offsets = [10.0]
    mock_db.get_embedding_source.return_value = mock_src
    mock_db.get_labels.return_value = []

    items = {source_url: {"Magpie": {"pos": [10.0]}}}
    with patch.object(target_module, 'ensure_embeddings_exist'):
        results = target_module.add_labels_to_db(mock_db, mock_config, items)

    outcome = results[source_url]["Magpie"]["pos"][0]
    assert outcome['outcome'] == 'added'
    args, _ = mock_db.insert_label.call_args
    assert args[0].embedding_id == 202


def test_add_labels_to_db_embedding_not_found_in_either_dataset(mock_db, mock_config):
    """
    When the embedding is absent from both datasets, outcome is 'embedding_not_found'.
    """
    source_url = "http://test.wav"
    mock_config.search_dataset_name = "search_set"
    mock_db.get_embeddings_by_source.return_value = np.array([])  # empty for all datasets

    items = {source_url: {"Magpie": {"pos": [10.0]}}}
    with patch.object(target_module, 'ensure_embeddings_exist'):
        results = target_module.add_labels_to_db(mock_db, mock_config, items)

    outcome = results[source_url]["Magpie"]["pos"][0]
    assert outcome['outcome'] == 'embedding_not_found'
    mock_db.insert_label.assert_not_called()


def test_add_labels_to_db_search_set_takes_precedence_over_fetched_labels(mock_db, mock_config):
    """
    When the same offset appears in both datasets, the search_set embedding_id is used.
    """
    source_url = "http://test.wav"
    mock_config.search_dataset_name = "search_set"

    def get_embeddings_side_effect(dataset_name, source_id):
        if dataset_name == 'search_set':
            return np.array([101])  # search_set embedding
        return np.array([202])      # fetched_labels duplicate

    def get_source_side_effect(eid):
        mock_src = MagicMock()
        mock_src.offsets = [10.0]  # same offset in both
        return mock_src

    mock_db.get_embeddings_by_source.side_effect = get_embeddings_side_effect
    mock_db.get_embedding_source.side_effect = get_source_side_effect
    mock_db.get_labels.return_value = []

    items = {source_url: {"Magpie": {"pos": [10.0]}}}
    with patch.object(target_module, 'ensure_embeddings_exist'):
        results = target_module.add_labels_to_db(mock_db, mock_config, items)

    outcome = results[source_url]["Magpie"]["pos"][0]
    assert outcome['outcome'] == 'added'
    # search_set id (101) must win over fetched_labels id (202)
    args, _ = mock_db.insert_label.call_args
    assert args[0].embedding_id == 101


def test_add_labels_to_db_logic(mock_db, mock_config):
    """
    The Core Logic Test.
    We mock the DB state to simulate:
    1. An embedding exists at offset 10.0
    2. An embedding exists at offset 20.0 (but already has the label)
    3. An embedding exists at offset 30.0 (but has the wrong label type)
    4. No embedding exists at offset 99.9
    """
    source_url = "http://test.wav"
    
    # --- 1. Mock DB State ---
    
    # Mock Embeddings for the source
    # We pretend DB has IDs 100, 200, 300 for this source
    mock_db.get_embeddings_by_source.return_value = np.array([100, 200, 300])

    # Mock Offset Lookup
    # This maps the ID back to the offset so the code can match your input list
    def get_source_side_effect(eid):
        # Return a fake object with .offsets attribute
        mock_src = MagicMock()
        if eid == 100: mock_src.offsets = [10.0]
        if eid == 200: mock_src.offsets = [20.0]
        if eid == 300: mock_src.offsets = [30.0]
        return mock_src
    mock_db.get_embedding_source.side_effect = get_source_side_effect

    # C. Mock Existing Labels
    # ID 200 already has "Magpie" (Positive) -> Should SKIP
    # ID 300 has "Magpie" (Negative) -> Should REPLACE (because we are trying to add Positive)
    def get_labels_side_effect(eid):
        if eid == 200:
            return (hoplite_interface.Label(200, "Magpie", hoplite_interface.LabelType.POSITIVE, "test_user"),)
        if eid == 300:
            return (hoplite_interface.Label(300, "Magpie", hoplite_interface.LabelType.NEGATIVE, "test_user"),)
        return []
    mock_db.get_labels.side_effect = get_labels_side_effect

    # --- 2. Setup Input Items ---
    items = {
        source_url: {
            "Magpie": {
                "pos": [
                    10.0, # Case 1: Standard Add
                    20.0, # Case 2: Duplicate
                    30.0, # Case 3: Type Conflict
                    99.9  # Case 4: Missing Embedding
                ],
                "neg": []
            }
        }
    }

    # --- 3. Run ---
    # We mock ensure_embeddings_exist so it doesn't try to call add_parquet_to_db
    with patch.object(target_module, 'ensure_embeddings_exist'):
        results = target_module.add_labels_to_db(mock_db, mock_config, items)

    # --- 4. Assertions ---
    
    outcomes = results[source_url]["Magpie"]["pos"]
    def check_outcome(offset, expected_status):
        found = next((x for x in outcomes if x['offset'] == offset), None)
        assert found is not None, f"Offset {offset} missing from results"
        assert found['outcome'] == expected_status, f"Offset {offset}: Expected {expected_status}, got {found['outcome']}"

    # Verify outcomes
    check_outcome(10.0, "added")
    check_outcome(20.0, "already_exists")
    check_outcome(30.0, "replaced_type")
    check_outcome(99.9, "embedding_not_found")

    # Verify DB interactions
    # 10.0 -> Insert Label
    args, _ = mock_db.insert_label.call_args_list[0]
    assert args[0].embedding_id == 100
    assert args[0].type == hoplite_interface.LabelType.POSITIVE

    # 30.0 -> Update Label Type
    mock_db.update_label_type.assert_called_with(300, "Magpie", hoplite_interface.LabelType.POSITIVE)
    
    # 20.0 -> Should NOT insert or update
    # We verify insert_label was ONLY called for ID 100 (and not 200)
    for call_args in mock_db.insert_label.call_args_list:
        label_obj = call_args[0][0]
        assert label_obj.embedding_id != 200, "Should not insert duplicate label for ID 200"


def test_add_labeled_audio_examples(mock_db, mock_embedding_model, audio_files_folder):
    """
    Integration-style test for adding raw audio examples.
    Uses:
      - audio_files_folder fixture (creates real wav files)
      - mock_embedding_model fixture (mocks ML output)
    """
    label_text = "Frog"
    dataset_name = "test_ds"

    # Patch 'list_audio_files' to return the files created by the fixture
    # (Or just trust the function if it works on Path objects)
    
    # Run
    target_module.add_labeled_audio_examples(
        db=mock_db,
        source_folder=audio_files_folder,
        label=label_text,
        embedding_model=mock_embedding_model,
        dataset_name=dataset_name
    )

    # Verify
    # 1. Check Embeddings Generated (The folder fixture creates 2 files)
    assert mock_embedding_model.embed.call_count == 2
    
    # 2. Check DB Insertions
    assert mock_db.insert_embedding.call_count == 2
    assert mock_db.insert_label.call_count == 2
    assert mock_db.commit.call_count == 2

    # 3. verify arguments of the last insert
    args, _ = mock_db.insert_label.call_args
    inserted_label = args[0]
    assert inserted_label.label == "Frog"
    assert inserted_label.type == hoplite_interface.LabelType.POSITIVE