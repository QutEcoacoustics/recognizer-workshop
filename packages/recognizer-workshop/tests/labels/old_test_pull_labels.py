import pytest
from unittest.mock import MagicMock, call
import numpy as np
import pandas as pd
from recognizer_workshop.app.services.db_labels import add_db_labels
from perch_hoplite.db import interface as hoplite_interface
from recognizer_workshop.app.agile_config import AgileConfig

# --- Fixtures ---

@pytest.fixture
def mock_baw_api():
    """Mocks the BAW API client."""
    client = MagicMock()
    client.base_url = "https://api.ecosounds.org"
    return client

@pytest.fixture
def sample_label_items():
    """
    Creates the complex dictionary structure that 'add_labels_to_db' expects.
    Structure: { source_url: { label_name: { 'pos': [offsets], 'neg': [offsets] } } }
    """
    return {
        "https://api.ecosounds.org/audio_recordings/111/original": {
            "Magpie": {
                "pos": [10.0, 15.0],
                "neg": [30.0]
            }
        },
        "https://api.ecosounds.org/audio_recordings/222/original": {
            "Magpie": {
                "pos": [5.0],
                "neg": []
            }
        }
    }

@pytest.fixture
def mock_config():
    conf = MagicMock(spec=AgileConfig)
    conf.search_dataset_name = "test_dataset"
    conf.annotator_id = "test_user"
    return conf

# --- Unit Tests for Helper Logic ---

def test_get_event_segments_simple():
    """Test standard event coverage calculation."""
    event = {"start_time_seconds": 12.0, "end_time_seconds": 14.0} # 2s duration
    
    # Should fall into the 10-15s window
    # Overlap is 2s. Window is 5s.
    # Segment cover = 2/5 = 0.4
    # Event cover = 2/2 = 1.0
    # Score = 1.4
    
    segments = add_db_labels.get_event_segments(event, segment_duration=5)
    
    assert len(segments) == 1
    seg = segments[0]
    assert seg['start'] == 10
    assert seg['end'] == 15
    assert seg['cover_score'] == 1.4

def test_get_event_segments_spanning_multiple():
    """Test an event that spans across a segment boundary."""
    event = {"start_time_seconds": 4.0, "end_time_seconds": 11.0} # 7s duration
    # Segments: 0-5, 5-10, 10-15
    
    segments = add_db_labels.get_event_segments(event, segment_duration=5)
    
    # 0-5: overlap 1s (4-5). SegCov=0.2. EvCov=0.14. Score=0.34
    # 5-10: overlap 5s. SegCov=1.0. EvCov=0.71. Score=1.71
    # 10-15: overlap 1s (10-11). SegCov=0.2. EvCov=0.14. Score=0.34
    
    assert len(segments) == 3
    assert segments[0]['start'] == 0
    assert segments[1]['start'] == 5
    assert segments[2]['start'] == 10

def test_get_consensus():
    """Test verification consensus logic."""
    # Clear winner
    verifications = [{'confirmed': 'correct'}, {'confirmed': 'correct'}, {'confirmed': 'incorrect'}]
    assert add_db_labels.get_consensus(verifications) == 'positive'
    
    # Tie
    verifications = [{'confirmed': 'correct'}, {'confirmed': 'incorrect'}]
    assert add_db_labels.get_consensus(verifications) is None
    
    # Empty
    assert add_db_labels.get_consensus([]) is None


# --- Integration Tests for Adding Labels ---

def test_ensure_embeddings_exist_collects_offsets(mock_baw_api, sample_label_items):
    """
    Test that ensure_embeddings_exist correctly flattens the dictionary 
    and calls add_parquet_to_db with the correct structure.
    """
    # Create a dummy DB object that has metadata
    mock_db = MagicMock()
    mock_model_config = MagicMock()
    mock_model_config.model_config.window_size_s = 5.0
    mock_db.get_metadata.return_value = mock_model_config

    # Monkeypatch add_parquet_to_db so we don't actually download anything
    mock_add_parquet = MagicMock()
    
    # Apply monkeypatch to the module where it is IMPORTED (db_labels), not defined
    with pytest.MonkeyPatch.context() as m:
        m.setattr(add_db_labels, 'add_parquet_to_db', mock_add_parquet)
        
        add_db_labels.ensure_embeddings_exist(mock_db, MagicMock(), sample_label_items, mock_baw_api)

    # Assert it was called
    assert mock_add_parquet.called
    
    # Inspect arguments
    call_args = mock_add_parquet.call_args
    kwargs = call_args.kwargs
    parquet_source = kwargs['parquet_source']
    
    # We expect 2 items in the list (111 and 222)
    assert len(parquet_source) == 2
    
    # Check item 111
    item_111 = next(item for item in parquet_source if '111' in item['source_url'])
    # Should have offsets 10.0, 15.0, 30.0 (order agnostic)
    assert set(item_111['offsets']) == {10.0, 15.0, 30.0}

    # Check item 222
    item_222 = next(item for item in parquet_source if '222' in item['source_url'])
    assert set(item_222['offsets']) == {5.0}


def test_add_labels_to_db_happy_path(db_with_path, mock_config, mock_baw_api, sample_label_items):
    """
    Tests the full flow of adding labels:
    1. Checks/Fetches missing embeddings (Mocked)
    2. Inserts labels into DB (Real SQLite)
    3. Returns outcomes
    """
    db, _ = db_with_path
    
    # 1. Setup Pre-existing embeddings
    # We manually add embeddings for ARID 111 (offsets 10, 15) to simulate them existing
    # But we leave out offset 30, and leave out ARID 222 entirely.
    # This tests that the logic handles existing vs missing correctly.
    
    # Helper to insert a dummy embedding
    def insert_dummy(arid, offset):
        emb = np.zeros(1280, dtype=np.float32)
        src = f"https://api.ecosounds.org/audio_recordings/{arid}/original"
        # Note: insert_embedding expects specific source object
        source_obj = hoplite_interface.EmbeddingSource(
            dataset_name=mock_config.search_dataset_name, 
            source_id=src, 
            offsets=np.array([float(offset)])
        )
        db.insert_embedding(emb, source_obj)

    insert_dummy(111, 10.0)
    insert_dummy(111, 15.0)
    # 111/30.0 is MISSING
    # 222/5.0 is MISSING
    db.commit()

    # 2. Mock the ensure_embeddings_exist logic
    # We want to simulate that ensure_embeddings_exist SUCCESSFULLY downloaded the missing ones.
    # So our side_effect needs to actually insert the missing embeddings into the DB 
    # so that the next step of add_labels_to_db can find them.
    def mock_downloader_side_effect(**kwargs):
        # Insert the missing ones now
        insert_dummy(111, 30.0)
        insert_dummy(222, 5.0)
        db.commit()

    with pytest.MonkeyPatch.context() as m:
        # Mock the underlying parquet adder to simulate a successful download
        m.setattr(add_db_labels, 'add_parquet_to_db', mock_downloader_side_effect)

        # ACT
        results = add_db_labels.add_labels_to_db(db, mock_config, sample_label_items, mock_baw_api)

    # ASSERTIONS

    # 1. Check Outcomes format
    # 111 / Magpie / pos
    outcomes_111_pos = results["https://api.ecosounds.org/audio_recordings/111/original"]["Magpie"]["pos"]
    assert len(outcomes_111_pos) == 2
    assert outcomes_111_pos[0]['outcome'] == 'added' 
    assert outcomes_111_pos[1]['outcome'] == 'added'

    # 111 / Magpie / neg (This was the one missing initially, but added by our mock)
    outcomes_111_neg = results["https://api.ecosounds.org/audio_recordings/111/original"]["Magpie"]["neg"]
    assert len(outcomes_111_neg) == 1
    assert outcomes_111_neg[0]['outcome'] == 'added'

    # 2. Check Database Content
    # Get labels for 111 at offset 30 (Negative)
    # We need to find the embedding ID first
    emb_ids = db.get_embeddings_by_source(mock_config.search_dataset_name, "https://api.ecosounds.org/audio_recordings/111/original")
    
    # This is a bit manual, but we iterate to find the one with offset 30
    target_eid = None
    for eid in emb_ids:
        src = db.get_embedding_source(eid)
        if src.offsets[0] == 30.0:
            target_eid = eid
            break
    
    assert target_eid is not None
    labels = db.get_labels(target_eid)
    assert len(labels) == 1
    assert labels[0].label == "Magpie"
    assert labels[0].type == hoplite_interface.LabelType.NEGATIVE


def test_add_labels_handles_missing_embeddings_gracefully(db_with_path, mock_config, mock_baw_api, sample_label_items):
    """
    Test that if ensure_embeddings_exist fails (or returns 404), 
    the label adder reports 'embedding_not_found'.
    """
    db, _ = db_with_path
    
    # We do NOT insert any embeddings.
    # We Mock the downloader to do NOTHING (simulating a 404 or failure to download).
    mock_downloader = MagicMock()
    
    with pytest.MonkeyPatch.context() as m:
        m.setattr(add_db_labels, 'add_parquet_to_db', mock_downloader)

        results = add_db_labels.add_labels_to_db(db, mock_config, sample_label_items, mock_baw_api)

    # Check that outcomes are 'embedding_not_found'
    outcomes = results["https://api.ecosounds.org/audio_recordings/111/original"]["Magpie"]["pos"]
    assert outcomes[0]['outcome'] == 'embedding_not_found'


# --- Data Shaping Logic ---

def test_labels_from_event_taggings_shaping(mock_baw_api):
    """
    Tests that the function correctly processes events into the required dictionary format.
    We mock the heavy 'get_events_with_verifications_and_tags' function to avoid all the API calls.
    """
    
    # 1. Setup Mock Events
    # Event 1: Positive Magpie
    e1 = {
        'id': 1, 'audio_recording_id': 100, 
        'start_time_seconds': 10.0, 'end_time_seconds': 15.0,
        'taggings': [{'tag': {'text': 'Magpie'}, 'type': 'positive'}] # Verified positive
    }
    # Event 2: Negative Crow (Target label Magpie, so this is a distractor/untagged for Magpie)
    # But wait, logic says: "if not label matches tag... set to negative"
    e2 = {
        'id': 2, 'audio_recording_id': 100, 
        'start_time_seconds': 20.0, 'end_time_seconds': 25.0,
        'taggings': [{'tag': {'text': 'Crow'}, 'type': 'positive'}] 
    }
    
    events = [e1, e2]
    
    class MockLabel:
        def __init__(self, name): self.name = name
        def __str__(self): return self.name
        def get_baw_tags(self): return [self.name]

    target_labels = [MockLabel("Magpie")]
    
    # 2. Mock the helper function that gets the events
    with pytest.MonkeyPatch.context() as m:
        m.setattr(add_db_labels, 'get_events_with_verifications_and_tags', MagicMock(return_value=events))
        
        # ACT
        items, counts = add_db_labels.labels_from_event_taggings(mock_baw_api, {}, target_labels)

    # ASSERT
    source_url = "https://api.ecosounds.org/audio_recordings/100/original"
    assert source_url in items
    
    # Check Event 1 (10.0s -> Segment 10.0) -> Pos
    assert 10.0 in items[source_url]["Magpie"]["pos"]
    
    # Check Event 2 (20.0s -> Segment 20.0)
    # The Crow tag does NOT match Magpie.
    # Logic: "elif not cur_label_matches_tag and tagging['type'] == 'positive': items... = 'neg'"
    # So this should be a negative label for Magpie.
    assert 20.0 in items[source_url]["Magpie"]["neg"]

    assert counts["Magpie"]["positive"] == 1
    assert counts["Magpie"]["negative"] == 1