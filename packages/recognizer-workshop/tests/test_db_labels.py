from unittest import result
import pytest
import numpy as np

from pathlib import Path





from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db import interface as hoplite_interface

# Functions to test
from recognizer_workshop.app.services.db_labels import db_labels

from recognizer_workshop.app.services.db_labels.dataclasses import LabelCounts

# This is a constant we will use across our tests
TEST_ANNOTATOR_ID = "test_suite_annotator"

# --- Fixtures: Reusable setup code for our tests ---



# --- Tests ---

def test_add_embedding_to_db(db_with_path):
    """Tests adding a single embedding directly to the database."""
    # Arrange: Get the initial state of the DB
    db, _ = db_with_path
    initial_count = db.count_embeddings()
    embedding = np.random.rand(1280).astype(np.float32)
    source_path = "/fake/path/audio.wav"
    label = "TestLabel"

    # Act: Call the function we are testing
    db_labels.add_embedding_to_db(
        db=db,
        dataset_name="test_dataset",
        embedding=embedding,
        source=source_path,
        label=label,
        annotator_id=TEST_ANNOTATOR_ID
    )

    # Assert: Check that the DB state has changed correctly
    assert db.count_embeddings() == initial_count + 1
    
    # Verify the label was added
    labels = db.get_embeddings_by_label(label=label, label_type=hoplite_interface.LabelType.POSITIVE)
    assert len(labels) > 0



def test_add_labeled_audio_examples(db_with_path, audio_files_folder, mock_embedding_model):
    """
    Tests the main function that processes a folder of audio files.
    """
    db_connection, _ = db_with_path
    # Arrange
    initial_count = db_connection.count_embeddings()
    label_to_add = "Eurasian Spangled Drongo"
    
    # # We need to patch the original function to pass the annotator_id
    # def patched_add_embedding_to_db(db, dataset_name, embedding, source, query_label, annotator_id):
    #     """A patched version for the test."""
    #     db_labels.add_embedding_to_db(db, dataset_name, embedding, source, query_label, annotator_id=annotator_id)

    # # Monkeypatch the original module to use our test-friendly version
    # # and also to find audio files in our temp directory
    # pytest.MonkeyPatch().setattr(db_labels, 'add_embedding_to_db', patched_add_embedding_to_db)
    pytest.MonkeyPatch().setattr(db_labels, 'list_audio_files', lambda path: list(path.glob("*.wav")))
    
    # Act
    db_labels.add_labeled_audio_examples(
        db=db_connection,
        source_folder=audio_files_folder,
        label=label_to_add,
        embedding_model=mock_embedding_model,
        dataset_name="test_audio_files",
        annotator_id=TEST_ANNOTATOR_ID
    )
    
    # Assert
    # We created 2 audio files, so 2 new embeddings should be in the DB
    assert db_connection.count_embeddings() == initial_count + 2
    mock_embedding_model.embed.assert_called()
    assert mock_embedding_model.embed.call_count == 2
    
    # Check that the labels were correctly added
    labels = db_connection.get_embeddings_by_label(label=label_to_add, label_type=hoplite_interface.LabelType.POSITIVE)
    assert len(labels) > 0


def test_get_label_counts(db_with_path):
    """
    Tests that the label count summary is generated correctly, and that the LabelCounts dataclass works.
    """
    db_connection, _ = db_with_path
    # Arrange: Add some data to the database first
    embedding = np.random.rand(1280).astype(np.float32)
    db_labels.add_embedding_to_db(db_connection, "d1", embedding, "s1", "BirdA", TEST_ANNOTATOR_ID)
    db_labels.add_embedding_to_db(db_connection, "d1", embedding, "s2", "BirdA", TEST_ANNOTATOR_ID)
    
    # Act: Run the function that prints to the console
    result = db_labels.get_label_counts(db_connection)
    
    assert len(result) == 1
    assert "BirdA" in result
    assert isinstance(result["BirdA"], LabelCounts)
    assert result["BirdA"].positive == 2
    assert result["BirdA"].negative == 0