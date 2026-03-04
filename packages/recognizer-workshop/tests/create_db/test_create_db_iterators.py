import pytest
from unittest.mock import ANY
from baw_helpers.baw_api import AuthenticationError

from perch_hoplite.db import sqlite_usearch_impl

# Adjust import to match your project structure
from recognizer_workshop.app.create_database.create_database import (
    iter_parquet_dataframes, 
)

def test_remote_iterator_happy_path(mock_api_client, create_fake_parquet_bytes):
    """
    Test standard successful download of multiple files.

    We don't need to mock the correct parquet structure. We just need to test that the iterator calls the 
    analysis_result_file method with correct parameters and yields dataframes.
    """
    
    # some fake parquet data for two files
    df1_data = [{'source': 'http://url/1', 'offset': 0.0, 'emb': 1.0}, {'source': 'http://url/1', 'offset': 5.0, 'emb': 2.0}]
    df2_data = [{'source': 'http://url/2', 'offset': 0.0, 'emb': 3.0}]
    
    # configure the mock api client to return these fake parquet bytes
    mock_api_client.analysis_result_file.side_effect = [
        create_fake_parquet_bytes(df1_data),
        create_fake_parquet_bytes(df2_data)
    ]

    # 2. Run Iterator
    sources = [{'id': 101}, {'id': 102}]
    iterator = iter_parquet_dataframes(sources, api_client=mock_api_client, prefetch=0)
    results = list(iterator)

    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 1
    
    assert mock_api_client.analysis_result_file.call_count == 2
    mock_api_client.analysis_result_file.assert_any_call(
        arid=101, analysis_job_id=ANY, analysis_name=ANY, to_memory=True
    )

def test_remote_iterator_filters_offsets(mock_api_client, create_fake_parquet_bytes):
    """Test that explicit 'offsets' in the source dict actually filter the dataframe."""
    
    # some fake parquet data for one file with multiple offsets
    full_data = [
        {'source': 's1', 'offset': 0.0},
        {'source': 's1', 'offset': 5.0},
        {'source': 's1', 'offset': 10.0}
    ]
    mock_api_client.analysis_result_file.return_value = create_fake_parquet_bytes(full_data)
    
    # only offset 5.0 is requested
    sources = [{'id': 101, 'offsets': [5.0]}]
    
    iterator = iter_parquet_dataframes(sources, api_client=mock_api_client)
    results = list(iterator)

    assert len(results) == 1
    df = results[0]
    assert len(df) == 1
    assert df.iloc[0]['offset'] == 5.0



def test_remote_iterator_skips_on_auth_error_until_max_errors(mock_api_client, create_fake_parquet_bytes):
    """Test that the iterator skips a failed item and continues to the next one."""
    
    # Fake parquet data for the successful calls 
    item_102 = [{'source': 's102', 'offset': 0.0}]
    item_104 = [{'source': 's104', 'offset': 5.0}]
    item_106 = [{'source': 's106', 'offset': 5.0}]
    item_109 = [{'source': 's109', 'offset': 5.0}]
    
    # 1. Setup Mock: First call fails, Second call succeeds
    mock_api_client.analysis_result_file.side_effect = [
        AuthenticationError("Token expired"),       # ID 101 fails
        create_fake_parquet_bytes(item_102),      # ID 102 succeeds
        AuthenticationError("Token expired"),       # ID 103 fails
        create_fake_parquet_bytes(item_104),      # ID 104 succeeds
        AuthenticationError("Token expired"),       # ID 105 fails
        create_fake_parquet_bytes(item_106),      # ID 106 succeeds
        AuthenticationError("Token expired"),       # ID 107 fails
        AuthenticationError("Token expired"),       # ID 108 fails
        create_fake_parquet_bytes(item_109),      # ID 109 succeeds    

    ]

    sources = [{'id': 101}, {'id': 102}, {'id': 103}, {'id': 104}, {'id': 105}, {'id': 106}, {'id': 107}, {'id': 108}, {'id': 109}]
    
    iterator = iter_parquet_dataframes(sources, api_client=mock_api_client, max_auth_errors=2, prefetch=1)
    results = []
    with pytest.raises(RuntimeError, match="Aborted after 2 consecutive"):
        for item in iterator:
            results.append(item)
            
    # Assertions
    
    # We got 3 successful results before the crash (102, 104, 106)
    assert len(results) == 3
    assert results[0].iloc[0]['source'] == 's102'
    assert results[2].iloc[0]['source'] == 's106'
    
    # We made exactly 9 calls. The last item is prefected before the exception was raised. 
    assert mock_api_client.analysis_result_file.call_count == 9


@pytest.mark.parametrize("max_auth_errors, prefetch", [
    (1, 0),  # Strict: Fail immediately, single threaded
    (2, 0),  # Retry once, single threaded
    (1, 1),  # Fail immediately, 1-ahead buffer
    (3, 2),  # Retry twice, 2-ahead buffer (larger pipeline)
])
def test_remote_iterator_consecutive_failures_matrix(
    mock_api_client, 
    create_fake_parquet_bytes, 
    max_auth_errors, 
    prefetch
):
    """
    Parametrized test to verify iterator behavior with different error limits and prefetch sizes.
    Scenario: 2 Successes -> M Failures -> Crash.
    This is testing the same as the test_remote_iterator_skips_on_auth_error_until_max_errors but with a matrix for completeness
    whereas test_remote_iterator_skips_on_auth_error_until_max_errors is more explicit for clarity
    """
    
    # We will have 2 successes, then 'max_auth_errors' consecutive failures.
    num_success_before = 2
    crash_index = num_success_before + max_auth_errors - 1
    
    # create enough source items to cover the scenario
    # we need 1 extra for the queue buffer, plus 1 for the zero indexing (we subtracted when calculating the crash_index)
    expected_calls = crash_index + prefetch + 2
    # we add a few extra items because we are not testing the end iteration logic
    sources = [{'id': i} for i in range(expected_calls + 4)]
    
    # Build responses from the mock API client
    # Successes
    side_effects = [create_fake_parquet_bytes([{'source': f's{i}'}]) for i in range(num_success_before)]
    # Failures (enough to crash)
    side_effects.extend([AuthenticationError(f"Fail {i}") for i in range(max_auth_errors)])
    # some requests are made after the crash point due to prefetching, so we need to add some more successful responses
    # the expected number is the prefetch size + 1, plus we add some for good measure. 
    num_items_after_auth_errors = prefetch + 3
    side_effects.extend([create_fake_parquet_bytes([{'source': 'buffer'}]) for _ in range(num_items_after_auth_errors)])
    
    mock_api_client.analysis_result_file.side_effect = side_effects

    # 3. Run Iterator
    iterator = iter_parquet_dataframes(
        sources, 
        api_client=mock_api_client, 
        max_auth_errors=max_auth_errors, 
        prefetch=prefetch
    )
    
    results = []
    with pytest.raises(RuntimeError, match=f"Aborted after {max_auth_errors} consecutive"):
        for item in iterator:
            results.append(item)

    assert len(results) == num_success_before
    assert mock_api_client.analysis_result_file.call_count == expected_calls



def test_normalization_integration(mock_api_client, create_fake_parquet_bytes):
    """Test that passing simple integers correctly triggers the remote iterator via normalization."""
    
    mock_api_client.analysis_result_file.return_value = create_fake_parquet_bytes([{'a': 1}])
    
    sources = [999] 
    iterator = iter_parquet_dataframes(sources, api_client=mock_api_client)
    results = list(iterator)
    
    assert len(results) == 1
    mock_api_client.analysis_result_file.assert_called_with(
        arid=999, analysis_job_id=ANY, analysis_name=ANY, to_memory=True
    )


def test_remote_iterator_deduplicates_against_real_db(
    mock_api_client, 
    create_fake_parquet_bytes, 
    test_db_path
):
    """
    Test that the iterator correctly interacts with a REAL database instance 
    to filter out existing offsets.
    """
    from recognizer_workshop.perch_hoplite_custom_modifications import sqlite_usearch_impl as sqlite_usearch_impl_custom

    db = sqlite_usearch_impl_custom.ThreadedSQLiteUsearchDB.create(test_db_path)


    
    # this actual test database has embeddings for this source with 500 embeddings, consecutive from 0 with hop 5
    # i.e. indices 0 to 499, offsets 0.0 to 2495.0
    test_arid = 4207561
    existing_db_source = f'https://api.ecosounds.org/audio_recordings/{test_arid}/original'
    existing_db_offsets = [float(i * 5) for i in range(500)]  


    # imagine we are adding a remote source that contains some existing and some new offsets for this same ARID
    mock_remote_4207561 = [{'source': existing_db_source, 'offset': float(i * 5)} for i in range(495,505)]
    mock_api_client.analysis_result_file.return_value = create_fake_parquet_bytes(mock_remote_4207561)
    
    sources = [{'id': test_arid}]
    
    # If get_offsets_from_db is missing or broken, this line will crash
    iterator = iter_parquet_dataframes(sources, api_client=mock_api_client, db=db)
    results = list(iterator)

    # 5. Assert
    # The iterator should have checked the DB, saw that offsets [495..500]*5 and only yielded the new offsets [501..505]*5
    assert len(results) == 1
    assert len(results[0]) == 5
    assert results[0].iloc[0]['offset'] == 2500.0  # (499 + 1)*5
