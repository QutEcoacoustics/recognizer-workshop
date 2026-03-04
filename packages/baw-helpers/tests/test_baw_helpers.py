import os
from pathlib import Path
import pytest

from baw_helpers import baw_api
from baw_helpers.baw_helpers import simple_filter, get_random_audio_recordings, get_events_for_sites, get_embedding_counts
from baw_helpers.baw_helpers import sample_filter_pages

STAGING_URL = "https://api.staging.ecosounds.org"


def test_simple_filter():
    """Test the simple_filter function."""

    # Test with all parameters set
    filter_params = simple_filter(
        project_ids=1,
        region_ids=2,
        site_ids=3,
        arids=4
    )
    expected = {
        "filter": {
            "projects.id": {"eq": 1},
            "regions.id": {"eq": 2},
            "sites.id": {"eq": 3},
            "id": {"eq": 4}
        }
    }
    assert filter_params == expected

    # Test with some parameters set to None
    filter_params = simple_filter(
        project_ids=None,
        region_ids=2,
        site_ids=[1,2,3,4],
        arids=4
    )
    expected = {
        "filter": {
            "regions.id": {"eq": 2},
            "sites.id": {"in": [1, 2, 3, 4]},
            "id": {"eq": 4}
        }
    }
    assert filter_params == expected

    # Test with all parameters None
    with pytest.raises(ValueError, match="At least one filter parameter must be provided"):
        simple_filter()

@pytest.mark.integration
def test_get_random_audio_recordings(staging_api_client, tmp_path):
    
    # region 116 = NSW scorecards greater blue mountains
    filter_params = simple_filter(project_ids=1)

    save_dir = tmp_path / "filelist"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run with a fixed seed for determinism
    recordings, cache_path = get_random_audio_recordings(
        api=staging_api_client,
        filter_params=filter_params,
        save_dir=save_dir,
        limit=5,
        seed=123,
        use_cache=True,
    )

    # 1) Correct types
    assert isinstance(recordings, list)
    assert isinstance(cache_path, Path)

    # 2) Length matches limit
    assert len(recordings) == 5

    # 3) Cache file was written
    assert cache_path.exists()
    assert cache_path.parent == save_dir

    # 4) Determinism: running again with the same seed yields same IDs
    recs2, _ = get_random_audio_recordings(
        api=staging_api_client,
        filter_params=filter_params,
        save_dir=save_dir,
        limit=5,
        seed=123,
        use_cache=True,
    )
    assert [r["id"] for r in recordings] == [r["id"] for r in recs2]


    from baw_helpers.baw_helpers import get_events_for_sites  # Ensure this is imported

@pytest.mark.integration
def test_get_events_for_sites(prod_api_client):
    """
    Integration test for get_events_for_sites.
    Verifies that the filter structure, projection, and tag hydration work 
    against the real BAW API.
    """
    # 1. Setup Filter (Using Project 1 as a reliable source of test data)
    #    Note: simple_filter returns a dict like {'filter': {...}}
    filter_params = simple_filter(project_ids=[1084])

    # 2. Call the function under test
    #    This will likely fail initially if the filter nesting is incorrect
    events = get_events_for_sites(prod_api_client, filter_params)

    # 3. Basic Assertions
    assert isinstance(events, list), "Expected a list of events"

    # 4. Deep Structure Assertions
    #    Only run these if we actually found events (which we hope we do in Proj 1)
    if len(events) > 0:
        event = events[0]
        
        # Check standard fields
        assert 'id' in event
        assert 'audio_recording_id' in event
        
        # Check the specific projection you are worried about
        # If the API ignores the 'add' projection, this key will be missing
        assert 'verification_summary' in event, (
            "verification_summary missing. Projection likely malformed."
        )
        
        # Check Tag Hydration
        # Your function manually fetches tags and inserts them into 'taggings'
        if event.get('taggings'):
            tagging = event['taggings'][0]
            assert 'tag' in tagging, "Tag details were not hydrated into the tagging object"
            assert 'text' in tagging['tag'], "Hydrated tag object is missing 'text'"
            
    else:
        # Warning if no data found, so we know the test didn't really 'test' the hydration
        pytest.warns(UserWarning, match="No events found in Project 1. Hydration logic was not exercised.")


def test_get_embedding_counts(prod_api_client):
    """Test the get_embedding_counts function."""

    # 1. Setup Filter (Using Project 1 or a known stable project)
    filter_params = simple_filter(project_ids=1084)

    # 2. Call Function
    # Using a known job ID on staging (e.g., default 3 is usually safe, or use 1)
    result = get_embedding_counts(
        baw_api=prod_api_client,
        filter_params=filter_params
    )

    assert result is not None, "Expected a result dictionary"



def test_sample_filter_pages_paging_and_caching(mock_api_client, tmp_path):
    """
    Test that get_filter_result:
    1. Automatically pages through results until a partial page is found.
    2. Aggregates all items into a single list.
    3. Caches the result to disk.
    4. Uses the cache on subsequent calls.
    """


    # --- Setup ---
    save_dir = tmp_path / "cache_test"
    filter_params = {"filter": {"id": {"gt": 0}}}
    model = "projects"
    page_size = 2  # Small page size to force multiple iterations
    common_meta = {"paging": {"total": 5}}

    # --- Mocking the API Paging ---
    # We simulate 3 pages:
    # Page 1: Full (2 items) -> Loop continues
    # Page 2: Full (2 items) -> Loop continues
    # Page 3: Partial (1 item) -> Loop ends
    mock_api_client.baw_filter.side_effect = [
        {"data": [{"id": 1}, {"id": 2}], "meta": common_meta}, 
        {"data": [{"id": 3}, {"id": 4}], "meta": common_meta},
        {"data": [{"id": 5}],            "meta": common_meta}             
    ]

    # --- Execution 1 (API Fetch) ---
    data, cache_path = sample_filter_pages(
        api=mock_api_client,
        model=model,
        filter_params=filter_params,
        save_path=save_dir,
        page_size=page_size,
        use_cache=True
    )

    # --- Assertions 1: Aggregation & Logic ---
    # 1. Check total items (2 + 2 + 1 = 5)
    assert len(data) == 5
    assert data[0]['id'] == 1
    assert data[-1]['id'] == 5
    
    # 2. Check API usage
    # Should have been called 3 times
    assert mock_api_client.baw_filter.call_count == 3
    
    # Verify the 'page' parameter incremented correctly in the calls
    calls = mock_api_client.baw_filter.call_args_list
    assert calls[0].kwargs['qsp']['page'] == 1
    assert calls[1].kwargs['qsp']['page'] == 2
    assert calls[2].kwargs['qsp']['page'] == 3
    
    # 3. Check Cache File creation
    assert cache_path.exists()
    assert cache_path.parent == save_dir

    # --- Execution 2 (Cache Hit) ---
    # Reset the mock to ensure we don't call the API again
    mock_api_client.reset_mock()
    
    data_cached, path_cached = sample_filter_pages(
        api=mock_api_client,
        model=model,
        filter_params=filter_params,
        save_path=save_dir,
        page_size=page_size,
        use_cache=True
    )

    # --- Assertions 2: Cache Behavior ---
    # 1. Data should be identical
    assert data_cached == data
    assert path_cached == cache_path
    
    # 2. API should NOT have been called
    mock_api_client.baw_filter.assert_not_called()


@pytest.mark.integration
def test_sample_filter_pages_real_api_shape_and_cache(staging_api_client, tmp_path):
    save_dir = tmp_path / "sample_filter_pages_real"
    filter_params = simple_filter(project_ids=1)
    filter_params["projection"] = {
            "include": [
                "id",
                "site_id",
                "canonical_file_name",
                "sites.name",
                "regions.name",
                "regions.id",
                "duration_seconds",
            ]
        }
    data, cache_path = sample_filter_pages(
        api=staging_api_client,
        model="audio_recordings",
        filter_params=filter_params,
        save_path=save_dir,
        page_size=50,
        limit=120,
        use_cache=True,
    )

    assert isinstance(data, list)
    assert cache_path.exists()
    assert cache_path.parent == save_dir
    assert all(not isinstance(item, list) for item in data)

    if len(data) > 0:
        assert isinstance(data[0], dict)

    data_cached, path_cached = sample_filter_pages(
        api=staging_api_client,
        model="audio_recordings",
        filter_params=filter_params,
        save_path=save_dir,
        page_size=50,
        limit=120,
        use_cache=True,
    )

    assert path_cached == cache_path
    assert isinstance(data_cached, list)
    assert all(not isinstance(item, list) for item in data_cached)