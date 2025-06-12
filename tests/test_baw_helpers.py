import os

import recognizer_workshop.baw_helpers as baw_helpers
import recognizer_workshop.baw_api as baw_api


api_token = os.environ.get("API_TOKEN")
api_base_url = os.environ.get("API_BASE_URL")


def test_simple_filter():
    """Test the simple_filter function."""
    # Test with all parameters set
    filter_params = baw_helpers.simple_filter(
        project_id=1,
        region_id=2,
        site_id=3,
        arid=4
    )
    expected = {
        "projects.id": {"eq": 1},
        "regions.id": {"eq": 2},
        "sites.id": {"eq": 3},
        "id": {"eq": 4}
    }
    assert filter_params == expected

    # Test with some parameters set to None
    filter_params = baw_helpers.simple_filter(
        project_id=None,
        region_id=2,
        site_id=[1,2,3,4],
        arid=4
    )
    expected = {
        "regions.id": {"eq": 2},
        "sites.id": {"in": [1, 2, 3, 4]},
        "id": {"eq": 4}
    }
    assert filter_params == expected

    # Test with all parameters None
    filter_params = baw_helpers.simple_filter()
    assert filter_params is None


def test_get_random_audio_recordings():
    
    api = baw_api.baw_api(api_token, api_base_url, debug=True)

    # region 116 = NSW scorecards greater blue mountains
    filter_params = baw_helpers.simple_filter(region_id=116)

    recordings = baw_helpers.get_random_audio_recordings(
        baw_api=api,
        filter=filter_params,
        limit=5
    )

    assert isinstance(recordings, list)
    assert len(recordings) == 5
    