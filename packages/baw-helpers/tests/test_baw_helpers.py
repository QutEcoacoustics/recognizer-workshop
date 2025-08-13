import os
import pytest

from baw_helpers import baw_api
from baw_helpers.baw_helpers import simple_filter, get_random_audio_recordings


STAGING_URL = "https://api.staging.ecosounds.org"

# api_token = os.environ.get("API_TOKEN")
# api_base_url = os.environ.get("API_BASE_URL")

@pytest.fixture(scope="module")
def api_client():
    """
    Integration client against BAW staging. Skips if no/invalid token.
    """
    token = os.getenv("STAGING_API_TOKEN")
    if not token:
        pytest.skip("requires STAGING_API_TOKEN")

    client = baw_api.baw_api(
        token=token,
        base_url=STAGING_URL,
        allow_prompt=False,
        debug=True,
    )

    if not client.test_token(default=False):
        pytest.skip("invalid/expired token or service unavailable")

    return client


def test_simple_filter():
    """Test the simple_filter function."""

    # Test with all parameters set
    filter_params = simple_filter(
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
    filter_params = simple_filter(
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
    filter_params = simple_filter()
    assert filter_params is None

@pytest.mark.integration
def test_get_random_audio_recordings():
    
    api = baw_api.baw_api(api_client)

    # region 116 = NSW scorecards greater blue mountains
    filter_params = simple_filter(region_id=116)

    recordings = get_random_audio_recordings(
        baw_api=api,
        filter=filter_params,
        limit=5
    )

    assert isinstance(recordings, list)
    assert len(recordings) == 5
    