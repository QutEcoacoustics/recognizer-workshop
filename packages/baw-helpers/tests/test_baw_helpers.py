import os
from pathlib import Path
import pytest

from baw_helpers import baw_api
from baw_helpers.baw_helpers import simple_filter, get_random_audio_recordings


STAGING_URL = "https://api.staging.ecosounds.org"


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
    filter_params = simple_filter()
    assert filter_params is None

@pytest.mark.integration
def test_get_random_audio_recordings(api_client, tmp_path):
    
    # region 116 = NSW scorecards greater blue mountains
    filter_params = simple_filter(project_ids=1)

    save_dir = tmp_path / "filelist"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run with a fixed seed for determinism
    recordings, cache_path = get_random_audio_recordings(
        api=api_client,
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
        api=api_client,
        filter_params=filter_params,
        save_dir=save_dir,
        limit=5,
        seed=123,
        use_cache=True,
    )
    assert [r["id"] for r in recordings] == [r["id"] for r in recs2]