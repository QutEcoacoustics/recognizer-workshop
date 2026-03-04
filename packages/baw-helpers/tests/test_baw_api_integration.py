# tests/test_baw_api_integration.py
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

pytestmark = pytest.mark.integration  # run with: pytest -m integration

def test_status_ok(staging_api_client):
    """GET /status returns 2xx and JSON (or None if empty body)."""
    resp = staging_api_client.get("/status", timeout=15)
    # We don't assume exact schema; just ensure request worked.
    assert staging_api_client.last_response is not None
    assert 200 <= staging_api_client.last_response.status_code < 300
    assert resp is None or isinstance(resp, (dict, list))

def test_baw_filter_smoke(staging_api_client):
    """
    POST /audio_recordings/filter/ with a minimal, harmless filter.
    We don't assume response schema; just assert it is 2xx and JSON-ish.
    """
    # A super-safe filter: IDs greater than 0, and page down the result set
    qsp = {"per": 1, "page": 1}
    payload = {"filter": {"id": {"gt": 0}}}
    resp = staging_api_client.baw_filter("audio_recordings", payload, qsp=qsp)
    # Assert transport worked
    r = staging_api_client.last_response
    assert r is not None and 200 <= r.status_code < 300
    # Assert JSON-ish
    assert resp is None or isinstance(resp, (dict, list))

def test_download_streams_to_disk(staging_api_client, tmp_path):
    """
    Stream a small endpoint to disk to exercise download path.
    /status is JSON but streaming works regardless of content type.
    """
    dest = tmp_path / "status.json"
    saved = staging_api_client.download("/status", str(dest), timeout=15)
    assert saved == str(dest)
    assert dest.exists() and dest.stat().st_size > 0

    # Try to parse what we downloaded (don't fail test if not JSON)
    try:
        txt = dest.read_text("utf-8")
        # Accept either JSON or plain text; prefer to parse JSON if it is
        json.loads(txt)
    except Exception:
        # It’s okay if it’s not JSON; this was a streaming test
        pass

@pytest.mark.parametrize("workers", [1, 3])
def test_thread_safety_smoke(staging_api_client, workers):
    """
    Fire several concurrent GETs to ensure no crashes and sessions per thread behave.
    We don't assert thread-local internals; just that all calls succeed.
    """
    urls = ["/status"] * 5
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(staging_api_client.get, path, timeout=15) for path in urls]
        for f in as_completed(futs):
            resp = f.result()
            # Any of these are acceptable: dict/list JSON body or None (204/empty)
            assert resp is None or isinstance(resp, (dict, list))

