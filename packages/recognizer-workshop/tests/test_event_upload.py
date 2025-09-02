import os
import pytest
from pathlib import Path

from baw_helpers import baw_api
from baw_helpers.baw_helpers import annotation_upload_csv
from recognizer_workshop.app.agile_modeling_helpers import get_or_create_workshop_provenance

STAGING_URL = "https://api.staging.ecosounds.org"

@pytest.fixture(scope="module")
def api_client():
    token = os.getenv("STAGING_API_TOKEN")
    if not token:
        pytest.skip("requires STAGING_API_TOKEN")

    client = baw_api.baw_api(token=token, base_url=STAGING_URL, allow_prompt=False, debug=True)

    if not client.test_token(default=False):
        pytest.skip("invalid/expired token or service unavailable")

    return client



@pytest.fixture
def data_dir(request):
    return Path(request.fspath).parent / "files"

def test_event_upload(api_client, data_dir):
     
    provenance = get_or_create_workshop_provenance(api_client)

    file_path = data_dir / 'recognizer_output_01.csv'

    response = annotation_upload_csv(api_client, filepath_to_csv=file_path, provenance_id=provenance['id'], import_name="test_import", commit='true')

    # todo: make a proper test - this was just an interactive one
    assert response is not None, "Response should not be None"