import os

from recognizer_workshop import baw_api
from recognizer_workshop.baw_helpers import annotation_upload_csv
from recognizer_workshop.agile_modeling_helpers import get_or_create_workshop_provenance

staging_token = os.environ.get('STAGING_API_TOKEN')

api = baw_api.baw_api(token=staging_token, base_url='https://api.staging.ecosounds.org', debug=True)

def test_event_upload():
 
    
    provenance = get_or_create_workshop_provenance(api)

    file_path = 'tests/files/recognizer_output_01.csv'

    response = annotation_upload_csv(api, filepath_to_csv=file_path, provenance_id=provenance['id'], commit='true')

    # todo: make a proper test - this was just an interactive one
    assert response is not None, "Response should not be None"