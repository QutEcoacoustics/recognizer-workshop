import os
import pytest
from pathlib import Path
import uuid

from baw_helpers import baw_api
from baw_helpers.baw_helpers import annotation_upload_csv
from recognizer_workshop.app.agile_modeling_helpers import get_or_create_workshop_provenance
import pandas as pd

STAGING_URL = "https://api.staging.ecosounds.org"




def test_event_upload(staging_api_client, data_dir, tmp_path):
     
    provenance = get_or_create_workshop_provenance(staging_api_client)

    file_path = data_dir / 'recognizer_output_01.csv'

    df = pd.read_csv(file_path)

    # Add a unique nonce to avoid duplicate rejection
    df['test_nonce'] = uuid.uuid4().hex
    unique_file_path = tmp_path / f"unique_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(unique_file_path, index=False)

    response = annotation_upload_csv(staging_api_client, filepath_to_csv=unique_file_path, provenance_id=provenance['id'], import_name="test_import", commit='true')
    
    # if we get a response back without an http error, we assume success
    assert response is not None, "Response should not be None"