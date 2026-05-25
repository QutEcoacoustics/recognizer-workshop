# tests/conftest.py
import os
import io
from pathlib import Path
import pytest
import numpy as np
import shutil
from scipy.io.wavfile import write as write_wav
from dotenv import load_dotenv, find_dotenv

from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from perch_hoplite.db import sqlite_usearch_impl

# Load .env once for the test session so token-based integration fixtures work in local dev.
load_dotenv(find_dotenv(usecwd=True), override=False)

PROD_URL = os.getenv("PROD_API_BASE_URL", "https://api.ecosounds.org")
STAGING_URL = os.getenv("STAGING_API_BASE_URL", "https://api.staging.ecosounds.org")


def _handle_token_issue(message: str):
    """
    We can specify whether to skip or fail tests when tokens are missing/invalid via PYTEST_TOKEN_POLICY env var.
    - If PYTEST_TOKEN_POLICY=fail(default), the test suite will fail immediately with an error message.
    - If PYTEST_TOKEN_POLICY=skip, the test suite will skip with a message indicating the token issue.
    
    The primary use case for this is so we can streamline CI tests for PRs so that they skip integration tests. 
    For releases, which happen less often, we want to fail if tokens are missing/invalid to catch issues. 
    
    """
    token_policy = os.getenv("PYTEST_TOKEN_POLICY", "fail").strip().lower()
    if token_policy not in {"skip", "fail"}:
        token_policy = "skip"
    if token_policy == "fail":
        pytest.fail(message)
    pytest.skip(message)


def pytest_collection_modifyitems(config, items):
    """
    Dynamically add 'integration' marker to tests that require real API tokens, 
    so they can be easily skipped in CI or run separately.
    """
    _ = config
    for item in items:
        fixturenames = getattr(item, "fixturenames", [])
        if "prod_api_client" in fixturenames or "staging_api_client" in fixturenames:
            item.add_marker(pytest.mark.integration)

@pytest.fixture(scope="session")
def prod_api_client():
    """Integration client against production BAW. Skips if token invalid/unreachable."""
    token = os.getenv("ECOSOUNDS_API_TOKEN")
    if not token:
        _handle_token_issue("ECOSOUNDS_API_TOKEN missing/invalid or service unavailable")

    try:
        from baw_helpers.baw_api import baw_api
    except Exception:
        # If you import as: from baw_helpers import baw_api as baw_mod ; baw_mod.baw_api
        from baw_helpers import baw_api as baw_mod
        baw_api = baw_mod.baw_api

    client = baw_api(
        token=token,
        base_url=PROD_URL,
        allow_prompt=False,     
    )

    # Quick health check; skip suite if not valid or service down
    if not client.test_token(default=False):
        _handle_token_issue("ECOSOUNDS_API_TOKEN missing/invalid or service unavailable")

    return client



@pytest.fixture(scope="session")
def staging_api_client():
    """
    Integration client against STAGING BAW. 
    Requires STAGING_API_TOKEN env var to be set, otherwise skips.
    """
    token = os.getenv("STAGING_API_TOKEN")
    if not token:
        _handle_token_issue("STAGING_API_TOKEN not found in env")

    try:
        from baw_helpers.baw_api import baw_api
    except Exception:
        from baw_helpers import baw_api as baw_mod
        baw_api = baw_mod.baw_api

    # Initialize client against Staging
    client = baw_api(
        token=token,
        base_url=STAGING_URL,
        allow_prompt=False,
    )

    # Health check
    if not client.test_token(default=False):
        _handle_token_issue("STAGING_API_TOKEN is invalid or Staging service is unavailable")

    return client

@pytest.fixture
def mock_embedding_model():
    """Mocks the embedding model to avoid using a real ML model in tests."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings = np.random.rand(1, 1, 1280).astype(np.float32)
    mock_model.embed.return_value = mock_response
    return mock_model


@pytest.fixture
def test_db_path(tmp_path):
    """
    Copies the template test DB into a temporary folder for this test run.
    This ensures the test is isolated and doesn't modify your original test DB.
    """
    source_db_dir = Path("packages/recognizer-workshop/tests/files/db")
    temp_db_dir = tmp_path / "db"
    shutil.copytree(source_db_dir, temp_db_dir)
    return temp_db_dir


@pytest.fixture
def db_with_path(test_db_path):
    """Creates a live connection to the temporary test database."""
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(test_db_path)
    return db, test_db_path


@pytest.fixture
def data_dir(request):
    return Path(request.fspath).parent / "files"


@pytest.fixture
def working_folder(tmp_path):
    """
    Copies the template test DB into a temporary folder for this test run.
    This ensures the test is isolated and doesn't modify your original test DB.
    """

    temp_working_folder = tmp_path / "working_folder"
    (temp_working_folder / 'models').mkdir(exist_ok=True, parents=True)
    (temp_working_folder / 'predictions').mkdir(exist_ok=True, parents=True)
    return temp_working_folder


@pytest.fixture
def audio_files_folder(tmp_path):
    """Creates a temporary folder with two dummy WAV files."""
    audio_dir = tmp_path / "audio_examples"
    audio_dir.mkdir()
    sample_rate = 32000
    
    # Create two dummy 6-second audio files
    for i in range(2):
        # Generate some random noise for audio data
        audio_data = (np.random.uniform(low=-1.0, high=1.0, size=(sample_rate * 6,)) * 32767).astype(np.int16)
        write_wav(audio_dir / f"example_{i}.wav", sample_rate, audio_data)
        
    return audio_dir

@pytest.fixture
def mock_api_client():
    """
    Mocks the baw_api client instance for unit tests.
    Use this when you don't want to hit the real API.
    """
    client = MagicMock()
    return client


@pytest.fixture
def create_fake_parquet_bytes():
    """
    Factory fixture that returns a function to create fake parquet bytes.
    Usage in test:
        def test_thing(create_fake_parquet_bytes):
            buf = create_fake_parquet_bytes([{'col': 1}])
    """
    def _create(data):
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        return buf
    return _create

