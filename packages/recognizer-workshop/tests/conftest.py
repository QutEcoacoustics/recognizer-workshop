# tests/conftest.py
import os
from pathlib import Path
import pytest
import numpy as np
import shutil
from scipy.io.wavfile import write as write_wav

from unittest.mock import MagicMock

from perch_hoplite.db import sqlite_usearch_impl

# We rely on your baw_api to read .env (BAW_AUTH_TOKEN) itself via update_token().
# No need to load dotenv here unless you want to; keeping it minimal.

PROD_URL = os.getenv("PROD_API_BASE_URL", "https://api.ecosounds.org")


@pytest.fixture(scope="session")
def prod_api_client():
    """Integration client against production BAW. Skips if token invalid/unreachable.
    We pass token=None and allow_prompt=False so your client will read BAW_AUTH_TOKEN from .env.
    """
    try:
        from baw_helpers.baw_api import baw_api
    except Exception:
        # If you import as: from baw_helpers import baw_api as baw_mod ; baw_mod.baw_api
        from baw_helpers import baw_api as baw_mod
        baw_api = baw_mod.baw_api

    client = baw_api(
        token=None,                 # let update_token() read BAW_AUTH_TOKEN from .env
        base_url=PROD_URL,
        allow_prompt=False,         # don’t prompt in CI
        debug=False,                # flip True locally if you want verbose request logs
    )

    # Quick health check; skip suite if not valid or service down
    if not client.test_token(default=False):
        pytest.skip("BAW_AUTH_TOKEN missing/invalid or service unavailable")

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
