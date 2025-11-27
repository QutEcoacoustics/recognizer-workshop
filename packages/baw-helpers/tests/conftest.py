# tests/conftest.py
import os
import pytest
from unittest.mock import patch
from dotenv import load_dotenv, find_dotenv

# Load .env once for the whole test run (so STAGING_API_TOKEN is available)
# If you already load it elsewhere, you can remove this.
load_dotenv(find_dotenv(usecwd=True), override=False)

STAGING_URL = os.getenv("STAGING_API_BASE_URL", "https://api.staging.ecosounds.org")

# ==================== Environment Fixtures ====================

@pytest.fixture
def mock_env_no_token(monkeypatch):
    """Clear any existing token from environment."""
    monkeypatch.delenv("BAW_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("BAW_ALLOW_PROMPT", raising=False)


# ==================== File System Fixtures ====================

@pytest.fixture
def mock_no_dotenv():
    """Mock that no .env file exists anywhere."""
    with patch('baw_helpers.baw_api.find_dotenv', return_value=None), \
         patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
         patch('baw_helpers.baw_api.os.getenv', return_value=None):
        yield


@pytest.fixture
def mock_dotenv_with_token():
    """Mock that .env file exists with a valid token."""
    with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
         patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
         patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': 'dotenv_token'}):
        yield


@pytest.fixture
def mock_dotenv_empty():
    """Mock that .env file exists but has no token."""
    with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
         patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
         patch('baw_helpers.baw_api.dotenv_values', return_value={}):
        yield


# ==================== Integration Test Fixtures ====================



@pytest.fixture(scope="session")
def staging_token():
    """Provide the staging token or skip the whole suite."""
    token = os.getenv("STAGING_API_TOKEN")
    if not token:
        pytest.skip("requires STAGING_API_TOKEN in environment or .env")
    return token

@pytest.fixture(scope="session")
def api_client(staging_token):
    """
    Integration client against BAW staging. Skips if token invalid/unreachable.
    """
    # Import your class the same way you do in your repo:
    #  - If module file is baw_helpers/baw_api.py:
    #      from baw_helpers.baw_api import baw_api
    #  - If you expose baw_api via package __init__.py:
    #      from baw_helpers import baw_api as baw_mod ; baw_mod.baw_api
    try:
        from baw_helpers.baw_api import baw_api
    except Exception:
        # fallback if you import as "from baw_helpers import baw_api; baw_api.baw_api(...)"
        from baw_helpers import baw_api as baw_mod
        baw_api = baw_mod.baw_api

    client = baw_api(
        token=staging_token,
        base_url=STAGING_URL,
        allow_prompt=False
    )

    # Quick health check; skip suite if not valid or service down
    if not client.test_token(default=False):
        pytest.skip("invalid/expired token or service unavailable")

    return client


@pytest.fixture(scope="session")
def anonymous_client():
    """
    Anonymous client (no token) for testing public endpoints.
    """
    try:
        from baw_helpers.baw_api import baw_api
    except Exception:
        from baw_helpers import baw_api as baw_mod
        baw_api = baw_mod.baw_api

    client = baw_api(
        token=None,
        base_url=STAGING_URL,
        allow_prompt=False,
        debug=False,
    )
    
    return client