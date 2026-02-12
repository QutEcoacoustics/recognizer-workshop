# tests/conftest.py
import os
import pytest
from unittest.mock import patch
from dotenv import load_dotenv, find_dotenv
from unittest.mock import MagicMock

# Load .env once for the whole test run (so STAGING_API_TOKEN is available)
# If you already load it elsewhere, you can remove this.
load_dotenv(find_dotenv(usecwd=True), override=False)

STAGING_URL = os.getenv("STAGING_API_BASE_URL", "https://api.staging.ecosounds.org")
PROD_URL = os.getenv("BAW_API_BASE_URL", "https://api.ecosounds.org")

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
def prod_token():
    """Provide the prod token or skip the whole suite."""
    # We explicitly check for BAW_AUTH_TOKEN here so we can skip gracefully
    token = os.getenv("BAW_AUTH_TOKEN")
    if not token:
        pytest.skip("requires BAW_AUTH_TOKEN in environment or .env")
    return token


def _create_cached_client(token, base_url, tmp_path_factory, env_name):
    """
    Helper to create an authenticated client with tag caching enabled.
    DRY logic for both Staging and Prod clients.
    """
    # Import logic matching your repo structure
    try:
        from baw_helpers.baw_api import baw_api
    except Exception:
        from baw_helpers import baw_api as baw_mod
        baw_api = baw_mod.baw_api

    client = baw_api(
        token=token,
        base_url=base_url,
        allow_prompt=False,
        anonymous=(token is None)
    )

    # Health check
    if not client.test_token(default=False):
        pytest.skip(f"Invalid/expired {env_name} token or service unavailable")

    # Enable Tag Cache
    # We use a unique filename per env to prevent collisions
    cache_dir = tmp_path_factory.mktemp("baw_cache")
    cache_filename = f"{env_name.lower()}_tags_cache.json"
    cache_file = cache_dir / cache_filename
    
    client.enable_tag_cache(cache_file=str(cache_file))

    return client


@pytest.fixture(scope="session")
def staging_api_client(staging_token, tmp_path_factory):
    """
    Integration client against BAW staging with tag cache enabled.
    """
    return _create_cached_client(
        token=staging_token,
        base_url=STAGING_URL,
        tmp_path_factory=tmp_path_factory,
        env_name="Staging"
    )


@pytest.fixture(scope="session")
def prod_api_client(prod_token, tmp_path_factory):
    """
    Integration client against BAW production with tag cache enabled.
    """
    return _create_cached_client(
        token=prod_token,
        base_url=PROD_URL,
        tmp_path_factory=tmp_path_factory,
        env_name="Production"
    )


@pytest.fixture(scope="session")
def anonymous_client(tmp_path_factory):
    """
    Anonymous client (no token) for testing public endpoints.
    """
    return _create_cached_client(
        token=None,
        base_url=PROD_URL,
        tmp_path_factory=tmp_path_factory,
        env_name="Anonymous"
    )

@pytest.fixture
def mock_api_client():
    """
    Mocks the baw_api client instance for unit tests.
    Use this when you don't want to hit the real API.
    """
    client = MagicMock()
    return client