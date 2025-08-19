# tests/conftest.py
import os
import pytest
from dotenv import load_dotenv, find_dotenv

# Load .env once for the whole test run (so STAGING_API_TOKEN is available)
# If you already load it elsewhere, you can remove this.
load_dotenv(find_dotenv(usecwd=True), override=False)

STAGING_URL = os.getenv("STAGING_API_BASE_URL", "https://api.staging.ecosounds.org")

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
        allow_prompt=False,
        debug=False,  # set True locally if you want verbose request/response logs
    )

    # Quick health check; skip suite if not valid or service down
    if not client.test_token(default=False):
        pytest.skip("invalid/expired token or service unavailable")

    return client
