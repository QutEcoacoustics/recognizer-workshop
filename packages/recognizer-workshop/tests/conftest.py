# tests/conftest.py
import os
from pathlib import Path
import pytest

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


