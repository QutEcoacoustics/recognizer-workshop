import re
import pytest
from importlib.metadata import PackageNotFoundError, version as metadata_version
from recognizer_workshop.app.services import version as version_service


def test_get_version_status_string_version_only(monkeypatch):
    monkeypatch.setattr(version_service, "get_installed_package_version", lambda: "2026.5.22")
    monkeypatch.setattr(version_service, "get_git_commit_short", lambda: "abcd123")
    result = version_service.get_version_status_string()
    assert result == "2026.5.22"


def test_get_version_status_string_includes_hash_when_enabled(monkeypatch):
    monkeypatch.setattr(version_service, "get_installed_package_version", lambda: "2026.5.22")
    monkeypatch.setattr(version_service, "get_git_commit_short", lambda: "abcd123")
    result = version_service.get_version_status_string(include_git_hash=True)
    assert result == "2026.5.22 (abcd123)"


def test_get_version_status_string_unknown_when_version_missing(monkeypatch):
    monkeypatch.setattr(version_service, "get_installed_package_version", lambda: None)
    monkeypatch.setattr(version_service, "get_git_commit_short", lambda: None)
    result = version_service.get_version_status_string(include_git_hash=True)
    assert result == "unknown"


# The following tests are more integration-style tests that check the actual behavior of the functions without mocking. 
# They only test the shape of the output string.

def test_get_git_commit():
    commit_hash = version_service.get_git_commit_short()
    if commit_hash is None:
        pytest.skip("No git metadata available in this test environment")
    assert re.fullmatch(r"[0-9a-f]{7,40}", commit_hash)


def test_get_installed_package_version():
    try:
        expected_version = metadata_version("recognizer-workshop")
    except PackageNotFoundError:
        # We are testing the service that returns the package version so that we can display it in the UI. 
        # This might fail if we are running this from source without actually installing the package. 
        # For this test to pass, we should have the package installed in the test environment. 
        pytest.fail("recognizer-workshop is not installed in this test environment")

    installed_version = version_service.get_installed_package_version()

    assert installed_version == expected_version
    assert isinstance(installed_version, str)
    assert installed_version.strip() == installed_version
    assert installed_version != ""


def test_get_version_status_string():
    status = version_service.get_version_status_string(include_git_hash=True)
    assert isinstance(status, str)
    assert status.strip() == status
    assert status != ""
    assert not status.startswith("version:")

    if " (" in status:
        version_part, suffix = status.split(" (", 1)
        assert version_part != ""
        assert suffix.endswith(")")
        commit_hash = suffix[:-1]
        assert re.fullmatch(r"[0-9a-f]{7,40}", commit_hash)