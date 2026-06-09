"""Tests for AppSettings, particularly enabled_domains filtering."""

import json
import os
import pytest
from unittest.mock import patch
from baw_helpers.settings import BawSettings


@pytest.fixture(autouse=True)
def _fresh_baw_settings():
    """Ensure each test gets a fresh BawSettings instance."""
    fresh = BawSettings()
    with patch("recognizer_workshop.app.app_settings.baw_settings", fresh):
        yield fresh


@pytest.fixture()
def _make_settings(tmp_path):
    """Factory to create AppSettings with a given config file."""
    from recognizer_workshop.app.app_settings import AppSettings

    def _factory(config: dict | None = None, env_path: str | None = None):
        if config is not None and env_path is None:
            settings_file = tmp_path / "app_settings.json"
            settings_file.write_text(json.dumps(config))
            env_path = str(settings_file)

        env = {**os.environ}
        if env_path:
            env["RW_APP_SETTINGS"] = env_path
        else:
            env.pop("RW_APP_SETTINGS", None)

        with patch.dict(os.environ, env, clear=True):
            return AppSettings()

    return _factory


class TestEnabledDomainsDefault:
    """Default enabled_domains returns all domains when all are enabled."""

    def test_returns_both_workbenches_by_default(self, _make_settings):
        settings = _make_settings()
        domains = settings.enabled_domains
        assert "ecosounds" in domains
        assert "a2o" in domains

    def test_default_matches_baw_settings_domains(self, _make_settings, _fresh_baw_settings):
        settings = _make_settings()
        assert settings.enabled_domains == _fresh_baw_settings.domains


class TestEnabledDomainsFiltering:
    """enabled_domains filters based on enabled_workbenches."""

    def test_single_workbench(self, _make_settings):
        settings = _make_settings({"enabled_workbenches": ["ecosounds"]})
        domains = settings.enabled_domains
        assert "ecosounds" in domains
        assert "a2o" not in domains

    def test_preserves_domain_info(self, _make_settings, _fresh_baw_settings):
        settings = _make_settings({"enabled_workbenches": ["ecosounds"]})
        assert settings.enabled_domains["ecosounds"] == _fresh_baw_settings.domains["ecosounds"]

    def test_empty_list_returns_nothing(self, _make_settings):
        settings = _make_settings({"enabled_workbenches": []})
        assert settings.enabled_domains == {}

    def test_unknown_workbench_ignored(self, _make_settings):
        settings = _make_settings({"enabled_workbenches": ["nonexistent"]})
        assert settings.enabled_domains == {}

    def test_mix_of_valid_and_invalid(self, _make_settings):
        settings = _make_settings({"enabled_workbenches": ["ecosounds", "nonexistent"]})
        domains = settings.enabled_domains
        assert list(domains.keys()) == ["ecosounds"]


class TestLoadConfig:
    """load_config reads JSON and updates settings."""

    def test_loads_from_file(self, _make_settings):
        settings = _make_settings({"default_workbench": "ecosounds"})
        assert settings.default_workbench == "ecosounds"

    def test_no_file_uses_defaults(self, _make_settings):
        settings = _make_settings()
        assert settings.enabled_workbenches == ["ecosounds", "a2o"]
        assert settings.default_workbench == "ecosounds"
