"""
Tests for loading embedding configuration in create_database. 

Pure perch-hoplite will insert a configuration dict in the database metadata during embedding. 
We are pulling parquet files from the workbench api, so we insert these configs in a different way. 
These test this functionality. 


"""


from pathlib import Path
import pytest
from ml_collections import config_dict

from recognizer_workshop.app.create_database.create_database import _load_embedding_config, DEFAULT_BAW_ANALYSIS_NAME


def test_dict_input_returns_configdict():
    src = {"a": 1, "b": {"c": 2}}
    out = _load_embedding_config(src)
    assert isinstance(out, config_dict.ConfigDict)
    assert out.to_dict() == src


def test_configdict_input_passthrough_as_configdict():
    src = config_dict.ConfigDict({"x": 10})
    out = _load_embedding_config(src)
    assert isinstance(out, config_dict.ConfigDict)
    assert out.to_dict() == {"x": 10}


def test_none_uses_default_named_config(monkeypatch):
    called = {}

    def fake_load_baw(name):
        called["name"] = name
        # Return a distinctive object so we can assert identity/equality.
        return {"model_key": "dummy", "name": name}

    # Patch in the module-under-test namespace
    import recognizer_workshop.app.create_database.create_database as mod
    monkeypatch.setattr(mod, "load_baw_embedding_config", fake_load_baw)

    out = _load_embedding_config(None)
    assert called["name"] == DEFAULT_BAW_ANALYSIS_NAME
    assert isinstance(out, config_dict.ConfigDict)
    assert out["name"] == DEFAULT_BAW_ANALYSIS_NAME


def test_path_input_calls_load_embedding_config(monkeypatch, tmp_path):
    # Create a dummy path that "exists"
    cfg_dir = tmp_path / "some_cfg_dir"
    cfg_dir.mkdir()

    seen = {}

    def fake_loader(path):
        seen["path"] = Path(path)
        return {"loaded_from": str(path)}

    import recognizer_workshop.app.create_database.create_database as mod
    monkeypatch.setattr(mod, "load_embedding_config", fake_loader)

    out = _load_embedding_config(cfg_dir)
    assert seen["path"] == cfg_dir
    assert isinstance(out, dict) or isinstance(out, config_dict.ConfigDict)
    # If your loader returns a dict, the function currently passes it through as-is


def test_str_named_config_calls_load_baw_and_returns(monkeypatch):
    called = {}

    def fake_load_baw(name):
        called["name"] = name
        return {"model_key": "dummy", "name": name}

    # Ensure Path(config).exists() is False for this test so we hit the str branch.
    import recognizer_workshop.app.create_database.create_database as mod
    monkeypatch.setattr(mod, "load_baw_embedding_config", fake_load_baw)

    # Make sure we don't accidentally trip the Path.exists branch
    monkeypatch.setattr(Path, "exists", lambda self: False, raising=True)

    cfg_name = "perch-embeddings_4"
    out = _load_embedding_config(cfg_name)
    assert called["name"] == cfg_name
    # This is what you want once you add `return legacy_config` in the function:
    assert isinstance(out, config_dict.ConfigDict)
    assert out["name"] == cfg_name


def test_unsupported_type_raises_type_error():
    with pytest.raises(TypeError):
        _load_embedding_config(12345)


def test_missing_path_raises_filenotfound_error():
    with pytest.raises(FileNotFoundError):
        _load_embedding_config(Path("non_existent/path"))


def test_unsupported_named_config_raises_filenotfound_error():
    with pytest.raises(FileNotFoundError):
        _load_embedding_config("pearch-imboddings_88")


