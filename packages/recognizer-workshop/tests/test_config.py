import json
from pathlib import Path
import pytest

# Import the classes from your file
# Make sure this import path is correct for your project structure
from recognizer_workshop.app.agile_engine import AgileConfig, LabelName, get_recent_files, _write_app_data, _APP_DATA_FILE


class TestAgileConfig:
    """Unit tests for the AgileConfig class using pytest."""

    def test_reset_method(self):
        """Test that the reset() method restores an instance to default values."""
        config = AgileConfig(annotator_id="test_user")
        assert config.annotator_id == "test_user"

        config.reset()
        assert config.annotator_id == "default_annotator"
        assert config.db_path is None
        assert config.config_path is None


    def test_default_creation(self):
        """Test that a default config object is created with correct defaults."""
        config = AgileConfig()
        assert config.db_path is None
        assert config.annotator_id == "default_annotator"
        assert config.target_labels == []
        assert config.baw_config == {}


    def test_post_init_directory_creation(self, tmp_path: Path):
        """Test that __post_init__ creates non-existent directories."""
        models_path = tmp_path / "models_folder"
        preds_path = tmp_path / "preds" / "run1"

        # Ensure paths do not exist before creating the config
        assert not models_path.exists()
        assert not preds_path.exists()

        # Create the config instance, which should trigger __post_init__
        config = AgileConfig(
            models_folder=models_path,
            predictions_folder=preds_path
        )

        # Assert that the directories were created
        assert models_path.exists()
        assert preds_path.exists()
        # db_path parent is checked for existence but not created
        assert config.db_path is None


    def test_save_as_and_load_preserves_data(self, tmp_path: Path):
        """Tests the save_as -> load roundtrip preserves all data and types."""
        db_path_abs = tmp_path / "data" / "database.sqlite"
        
        original_config = AgileConfig(
            db_path=db_path_abs,
            annotator_id="tester-01",
            target_labels=[LabelName("Koala", "Bellow")]
        )
        config_path = tmp_path / "config.json"

        original_config.save_as(config_path)
        
        # The instance should now know its own path
        assert original_config.config_path == config_path

        loaded_config = AgileConfig.load(config_path)

        assert loaded_config == original_config
        assert isinstance(loaded_config.db_path, Path)
        assert loaded_config.config_path == config_path


    def test_save_updates_file(self, tmp_path: Path):
        """Tests that save() correctly overwrites the file with new data."""
        config_path = tmp_path / "config.json"
        config = AgileConfig(annotator_id="version1")
        config.save_as(config_path)
        config.annotator_id = "version2"
        config.save()
        loaded_config = AgileConfig.load(config_path)
        assert loaded_config.annotator_id == "version2"


    def test_save_without_path_raises_error(self):
        """Tests that save() fails if the config has never been saved."""
        config = AgileConfig() # This config has no path
        with pytest.raises(ValueError, match="Use 'save_as"):
            config.save()


    def test_load_non_existent_raises_error(self, tmp_path: Path):
        """Tests that load() fails correctly if the file does not exist."""
        non_existent_path = tmp_path / "ghost.json"
        with pytest.raises(FileNotFoundError):
            AgileConfig.load(non_existent_path)


    def test_recent_files_logic(self, monkeypatch, tmp_path: Path):
        """Tests the recent files list management by patching the getter function."""
        
        temp_app_data_file = tmp_path / ".test_app_data.json"
        monkeypatch.setattr(
            "recognizer_workshop.app.agile_engine._get_app_data_file",
            lambda: temp_app_data_file
        )
        
        if temp_app_data_file.exists():
            temp_app_data_file.unlink()
            
        path1 = tmp_path / "proj1.json"
        path2 = tmp_path / "proj2.json"
        path3 = tmp_path / "proj3.json"

        config = AgileConfig()
        config.save_as(path1)
        assert get_recent_files() == [str(path1)]
        
        config.save_as(path2)
        assert get_recent_files() == [str(path2), str(path1)]
        
        config.save_as(path3)
        assert get_recent_files() == [str(path3), str(path2), str(path1)]

        # Loading a file should move it to the top
        AgileConfig.load(path1)
        assert get_recent_files() == [str(path1), str(path3), str(path2)]



    def test_save_and_load_roundtrip_relative(self, tmp_path: Path):
        """Test the full save/load cycle with default relative paths."""
        config_path = tmp_path / "config.json"
        db_path_abs = tmp_path / "data" / "database.sqlite"
        
        config = AgileConfig(
            db_path=db_path_abs,
            annotator_id="tester-01",
            target_labels=[LabelName("Koala", "Bellow")]
        )
        
        config._to_json(config_path)
        assert config_path.exists()

        with open(config_path, 'r') as f:
            data = json.load(f)
        assert data['db_path'] == 'data/database.sqlite'
        assert data['annotator_id'] == 'tester-01'

        loaded_config = AgileConfig.from_json(config_path)

        assert isinstance(loaded_config.db_path, Path)
        assert loaded_config.db_path == db_path_abs
        assert loaded_config.annotator_id == "tester-01"
        assert len(loaded_config.target_labels) == 1
        assert loaded_config.target_labels[0].species == "Koala"


    def test_save_absolute_paths(self, tmp_path: Path):
        """Test saving the config with absolute paths using relative_to='/'."""
        config_path = tmp_path / "abs_config.json"
        db_path_abs = tmp_path / "database.sqlite"
        
        config = AgileConfig(db_path=db_path_abs)
        
        # Save with the special flag to force absolute paths
        config._to_json(config_path, relative_to="/")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
            
        assert data['db_path'] == str(db_path_abs.as_posix())


    def test_update_config_and_save(self, tmp_path: Path):
        """Test that updating an instance and saving it persists changes."""
        config_path = tmp_path / "updated_config.json"

        # 1. Create a default config and save it
        config = AgileConfig()
        config._to_json(config_path)

        # 2. Update the attributes
        config.annotator_id = "final_annotator"
        config.db_path = tmp_path / "production.db"
        config.target_labels.append(LabelName("Frog"))

        # 3. Save the modified config over the old one
        config._to_json(config_path)

        # 4. Load it back
        reloaded_config = AgileConfig.from_json(config_path)

        # 5. Assert the changes were persisted
        assert reloaded_config.annotator_id == "final_annotator"
        assert reloaded_config.db_path == tmp_path / "production.db"
        assert len(reloaded_config.target_labels) == 1
        assert reloaded_config.target_labels[0].species == "Frog"



    def test_full_config_roundtrip_preserves_data_and_types(self, tmp_path: Path):
        """
        Tests that saving a complex config and loading it back results in an
        identical object with all data types correctly restored.
        """
        # 1. Create a config with arbitrary, non-default values for all fields
        db_path = tmp_path / "test_db_path"
        embeddings_path = tmp_path / "embeddings"
        
        original_config = AgileConfig(
            db_path=db_path,
            embeddings_folder=embeddings_path,
            labeled_examples_folder=tmp_path / "examples",
            models_folder=tmp_path / "models",
            predictions_folder=tmp_path / "predictions",
            annotator_id="test-annotator-123",
            search_dataset_name="custom_search_set",
            max_embeddings_count=50000,
            baw_config={"api_url": "https://example.com", "timeout": 30},
            target_labels=[
                LabelName(species="Koala", call_type="Bellow"),
                LabelName(species="Frog", call_type=None)
            ]
        )

        config_path = tmp_path / "full_config.json"

        original_config._to_json(config_path)
        assert config_path.exists()

        loaded_config = AgileConfig.from_json(config_path)

        # because it's a dataclass, this checks all fields for equality
        assert original_config == loaded_config

        # check that dacite's type hooks worked as expected.
        assert isinstance(loaded_config.db_path, Path)
        assert loaded_config.db_path == db_path