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


    def test_path_access_side_effects(self, tmp_path: Path):
        """
        Test to reproduce the issue where accessing one path property 
        causes other path properties to become None.
        """
        config = AgileConfig()
        
        # Set all paths to relative Path objects
        config.db_path = Path("db")
        config.labeled_examples_folder = Path("labeled_examples")
        config.models_folder = Path("models")
        config.predictions_folder = Path("predictions")
        
        print("After setting all paths:")
        print(f"  db_path: {config.db_path}")
        print(f"  labeled_examples_folder: {config.labeled_examples_folder}")
        print(f"  models_folder: {config.models_folder}")
        print(f"  predictions_folder: {config.predictions_folder}")
        
        # Verify all paths are set correctly
        assert config.db_path == Path("db")
        assert config.labeled_examples_folder == Path("labeled_examples")
        assert config.models_folder == Path("models")
        assert config.predictions_folder == Path("predictions")
        
        # Now access db_path (this is what triggers the issue in the UI)
        db_path_str = str(config.db_path or "")
        print(f"\nAfter accessing db_path as string: '{db_path_str}'")
        print(f"  db_path: {config.db_path}")
        print(f"  labeled_examples_folder: {config.labeled_examples_folder}")
        print(f"  models_folder: {config.models_folder}")
        print(f"  predictions_folder: {config.predictions_folder}")
        
        # Check if other paths disappeared (this is the bug we're testing for)
        assert config.db_path == Path("db"), "db_path should remain unchanged"
        assert config.labeled_examples_folder == Path("labeled_examples"), "labeled_examples_folder disappeared!"
        assert config.models_folder == Path("models"), "models_folder disappeared!"
        assert config.predictions_folder == Path("predictions"), "predictions_folder disappeared!"
        
        # Also test accessing other paths individually
        labeled_str = str(config.labeled_examples_folder or "")
        models_str = str(config.models_folder or "")
        preds_str = str(config.predictions_folder or "")
        
        print(f"\nAfter accessing all paths as strings:")
        print(f"  db_path: {config.db_path}")
        print(f"  labeled_examples_folder: {config.labeled_examples_folder}")
        print(f"  models_folder: {config.models_folder}")
        print(f"  predictions_folder: {config.predictions_folder}")
        
        # Final verification that all paths are still intact
        assert config.db_path == Path("db")
        assert config.labeled_examples_folder == Path("labeled_examples") 
        assert config.models_folder == Path("models")
        assert config.predictions_folder == Path("predictions")


    def test_ui_workflow_reproduction(self, tmp_path: Path):
        """
        Test to reproduce the exact workflow from the UI config.py that causes paths to disappear.
        This mimics the sequence: set_default_paths_if_empty() -> sync_config_to_ui()
        """
        # Simulate the UI workflow
        config = AgileConfig()  # Fresh config like is_new_unsaved_config = True
        
        # Step 1: set_default_paths_if_empty() equivalent
        default_paths = {
            'db_path': Path("db"),
            'labeled_examples_folder': Path("labeled_examples"),
            'models_folder': Path("models"),
            'predictions_folder': Path("predictions"),
        }
        
        print("=== Step 1: Setting default paths ===")
        if not config.db_path:
            config.db_path = default_paths['db_path']
        if not config.labeled_examples_folder:
            config.labeled_examples_folder = default_paths['labeled_examples_folder']
        if not config.models_folder:
            config.models_folder = default_paths['models_folder']
        if not config.predictions_folder:
            config.predictions_folder = default_paths['predictions_folder']
            
        print(f"After setting defaults: db={config.db_path}, labeled={config.labeled_examples_folder}, models={config.models_folder}, preds={config.predictions_folder}")
        
        # Verify all paths are set
        assert config.db_path == Path("db")
        assert config.labeled_examples_folder == Path("labeled_examples")
        assert config.models_folder == Path("models") 
        assert config.predictions_folder == Path("predictions")
        
        # Step 2: sync_config_to_ui() equivalent - accessing paths one by one
        print("\n=== Step 2: Simulating sync_config_to_ui() ===")
        print("Before accessing db_path:")
        print(f"  db_path: {config.db_path}")
        print(f"  labeled_examples_folder: {config.labeled_examples_folder}")
        print(f"  models_folder: {config.models_folder}")
        print(f"  predictions_folder: {config.predictions_folder}")
        
        # This is the exact pattern from config.py that causes the issue
        db_path_value = str(config.db_path or "")
        print(f"\nAfter accessing db_path: '{db_path_value}'")
        print(f"  db_path: {config.db_path}")
        print(f"  labeled_examples_folder: {config.labeled_examples_folder}")
        print(f"  models_folder: {config.models_folder}")
        print(f"  predictions_folder: {config.predictions_folder}")
        
        # Check if paths disappeared after first access
        assert config.db_path == Path("db"), "db_path changed after access"
        assert config.labeled_examples_folder == Path("labeled_examples"), "labeled_examples_folder disappeared after db_path access!"
        assert config.models_folder == Path("models"), "models_folder disappeared after db_path access!"
        assert config.predictions_folder == Path("predictions"), "predictions_folder disappeared after db_path access!"
        
        # Continue with the other path accesses
        labeled_dir_value = str(config.labeled_examples_folder or "")
        models_dir_value = str(config.models_folder or "")
        preds_dir_value = str(config.predictions_folder or "")
        
        print(f"\nAfter accessing all paths:")
        print(f"  db_path: {config.db_path}")
        print(f"  labeled_examples_folder: {config.labeled_examples_folder}")
        print(f"  models_folder: {config.models_folder}")
        print(f"  predictions_folder: {config.predictions_folder}")
        
        # Final check
        assert config.db_path == Path("db")
        assert config.labeled_examples_folder == Path("labeled_examples")
        assert config.models_folder == Path("models")
        assert config.predictions_folder == Path("predictions")


    def test_ui_workflow_with_save_load(self, tmp_path: Path):
        """
        Test the UI workflow with save/load cycle which might trigger the bug.
        This simulates: create config -> set defaults -> save -> load -> access paths
        """
        config_file = tmp_path / "test_config.json"
        
        # Step 1: Create fresh config and set defaults (like new config in UI)
        config = AgileConfig()
        config.db_path = Path("db")
        config.labeled_examples_folder = Path("labeled_examples") 
        config.models_folder = Path("models")
        config.predictions_folder = Path("predictions")
        
        print("=== Step 1: After setting defaults ===")
        print(f"db={config.db_path}, labeled={config.labeled_examples_folder}, models={config.models_folder}, preds={config.predictions_folder}")
        
        # Step 2: Save the config (like Save As in UI)
        config.save_as(config_file)
        
        print("=== Step 2: After save_as ===")
        print(f"db={config.db_path}, labeled={config.labeled_examples_folder}, models={config.models_folder}, preds={config.predictions_folder}")
        
        # Verify paths are still there after save
        assert config.db_path == Path("db")
        assert config.labeled_examples_folder == Path("labeled_examples")
        assert config.models_folder == Path("models")
        assert config.predictions_folder == Path("predictions")
        
        # Step 3: Load the config (like loading recent file in UI)
        loaded_config = AgileConfig.load(config_file)
        
        print("=== Step 3: After load ===")
        print(f"db={loaded_config.db_path}, labeled={loaded_config.labeled_examples_folder}, models={loaded_config.models_folder}, preds={loaded_config.predictions_folder}")
        
        # Check paths after load
        assert loaded_config.db_path is not None
        assert loaded_config.labeled_examples_folder is not None
        assert loaded_config.models_folder is not None
        assert loaded_config.predictions_folder is not None
        
        # Step 4: Access paths one by one (like sync_config_to_ui)
        print("=== Step 4: Accessing paths individually ===")
        db_str = str(loaded_config.db_path or "")
        print(f"After db_path access: db={loaded_config.db_path}, labeled={loaded_config.labeled_examples_folder}, models={loaded_config.models_folder}, preds={loaded_config.predictions_folder}")
        
        # This is where the bug might manifest
        assert loaded_config.db_path is not None, "db_path became None"
        assert loaded_config.labeled_examples_folder is not None, "labeled_examples_folder became None after db_path access!"
        assert loaded_config.models_folder is not None, "models_folder became None after db_path access!"
        assert loaded_config.predictions_folder is not None, "predictions_folder became None after db_path access!"


    def test_concurrent_property_access(self, tmp_path: Path):
        """
        Test if the issue happens when multiple properties are accessed rapidly
        or when there are multiple references to the same config.
        """
        config = AgileConfig()
        
        # Set defaults
        config.db_path = Path("db")
        config.labeled_examples_folder = Path("labeled_examples")
        config.models_folder = Path("models")
        config.predictions_folder = Path("predictions")
        
        # Create multiple references (like in UI where vm.config is referenced from different places)
        config_ref1 = config
        config_ref2 = config
        
        print("=== Initial state ===")
        print(f"config: db={config.db_path}, labeled={config.labeled_examples_folder}")
        print(f"ref1: db={config_ref1.db_path}, labeled={config_ref1.labeled_examples_folder}")
        print(f"ref2: db={config_ref2.db_path}, labeled={config_ref2.labeled_examples_folder}")
        
        # Access properties through different references rapidly
        db_val_ref1 = str(config_ref1.db_path or "")
        labeled_val_ref2 = str(config_ref2.labeled_examples_folder or "")
        models_val_config = str(config.models_folder or "")
        
        print("=== After mixed access ===")
        print(f"config: db={config.db_path}, labeled={config.labeled_examples_folder}, models={config.models_folder}, preds={config.predictions_folder}")
        
        # Verify all paths are still there
        assert config.db_path == Path("db")
        assert config.labeled_examples_folder == Path("labeled_examples")
        assert config.models_folder == Path("models")
        assert config.predictions_folder == Path("predictions")
        
        # Also verify through the references
        assert config_ref1.labeled_examples_folder == Path("labeled_examples")
        assert config_ref2.models_folder == Path("models")