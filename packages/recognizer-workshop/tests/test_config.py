import json
from pathlib import Path

# Import the classes from your file
# Make sure this import path is correct for your project structure
from recognizer_workshop.app.agile_engine import AgileConfig, LabelName

# Note: The class no longer needs to inherit from unittest.TestCase
class TestAgileConfig:
    """Unit tests for the AgileConfig class using pytest."""


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


    def test_save_and_load_roundtrip_relative(self, tmp_path: Path):
        """Test the full save/load cycle with default relative paths."""
        config_path = tmp_path / "config.json"
        db_path_abs = tmp_path / "data" / "database.sqlite"
        
        # 1. Create the config with some data
        config = AgileConfig(
            db_path=db_path_abs,
            annotator_id="tester-01",
            target_labels=[LabelName("Koala", "Bellow")]
        )
        
        # 2. Save it to JSON
        config.to_json(config_path)
        assert config_path.exists()

        # 3. Manually check the saved JSON for relative paths
        with open(config_path, 'r') as f:
            data = json.load(f)
        assert data['db_path'] == 'data/database.sqlite'
        assert data['annotator_id'] == 'tester-01'

        # 4. Load the config back using from_json
        loaded_config = AgileConfig.from_json(config_path)

        # 5. Assert that the loaded config is identical and paths are absolute
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
        config.to_json(config_path, relative_to="/")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
            
        # Check that the saved path is the string representation of the absolute path
        assert data['db_path'] == str(db_path_abs.as_posix())


    def test_from_json_non_existent_creates_default(self, tmp_path: Path):
        """Test that from_json creates a default config file if none exists."""
        config_path = tmp_path / "new_config.json"
        
        # Ensure it doesn't exist
        assert not config_path.exists()
        
        # Call from_json, which should create and save a default config
        config = AgileConfig.from_json(config_path)
        
        # Assert that the file now exists
        assert config_path.exists()
        
        # Assert that the returned object has default values
        assert config.db_path is None
        assert config.annotator_id == "default_annotator"

        # Check the contents of the created file
        with open(config_path, 'r') as f:
            data = json.load(f)
        assert data['db_path'] is None
        assert data['annotator_id'] == "default_annotator"
        

    def test_update_config_and_save(self, tmp_path: Path):
        """Test that updating an instance and saving it persists changes."""
        config_path = tmp_path / "updated_config.json"

        # 1. Create a default config and save it
        config = AgileConfig()
        config.to_json(config_path)

        # 2. Update the attributes
        config.annotator_id = "final_annotator"
        config.db_path = tmp_path / "production.db"
        config.target_labels.append(LabelName("Frog"))

        # 3. Save the modified config over the old one
        config.to_json(config_path)

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

        # 2. Save the config to a JSON file
        original_config.to_json(config_path)
        assert config_path.exists()

        # 3. Load the config back from the file
        loaded_config = AgileConfig.from_json(config_path)

        # 4. Assert that the loaded object is identical to the original
        # Dataclasses automatically provide an __eq__ method that compares all fields.
        # This single assertion is powerful and checks everything.
        assert original_config == loaded_config

        # (Optional but good practice) Add a specific check for a restored type
        # to be certain that dacite's type hooks worked as expected.
        assert isinstance(loaded_config.db_path, Path)
        assert loaded_config.db_path == db_path