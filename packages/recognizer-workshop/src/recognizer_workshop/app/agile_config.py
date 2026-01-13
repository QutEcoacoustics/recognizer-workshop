
import dacite
import dataclasses
import json
import os
from pathlib import Path
from typing import Dict, Any, List


import logging
from recognizer_workshop.app.services.db_labels.dataclasses import LabelName

log = logging.getLogger(__name__)

_APP_DATA_FILE = Path.home() / ".recognizer_workshop_data.json"
_MAX_RECENT_FILES = 10


@dataclasses.dataclass
class AgileConfig:
    """
    Configuration class for the Agile Recognizer workshop.
    Manages loading, saving, and validating configuration settings, especially file paths.
    """

    # Note: type hinting is important here so dacite can serialize and deserialize things correctly

    # --- Paths ---
    db_path: Path | None = None
    embeddings_folder: Path | None = None
    labeled_examples_folder: Path | None = None
    models_folder: Path | None = None
    predictions_folder: Path | None = None

    # --- Configuration ---
    annotator_id: str = "default_annotator"
    search_dataset_name: str = 'search_set'
    max_embeddings_count: int = -1
    baw_config: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # list of species labels and optional call types
    target_labels: List[LabelName] = dataclasses.field(default_factory=list) 

    # Path to the current config file after a load or save_as, not part of the JSON serialization.
    config_path: Path | None = dataclasses.field(default=None, init=False, repr=False)

    _APP_DATA_FILE = Path.home() / ".recognizer_workshop_data.json"
    _MAX_RECENT_FILES = 10
    
    # List of fields that should be treated as paths
    _PATH_FIELDS: dataclasses.InitVar[List[str]] = [
        'db_path', 'embeddings_folder', 'labeled_examples_folder', 
        'models_folder', 'predictions_folder'
    ]

    def __post_init__(self, *args, **kwargs):
        """
        Called after initialization, for validation.
        """
        log.info("Running validation...")

        # convert all path strings to Path objects
        for file_name in self._PATH_FIELDS:
            setattr(self, file_name, Path(getattr(self, file_name)) if getattr(self, file_name) is not None else None)

        # Check if essential paths exist
        # Note: We check the PARENT of db_path, as the file may not exist yet.
        if self.db_path and not self.db_path.parent.exists():
            log.warning(f"Database directory does not exist: {self.db_path}. You may need to create it.")
        
        # we automatically create all folders if they don't exist, except for the db one, 
        # because it needs to exist before we start (this might change)
        for field_name in self._PATH_FIELDS:
            if 'db_path' in field_name: continue # already checked
            path_value = getattr(self, field_name)
            if path_value and not path_value.exists():
                log.info(f"Path for '{field_name}' does not exist: {path_value}. Creating it.")
                path_value.mkdir(parents=True, exist_ok=True)


    def reset(self):
        """Resets the config to its default state, clearing the current path."""
        # Create a new default instance and copy its values to this one.
        default_instance = AgileConfig()
        for field in dataclasses.fields(self):
            if field.init: # Don't overwrite fields like 'config_path'
                setattr(self, field.name, getattr(default_instance, field.name))
        self.config_path = None
        log.info("New configuration created.")


    def save(self):
        if not self.config_path:
            raise ValueError(
                "No file path is set for this configuration. "
                "Use 'save_as(path)' to save it for the first time."
            )
        self._to_json(self.config_path)
        log.info(f"Configuration saved to {self.config_path}")


    def save_as(self, path):
        self.config_path = Path(path).resolve()
        self._to_json(self.config_path)
        _add_to_recent_files(self.config_path)
        log.info(f"Configuration saved to new location: {self.config_path}")

    def __str__(self) -> str:
        """Readable string representation of the config."""
        lines = ["Agile Recognizer Config:"]
        for field in dataclasses.fields(self):
            if field.name.startswith('_'): continue # Skip private fields
            
            value = getattr(self, field.name)
            
            # Mask sensitive information like auth tokens if they exist
            if 'auth_token' in str(field.name) and isinstance(value, str) and len(value) > 2:
                value = f"{value[0]}**********{value[-1]}"
            
            lines.append(f"  - {field.name}: {value}")
        return "\n".join(lines)


    @classmethod
    def load(cls, path: str | Path) -> 'AgileConfig':
        """Loads a config from a path and updates the recent files list."""
        load_path = Path(path).resolve()

        if not load_path.exists():
            raise FileNotFoundError(f"Config file not found: {load_path}")

        instance = cls.from_json(load_path)
        instance.config_path = load_path
        _add_to_recent_files(load_path)
        return instance




    @classmethod
    def from_json(cls, config_path: str | Path, relative_to: str | Path | None = None) -> 'AgileConfig':
        """
        Loads configuration from a JSON file and returns a new AgileConfig instance.
        Paths within the JSON are resolved relative to a specified base directory.

        Args:
            config_path: The path to the JSON configuration file.
            relative_to: The base directory for resolving relative paths.
                         - If None (default), paths are relative to the config file's location.
                         - If ".", paths are relative to the current working directory.
                         - If a specific path is given, paths are relative to that directory.
        """
        config_path = Path(config_path).resolve()
        
        if not config_path.exists():
            log.warning(f"Config file not found at {config_path}. Creating a default one.")
            # Create a default instance and save it so it exists for next time
            default_config = cls()
            default_config._to_json(config_path)
            return default_config

        # Determine the base directory for path resolution
        if relative_to is None:
            base_dir = config_path.parent
        else:
            base_dir = Path(relative_to).resolve()

        with open(config_path, 'r') as f:
            data = json.load(f)

        # Resolve paths: if a path in the JSON is not absolute, join it with the base_dir
        for field_name in cls._PATH_FIELDS:
            if field_name in data and data[field_name] is not None:
                path_from_json = Path(data[field_name])
                if not path_from_json.is_absolute():
                    data[field_name] = base_dir / path_from_json

        instance = dacite.from_dict(
            data_class=cls,
            data=data,
            config=dacite.Config(type_hooks={Path: Path}),
        )

        return instance
    

 

    def _to_json(self, config_path: str | Path, relative_to: str | Path | None = None) -> None:
        """
        Saves the current configuration to a JSON file.
        Path objects are converted to strings, relative to a specified base directory.

        Args:
            config_path: The path where the JSON file will be saved.
            relative_to: The base directory for making paths relative.
                         - If None (default), paths are relative to the saved config file's location.
                         - If ".", paths are relative to the current working directory.
                         - If "/", paths are saved as absolute paths.
                         - If a specific path is given, paths are relative to that directory.
        """
        config_path = Path(config_path).resolve()
        
        # Determine the base directory for creating relative paths
        if relative_to is None:
            base_dir = config_path.parent
        elif str(relative_to) == "/":
            # TODO: why can't we just make the base dir "/"
            base_dir = None 
        else:
            base_dir = Path(relative_to).resolve()
            
        config_path.parent.mkdir(parents=True, exist_ok=True)

        
        data_to_save = dataclasses.asdict(self)
        del data_to_save['config_path'] # don't include the config path
      
        for field_name in self._PATH_FIELDS:
            if field_name in data_to_save and data_to_save[field_name] is not None:
                absolute_path = Path(data_to_save[field_name])
                if base_dir is None: 
                    # Save as absolute path
                    saved_path = absolute_path.as_posix()
                else: 
                    # Save as relative path
                    saved_path = os.path.relpath(absolute_path, base_dir)
                
                data_to_save[field_name] = saved_path.replace("\\", "/")

        with open(config_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        log.info(f"Configuration saved to: {config_path}")

