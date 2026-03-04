
import json
from pathlib import Path

_APP_DATA_FILE = Path.home() / ".recognizer_workshop_data.json"
_MAX_RECENT_FILES = 10

def get_recent_files(num: int = _MAX_RECENT_FILES) -> list[str]:
    data = read_app_data()
    return data.get("recent_files", [])[:num]

def add_to_recent_files(path: Path):
    """Adds a file path to the top of the recent files list."""
    data = read_app_data()
    recent_files = data.get("recent_files", [])
    str_path = str(path)
    if str_path in recent_files:
        recent_files.remove(str_path)
    recent_files.insert(0, str_path)
    data["recent_files"] = recent_files[:_MAX_RECENT_FILES]
    write_app_data(data)


def read_app_data() -> dict:
    """Reads the central app data file."""
    if get_app_data_file().exists():
        with open(get_app_data_file(), 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def write_app_data(data: dict):
    """Writes to the central app data file."""
    with open(get_app_data_file(), 'w') as f:
        json.dump(data, f, indent=4)



def get_app_data_file() -> Path:
    """Returns the path to the central app data file."""
    return _APP_DATA_FILE
