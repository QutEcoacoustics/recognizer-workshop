from __future__ import annotations

from importlib.resources import files, as_file
from ml_collections.config_dict import ConfigDict
from perch_hoplite.agile.convert_legacy import load_embedding_config


def load_baw_embedding_config(analysis_name: str) -> ConfigDict:
    """
    Load packaged JSON for an analysis_name, e.g. 'perch_embeddings_4'
    Expected location (inside the package):
        recognizer_workshop/resources/embedding_configs/<analysis_name>.json
    """
    pkg = "recognizer_workshop.resources.embedding_configs"
    fname = f"{analysis_name}.json"

    try:
        res = files(pkg).joinpath(fname)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Package data module '{pkg}' not found. "
            "Ensure it’s a real package (has __init__.py) and is included in the build."
        ) from e

    if not res.is_file():
        raise FileNotFoundError(f"Bundled config not found: {pkg}/{fname}")

    # Materialize to a real filesystem path and delegate to perch_hoplite loader
    with as_file(res) as cfg_path:
        return load_embedding_config(cfg_path.parent, filename=cfg_path.name)
