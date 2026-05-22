import importlib.metadata

try:
    __version__ = importlib.metadata.version("recognizer-workshop")
except importlib.metadata.PackageNotFoundError:
    # Fallback if not installed as a package
    __version__ = "unknown"
