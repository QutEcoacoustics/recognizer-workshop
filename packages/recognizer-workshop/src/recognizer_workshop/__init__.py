import importlib.metadata

try:
    # This reads the metadata baked into the package when it was installed
    __version__ = importlib.metadata.version("recognizer-workshop")
except importlib.metadata.PackageNotFoundError:
    # Fallback for when you are testing the code locally without installing it first
    __version__ = "dev"