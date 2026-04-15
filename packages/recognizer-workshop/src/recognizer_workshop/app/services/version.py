import subprocess
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

def get_git_commit_hash():
    """Get the current git commit hash if available."""
    try:
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        
        # Run git command to get the commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=current_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_git_commit_short():
    """Get the short git commit hash if available."""
    try:
        current_dir = Path(__file__).parent
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=current_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_installed_package_version():
    """Get the installed recognizer-workshop package version if available."""
    try:
        return package_version("recognizer-workshop")
    except PackageNotFoundError:
        return None