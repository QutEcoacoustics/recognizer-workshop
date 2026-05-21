import subprocess
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

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


def get_version_status_string(include_git_hash: bool = False) -> str:
    """Return a UI-ready version status string.

    Format:
    - "version: <value> (<short_hash>)" when both are available and include_git_hash is True
    - "version: <value>" when git hash is unavailable or include_git_hash is False
    """
    installed_version = get_installed_package_version() or "unknown"
    commit_hash = get_git_commit_short()

    if include_git_hash and commit_hash:
        return f"version: {installed_version} ({commit_hash})"
    return f"version: {installed_version}"