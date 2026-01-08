import pytest
from pathlib import Path

from baw_helpers.baw_utils import extract_arid_from_filename


@pytest.mark.parametrize("filename, expected_id", [
    # --- Standard Cases ---
    ("site_123.wav", 123),
    ("recording_45678.parquet", 45678),
    ("some_random_stuff_999.mp3", 999),
    
    # --- Start of String (No Underscore) ---
    ("1234.wav", 1234),
    ("555.mp3", 555),
    
    # --- Complex Extensions (Double Dots) ---
    ("1234.wav.mp3", 1234),
    ("archive_777.tar.gz", 777),
    ("data_100.part1.parquet", 100),
    
    # --- Multiple Numbers (Greedy Match Check) ---
    # Should always match the LAST number before the dot
    ("site_111_audio_222.wav", 222), 
    ("123_456.wav", 456),
    ("some_random_111.stuff_1234.mp3.wav", 1234), 
    
    # --- Dots in the filename ---
    ("site.v1_101.wav", 101),
    ("my.data.file_202.parquet", 202),
])
def test_extract_arid_success(filename, expected_id):
    """Test that valid filenames correctly return the ID."""
    assert extract_arid_from_filename(filename) == expected_id

def test_extract_arid_handles_path_objects():
    """Test that passing a pathlib.Path object works."""
    p = Path("/tmp/data/downloads/site_123.wav")
    assert extract_arid_from_filename(p) == 123


@pytest.mark.parametrize("filename", [
    "just_text.wav",       # No digits
    "no_extension_123",    # No dot/extension
    "123",                 # Just a number, no extension
    "site_123_final",      # Number exists but not followed by dot
    "",                    # Empty string
])
def test_extract_arid_returns_none(filename):
    """Test cases where no ID should be found."""
    assert extract_arid_from_filename(filename) is None