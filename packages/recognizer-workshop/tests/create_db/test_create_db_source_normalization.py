import pytest
from pathlib import Path
import numpy as np


from recognizer_workshop.app.create_database.create_database import complete_remote_source_dict, normalize_source

# --- FIXTURES ---

@pytest.fixture
def local_files(tmp_path):
    """Creates dummy local parquet files for testing."""
    f1 = tmp_path / "data1.parquet"
    f2 = tmp_path / "data2.parquet"
    f1.touch()
    f2.touch()
    return f1, f2

@pytest.fixture
def default_domain():
    return "https://api.ecosounds.org"

# --- TESTS: complete_remote_source_dict ---

class TestCompleteRemoteSourceDict:
    
    def test_completes_from_id_only(self, default_domain):
        """Should use default domain and generate URL from ID."""
        item = {'id': 123}
        result = complete_remote_source_dict(item)
        assert result['id'] == 123
        assert result['domain'] == default_domain
        assert result['source_url'] == f"{default_domain}/audio_recordings/123/original"

    def test_completes_from_url_only(self):
        """Should extract ID and Domain from URL."""
        url = "https://custom.org/audio_recordings/999/original"
        item = {'source_url': url}
        result = complete_remote_source_dict(item)
        assert result['id'] == 999
        assert result['domain'] == "https://custom.org"
        assert result['source_url'] == url

    def test_completes_from_id_and_domain(self):
        """Should generate URL from provided ID and Domain."""
        item = {'id': 55, 'domain': 'https://other.com'}
        result = complete_remote_source_dict(item)
        assert result['id'] == 55
        assert result['domain'] == "https://other.com"
        assert result['source_url'] == "https://other.com/audio_recordings/55/original"

    def test_raises_on_conflicting_ids(self):
        """Input ID does not match ID extracted from URL."""
        item = {
            'id': 123,
            'source_url': "https://api.ecosounds.org/audio_recordings/456/original"
        }
        with pytest.raises(ValueError, match="Conflicting ID"):
            complete_remote_source_dict(item)

    def test_raises_on_conflicting_domains(self):
        """Input Domain does not match Domain extracted from URL."""
        item = {
            'domain': "https://site-a.com",
            'source_url': "https://site-b.com/audio_recordings/123/original"
        }
        with pytest.raises(ValueError, match="Conflicting Domain"):
            complete_remote_source_dict(item)
            
    def test_raises_on_conflicting_url_structure(self):
        """URL provided does not match what would be generated."""
        # e.g. User provides a non-standard URL path
        item = {
            'id': 123,
            'source_url': "https://api.ecosounds.org/wrong_path/123/original"
        }
        with pytest.raises(ValueError, match="Conflicting source_url"):
            complete_remote_source_dict(item)

    def test_raises_on_missing_info(self):
        """No ID and no URL to extract from."""
        item = {'domain': 'https://google.com'}
        with pytest.raises(ValueError, match="Cannot resolve ID"):
            complete_remote_source_dict(item)


# --- TESTS: normalize_source ---

class TestNormalizeSource:

    # 1. Local Files
    
    def test_local_single_path_as_string(self, local_files):
        path_str = str(local_files[0])
        result, type_ = normalize_source(path_str)
        assert type_ == "local"
        assert isinstance(result, Path)
        assert result == Path(path_str)

    def test_local_single_path_obj(self, local_files):
        path_obj = local_files[0]
        result, type_ = normalize_source(path_obj)
        assert type_ == "local"
        assert isinstance(result, Path)
        assert result == path_obj

    def test_local_list_of_mixed_strings_and_paths(self, local_files):
        """Test mixed list of Path objects and strings."""
        source = [str(local_files[0]), local_files[1]]
        result, type_ = normalize_source(source)
        assert type_ == "local"
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(x, Path) for x in result)
        assert result[0] == local_files[0]
        assert result[1] == local_files[1]

    def test_local_raises_if_not_exists(self, tmp_path):
        bad_path = tmp_path / "ghost.parquet"
        with pytest.raises(FileNotFoundError):
            normalize_source(bad_path)

    # 2. Remote - Simple Integers

    def test_remote_single_arid(self):
        source = 123
        result, type_ = normalize_source(source)
        assert type_ == "remote"
        assert len(result) == 1
        assert result[0]['id'] == 123
        assert result[0]['domain'] == "https://api.ecosounds.org"

    def test_remote_list_arids(self):
        source = [101, 102]
        result, type_ = normalize_source(source)
        assert type_ == "remote"
        assert len(result) == 2
        assert result[0]['id'] == 101
        assert result[1]['id'] == 102

    def test_remote_numpy_int(self):
        source = np.int64(555)
        result, type_ = normalize_source(source)
        assert type_ == "remote"
        assert result[0]['id'] == 555

    # 3. Remote - URLs

    def test_remote_single_url(self):
        """Ensure single URL input preserves its domain."""
        url = "https://custom.server.org/audio_recordings/777/original"
        result, type_ = normalize_source(url)
        assert type_ == "remote"
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['id'] == 777
        assert result[0]['domain'] == "https://custom.server.org"
        assert result[0]['source_url'] == url

    def test_remote_list_urls(self):
        urls = [
            "https://api.ecosounds.org/audio_recordings/1/original",
            "https://other.server.com/audio_recordings/2/original"
        ]
        result, type_ = normalize_source(urls)
        assert type_ == "remote"
        assert len(result) == 2
        
        assert result[0]['id'] == 1
        assert result[0]['domain'] == "https://api.ecosounds.org"
        
        assert result[1]['id'] == 2
        assert result[1]['domain'] == "https://other.server.com"

    # 4. Remote - List of Dicts (Combinations)

    def test_remote_list_dicts_combinations(self):
        """Test various valid dictionary combinations."""
        source = [
            {'id': 500}, 
            {'source_url': "https://d.com/audio_recordings/600/original"},
            {'id': 700, 'domain': "https://x.com"}
        ]
        result, type_ = normalize_source(source)
        assert type_ == "remote"
        
        # Check first item (default domain)
        assert result[0]['id'] == 500
        assert result[0]['domain'] == "https://api.ecosounds.org"
        
        # Check second item (extracted ID)
        assert result[1]['id'] == 600
        assert result[1]['domain'] == "https://d.com"
        
        # Check third item (explicit domain)
        assert result[2]['id'] == 700
        assert result[2]['domain'] == "https://x.com"
        assert result[2]['source_url'] == "https://x.com/audio_recordings/700/original"

    def test_remote_list_dicts_missing_info(self):
        """Test a list where an item is missing critical info."""
        source = [
            {'id': 1}, 
            {'domain': 'https://broken.com'} # Missing ID and URL
        ]
        with pytest.raises(ValueError, match="Cannot resolve ID"):
            normalize_source(source)

    def test_remote_list_dicts_conflicting_info(self):
        """Test a list where an item has conflicting info."""
        source = [
            {'id': 1},
            {'id': 2, 'source_url': 'https://api.ecosounds.org/audio_recordings/999/original'} # Conflict
        ]
        with pytest.raises(ValueError, match="Conflicting ID"):
            normalize_source(source)

    def test_preserves_extra_keys(self):
        """Ensure that keys not involved in normalization (like offsets) are preserved."""
        item = {
            'id': 123, 
            'offsets': [5.0, 10.0], 
            'duration_seconds': 60
        }
        result = complete_remote_source_dict(item)
        
        # Check normalization happened
        assert result['source_url'] == "https://api.ecosounds.org/audio_recordings/123/original"
        
        # Check preservation happened (This would have failed before!)
        assert 'offsets' in result
        assert result['offsets'] == [5.0, 10.0]
        assert result['duration_seconds'] == 60