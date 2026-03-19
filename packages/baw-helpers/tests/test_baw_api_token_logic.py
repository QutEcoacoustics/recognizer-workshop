"""Unit tests for token validation and authentication logic."""
import pytest
from unittest.mock import patch
from baw_helpers.baw_api import baw_api, AuthenticationError


# ==================== Test-Specific Fixtures ====================

@pytest.fixture
def mock_test_token_valid():
    """Mock check_auth to always return ok=True."""
    with patch.object(baw_api, 'check_auth', return_value={'ok': True}) as mock:
        yield mock


@pytest.fixture
def mock_test_token_invalid():
    """Mock check_auth to always return ok=False."""
    with patch.object(baw_api, 'check_auth', return_value={'ok': False}) as mock, \
         patch.object(baw_api, 'auth_state', 'invalid_token', create=True), \
         patch.object(baw_api, 'auth_message', 'Mock invalid token', create=True):
        yield mock


# ==================== Tests ====================

class TestTokenPriorityChain:
    """Test #1, #7, #11, #13: Token source priority."""
    
    def test_priority_1_explicit_valid(self, mock_env_no_token, mock_test_token_valid):
        """#1: Explicit valid token is used (even if env has tokens)."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
             patch('baw_helpers.baw_api.os.getenv', return_value="env_var_token"), \
             patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': 'dotenv_token'}):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token="explicit_token", allow_prompt=False)
            
            assert result is True
            assert api.token == "explicit_token"
    
    def test_priority_7_env_var_valid(self, mock_env_no_token, mock_test_token_valid):
        """#7: Env var token used when no explicit token (even if .env has token)."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
             patch('baw_helpers.baw_api.os.getenv', return_value="env_var_token"), \
             patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': 'dotenv_token'}):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token=None, allow_prompt=False)
            
            assert result is True
            assert api.token == "env_var_token"
    
    def test_priority_11_dotenv_valid(self, mock_env_no_token, mock_test_token_valid):
        """#11: .env file token used when no explicit token or env var."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
             patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': 'dotenv_token'}):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token=None, allow_prompt=False)
            
            assert result is True
            assert api.token == "dotenv_token"
    
    def test_priority_13_no_token_anywhere(self, mock_env_no_token, mock_no_dotenv):
        """#13: No token anywhere results in anonymous mode."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        result = api.update_token(token=None, allow_prompt=False, anonymous=None)
        
        assert result is True
        assert api.token is None


class TestAnonymousModeNonInteractive:
    """Test #17, #18, #19: Anonymous mode with allow_prompt=False."""
    
    def test_17_valid_explicit_authenticated(self, mock_env_no_token, mock_no_dotenv, mock_test_token_valid):
        """#17: Valid explicit token succeeds."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        result = api.update_token(token="valid_token", allow_prompt=False, anonymous=False)
        
        assert result is True
        assert api.token == "valid_token"
    
    def test_18_forced_anonymous_ignores_token(self, mock_env_no_token, mock_no_dotenv, mock_test_token_invalid):
        """#18: Invalid explicit token proceeds to anonymous mode."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        result = api.update_token(token="invalid_token", allow_prompt=False, anonymous=True)
        
        assert result is True
        assert api.token is None
    
    def test_19_no_token_anonymous_ok(self, mock_env_no_token, mock_no_dotenv):
        """#19: No token proceeds to anonymous mode."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        result = api.update_token(token=None, allow_prompt=False, anonymous=True)
        
        assert result is True
        assert api.token is None


class TestAuthRequiredNonInteractive:
    """Test #22, #23, #24: Auth required with allow_prompt=False."""
    
    def test_22_valid_explicit_authenticated(self, mock_env_no_token, mock_no_dotenv, mock_test_token_valid):
        """#22: Valid explicit token succeeds when auth required."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        result = api.update_token(token="valid_token", allow_prompt=False, anonymous=False)
        
        assert result is True
        assert api.token == "valid_token"
    
    def test_23_invalid_explicit_auth_fails(self, mock_env_no_token, mock_no_dotenv, mock_test_token_invalid):
        """#23: Invalid explicit token raises exception when auth required."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        with pytest.raises(AuthenticationError):
            api.update_token(token="invalid_token", allow_prompt=False, anonymous=False)
    
    def test_24_no_token_auth_fails(self, mock_env_no_token, mock_no_dotenv):
        """#24: No token raises exception when auth required."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        with pytest.raises(AuthenticationError):
            api.update_token(token=None, allow_prompt=False, anonymous=False)


class TestInteractiveAuthRequired:
    """Test #32, #34, #36: Interactive prompting with require_auth=True."""
    
    def test_32_valid_explicit_no_prompt(self, mock_env_no_token, mock_no_dotenv, mock_test_token_valid):
        """#32: Valid explicit token, no prompting needed."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        result = api.update_token(token="valid_token", allow_prompt=True, anonymous=False)
        
        assert result is True
        assert api.token == "valid_token"
    

    
    def test_36_no_token_user_quits(self, mock_env_no_token):
        """#36: No token, user quits prompt -> raises exception."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
             patch('baw_helpers.baw_api.getpass.getpass', return_value="quit"):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            # FIX: Expect Error, not False
            with pytest.raises(AuthenticationError):
                api.update_token(token=None, allow_prompt=True, anonymous=False)


class TestEdgeCasesEmptyStrings:
    """Test #14, #15, #37: Empty string handling."""
    
    def test_14_empty_explicit_falls_back_to_env_var(self, mock_env_no_token, mock_test_token_valid):
        """#14: Empty string explicit token falls back to env var."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value=None), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
             patch('baw_helpers.baw_api.os.getenv', return_value="env_var_token"):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token="", allow_prompt=False)
            
            assert result is True
            assert api.token == "env_var_token"
    
    def test_15_empty_explicit_falls_back_to_dotenv(self, mock_env_no_token, mock_test_token_valid):
        """#15: Empty string explicit token falls back to .env file."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
             patch('baw_helpers.baw_api.os.getenv', return_value=None), \
             patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': 'dotenv_token'}):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token="", allow_prompt=False)
            
            assert result is True
            assert api.token == "dotenv_token"
    
    def test_37_empty_string_everywhere_anonymous(self, mock_env_no_token):
        """#37: Empty strings everywhere treated as no token."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
             patch('baw_helpers.baw_api.os.getenv', return_value=""), \
             patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': ''}):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token="", allow_prompt=False, anonymous=None)
            
            assert result is True
            assert api.token is None


class TestInteractivePromptingSuccess:
    """Interactive prompting with successful token entry (require_auth=True only)."""
    
    def test_no_token_user_enters_valid_require_auth(self, mock_env_no_token, mock_test_token_valid):
        """No token with anonymous=False, user enters valid token via prompt."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
             patch('baw_helpers.baw_api.getpass.getpass', return_value="user_token"), \
             patch('baw_helpers.baw_api.input', return_value="n"):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token=None, allow_prompt=True, anonymous=False)
            
            assert result is True
            assert api.token == "user_token"

    def test_user_chooses_anonymous_via_prompt(self, mock_env_no_token):
        """User explicitly enters 'a' in the prompt to go anonymous."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
            patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
            patch('baw_helpers.baw_api.getpass.getpass', return_value="a"):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            # Even though anonymous=False (Strict), the user override 'a' should succeed
            result = api.update_token(token=None, allow_prompt=True, anonymous=False)
            
            assert result is True
            assert api.token is None
    

class TestPromptRetryLogic:
    """Test #41: Retry logic in interactive prompt."""
    
    def test_41_invalid_then_valid_token(self, mock_env_no_token):
        """#41: User enters invalid token, then valid token."""
        test_token_results = [False, True]  # First attempt fails, second succeeds
        
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
             patch('baw_helpers.baw_api.getpass.getpass', side_effect=["bad_token", "good_token"]), \
             patch('baw_helpers.baw_api.input', return_value="n"), \
             patch.object(baw_api, 'test_token', side_effect=test_token_results):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token=None, allow_prompt=True, anonymous=False)
            
            assert result is True
            assert api.token == "good_token"


class TestSaveToEnvPrompting:
    """Test #42, #43: Save to .env prompting."""
    
    def test_42_user_saves_to_env(self, mock_env_no_token, mock_test_token_valid):
        """#42: User chooses to save token to .env."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
             patch('baw_helpers.baw_api.getpass.getpass', return_value="user_token"), \
             patch('baw_helpers.baw_api.input', return_value="y"), \
             patch('baw_helpers.baw_api.set_key') as mock_set_key:
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token=None, allow_prompt=True, anonymous=False)
            
            assert result is True
            assert api.token == "user_token"
            mock_set_key.assert_called_once_with("/fake/.env", 'BAW_AUTH_TOKEN', "user_token")
    
    def test_43_user_does_not_save_to_env(self, mock_env_no_token, mock_test_token_valid):
        """#43: User chooses NOT to save token to .env."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=False), \
             patch('baw_helpers.baw_api.getpass.getpass', return_value="user_token"), \
             patch('baw_helpers.baw_api.input', return_value="n"), \
             patch('baw_helpers.baw_api.set_key') as mock_set_key:
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            result = api.update_token(token=None, allow_prompt=True, anonymous=False)
            
            assert result is True
            assert api.token == "user_token"
            mock_set_key.assert_not_called()


class TestInvalidTokenNoFallback:
    """Test #4, #5, #6: Invalid explicit token doesn't fall back to env/dotenv."""
    
    def test_4_invalid_explicit_ignores_valid_env(self, mock_env_no_token, mock_test_token_invalid):
        """#4: Invalid explicit token raises exception (doesn't fall back to valid env var)."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
             patch('baw_helpers.baw_api.os.getenv', return_value="valid_env_token"), \
             patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': 'valid_dotenv_token'}):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            with pytest.raises(AuthenticationError):
                api.update_token(token="invalid_explicit", allow_prompt=False)
    
    def test_5_invalid_explicit_all_invalid(self, mock_env_no_token, mock_test_token_invalid):
        """#5: Invalid explicit token raises exception (all sources invalid)."""
        with patch('baw_helpers.baw_api.find_dotenv', return_value="/fake/.env"), \
             patch('baw_helpers.baw_api.os.path.exists', return_value=True), \
             patch('baw_helpers.baw_api.os.getenv', return_value="invalid_env"), \
             patch('baw_helpers.baw_api.dotenv_values', return_value={'BAW_AUTH_TOKEN': 'invalid_dotenv'}):
            
            api = baw_api.__new__(baw_api)
            api.base_url = "https://test.example.com"
            api.token = None
            
            with pytest.raises(AuthenticationError):
                api.update_token(token="invalid_explicit", allow_prompt=False)
    
    def test_6_invalid_explicit_no_env(self, mock_env_no_token, mock_no_dotenv, mock_test_token_invalid):
        """#6: Invalid explicit token raises exception (no env tokens available)."""
        api = baw_api.__new__(baw_api)
        api.base_url = "https://test.example.com"
        api.token = None
        
        with pytest.raises(AuthenticationError):
            api.update_token(token="invalid_explicit", allow_prompt=False)