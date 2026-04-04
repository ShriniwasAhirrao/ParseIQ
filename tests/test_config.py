import os
import pytest
from unittest.mock import patch
import config

class TestConfig:
    def test_validate_config_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            config.Config.OPENROUTER_API_KEY = None
            issues = config.Config.validate_config()
            # Should report at least the missing API key issue
            assert 'api_key' in issues

    def test_validate_config_with_api_key(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-or-v1-1234567890'}):
            config.Config.OPENROUTER_API_KEY = 'sk-or-v1-1234567890'
            issues = config.Config.validate_config()
            # No api_key issue when key is set
            assert 'api_key' not in issues

    def test_get_api_key_env(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-or-v1-abcdef'}):
            config.Config.OPENROUTER_API_KEY = 'sk-or-v1-abcdef'
            api_key = config.Config.get_api_key()
            assert api_key == 'sk-or-v1-abcdef'

    def test_get_llm_config(self):
        with patch.object(config.Config, 'get_api_key', return_value='dummy_key'):
            llm_config = config.Config.get_llm_config()
            assert llm_config['api_key'] == 'dummy_key'
            assert 'model' in llm_config
            assert 'base_url' in llm_config

    def test_ensure_directories(self, tmp_path):
        dirs = ['input', 'output', 'logs', 'debug_output', 'step2_llm_enricher']
        import os as real_os
        original_makedirs = real_os.makedirs
        def makedirs_side_effect(path, exist_ok=True):
            original_makedirs(tmp_path / path, exist_ok=exist_ok)
        with patch('os.makedirs', side_effect=makedirs_side_effect) as mock_makedirs:
            created_dirs = config.Config.ensure_directories()
            for d in dirs:
                assert d in created_dirs

    def test_print_config_summary(self, capsys):
        config.Config.OPENROUTER_API_KEY = 'sk-or-v1-abcdef'
        config.Config.print_config_summary()
        captured = capsys.readouterr()
        assert "Model:" in captured.out
        assert "API Key: ✅ Set" in captured.out
