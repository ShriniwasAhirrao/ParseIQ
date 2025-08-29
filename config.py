import os
import sys
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class Config:
    """
    Secure configuration settings for the metadata enrichment agent.
    All sensitive data must be provided via environment variables.
    """
    
    # Required Environment Variables
    _REQUIRED_ENV_VARS = {
        'OPENROUTER_API_KEY': 'OpenRouter API key for LLM access'
    }
    
    # Optional Environment Variables with defaults
    _OPTIONAL_ENV_VARS = {
        'SITE_URL': 'http://localhost:3000',
        'SITE_NAME': 'Metadata Enrichment Agent',
        'LOG_LEVEL': 'INFO',
        'MAX_TOKENS': '4000',
        'TEMPERATURE': '0.1',
        'TIMEOUT': '120'
    }
    
    # API Configuration
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct:free"
    
    # Data Quality Thresholds
    ANOMALY_THRESHOLDS = {
        'high_null_rate': 30.0,  # Percentage
        'avg_length_too_short': 2,  # Characters
        'avg_length_too_long': 500,  # Characters
        'z_score_threshold': 3.0,  # Standard deviations
        'iqr_multiplier': 1.5,  # IQR outlier detection
        'min_unique_ratio': 0.1,  # Minimum unique values ratio
        'max_null_percentage': 50.0  # Maximum allowed null percentage
    }
    
    # Statistical Analysis Settings
    STATISTICAL_SETTINGS = {
        'calculate_percentiles': [25, 50, 75, 90, 95, 99],
        'regex_patterns': {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$',
            'url': r'^https?://[^\s]+$',
            'ip_address': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        }
    }
    
    # File Processing Settings
    FILE_SETTINGS = {
        'max_file_size_mb': 100,
        'supported_formats': ['.json', '.csv', '.xml', '.xlsx', '.tsv', '.parquet'],
        'encoding': 'utf-8',
        'csv_delimiter_detection': True,
        'chunk_size': 10000  # For processing large files
    }
    
    # Default directories
    DIRECTORIES = {
        'input': 'input',
        'output': 'output',
        'logs': 'logs',
        'debug_output': 'debug_output',
        'templates': 'step2_llm_enricher',
        'temp': 'temp'
    }
    
    def __init__(self):
        """Initialize configuration with validation"""
        self._validated = False
        self._config_cache = {}
        self.validate_configuration()
    
    @classmethod
    def _get_env_var(cls, var_name: str, default: Optional[str] = None, required: bool = True) -> str:
        """
        Safely get environment variable with validation
        
        Args:
            var_name: Name of environment variable
            default: Default value if not required
            required: Whether the variable is required
            
        Returns:
            Environment variable value
            
        Raises:
            ConfigError: If required variable is missing
        """
        value = os.getenv(var_name, default)
        
        if required and not value:
            raise ConfigError(
                f"Required environment variable '{var_name}' is not set. "
                f"Description: {cls._REQUIRED_ENV_VARS.get(var_name, 'No description available')}"
            )
        
        return value
    
    @classmethod
    def _validate_api_key(cls, api_key: str) -> bool:
        """
        Validate OpenRouter API key format
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid format
        """
        if not api_key:
            return False
        
        # OpenRouter API keys typically start with 'sk-or-v1-'
        if not api_key.startswith('sk-or-v1-'):
            logging.warning("API key format may be incorrect - expected to start with 'sk-or-v1-'")
        
        # Basic length check (OpenRouter keys are typically long)
        if len(api_key) < 20:
            return False
        
        return True
    
    @classmethod
    def _validate_numeric_env_var(cls, var_name: str, value: str, min_val: float = None, max_val: float = None) -> float:
        """
        Validate and convert numeric environment variable
        
        Args:
            var_name: Variable name for error messages
            value: String value to convert
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Converted numeric value
            
        Raises:
            ConfigError: If validation fails
        """
        try:
            num_value = float(value)
        except ValueError:
            raise ConfigError(f"Environment variable '{var_name}' must be a number, got: {value}")
        
        if min_val is not None and num_value < min_val:
            raise ConfigError(f"Environment variable '{var_name}' must be >= {min_val}, got: {num_value}")
        
        if max_val is not None and num_value > max_val:
            raise ConfigError(f"Environment variable '{var_name}' must be <= {max_val}, got: {num_value}")
        
        return num_value
    
    def validate_configuration(self) -> None:
        """
        Validate all configuration settings
        
        Raises:
            ConfigError: If validation fails
        """
        try:
            # Validate required environment variables
            for var_name, description in self._REQUIRED_ENV_VARS.items():
                value = self._get_env_var(var_name, required=True)
                
                # Special validation for API key
                if var_name == 'OPENROUTER_API_KEY':
                    if not self._validate_api_key(value):
                        raise ConfigError(f"Invalid API key format for '{var_name}'")
            
            # Validate numeric environment variables
            max_tokens = self._validate_numeric_env_var('MAX_TOKENS', 
                                                      self._get_env_var('MAX_TOKENS', '4000', False),
                                                      min_val=100, max_val=32000)
            
            temperature = self._validate_numeric_env_var('TEMPERATURE',
                                                       self._get_env_var('TEMPERATURE', '0.1', False),
                                                       min_val=0.0, max_val=2.0)
            
            timeout = self._validate_numeric_env_var('TIMEOUT',
                                                   self._get_env_var('TIMEOUT', '120', False),
                                                   min_val=10, max_val=600)
            
            # Validate log level
            log_level = self._get_env_var('LOG_LEVEL', 'INFO', False).upper()
            valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
            if log_level not in valid_levels:
                raise ConfigError(f"Invalid LOG_LEVEL: {log_level}. Must be one of: {valid_levels}")
            
            # Cache validated values
            self._config_cache = {
                'max_tokens': int(max_tokens),
                'temperature': temperature,
                'timeout': int(timeout),
                'log_level': log_level
            }
            
            self._validated = True
            
        except Exception as e:
            raise ConfigError(f"Configuration validation failed: {str(e)}")
    
    @property
    def openrouter_api_key(self) -> str:
        """Get validated API key"""
        if not self._validated:
            raise ConfigError("Configuration not validated")
        return self._get_env_var('OPENROUTER_API_KEY', required=True)
    
    @property
    def site_url(self) -> str:
        """Get site URL"""
        return self._get_env_var('SITE_URL', self._OPTIONAL_ENV_VARS['SITE_URL'], False)
    
    @property
    def site_name(self) -> str:
        """Get site name"""
        return self._get_env_var('SITE_NAME', self._OPTIONAL_ENV_VARS['SITE_NAME'], False)
    
    @property
    def llm_settings(self) -> Dict[str, Any]:
        """Get LLM settings with validated values"""
        if not self._validated:
            raise ConfigError("Configuration not validated")
        
        return {
            'max_tokens': self._config_cache['max_tokens'],
            'temperature': self._config_cache['temperature'],
            'top_p': 0.9,
            'timeout': self._config_cache['timeout'],
            'retry_attempts': 3,
            'retry_delay': 2
        }
    
    @property
    def logging_settings(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'log_level': self._config_cache.get('log_level', 'INFO'),
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_to_file': True,
            'log_to_console': True,
            'max_log_files': 10,
            'max_log_size_mb': 50
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get complete LLM configuration dictionary
        
        Returns:
            Complete LLM configuration
            
        Raises:
            ConfigError: If configuration is invalid
        """
        if not self._validated:
            raise ConfigError("Configuration not validated")
        
        return {
            'api_key': self.openrouter_api_key,
            'base_url': 'https://openrouter.ai/api/v1',
            'model': self.MODEL_NAME,
            'site_url': self.site_url,
            'site_name': self.site_name,
            **self.llm_settings
        }
    
    def ensure_directories(self) -> List[str]:
        """
        Ensure all required directories exist
        
        Returns:
            List of created/verified directories
        """
        created_dirs = []
        
        for dir_name, dir_path in self.DIRECTORIES.items():
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
            except Exception as e:
                raise ConfigError(f"Failed to create directory '{dir_path}': {str(e)}")
        
        return created_dirs
    
    def get_prompt_template_path(self) -> str:
        """Get the path to the prompt template file"""
        return os.path.join(self.DIRECTORIES['templates'], 'prompt_template.txt')
    
    def print_config_summary(self) -> None:
        """Print a summary of current configuration (without sensitive data)"""
        if not self._validated:
            print("❌ Configuration not validated!")
            return
        
        print("\n📋 Configuration Summary:")
        print("=" * 60)
        print(f"🤖 Model: {self.MODEL_NAME}")
        print(f"🌐 API URL: {self.OPENROUTER_URL}")
        print(f"🏠 Site URL: {self.site_url}")
        print(f"🏷️  Site Name: {self.site_name}")
        print(f"🔢 Max Tokens: {self.llm_settings['max_tokens']}")
        print(f"🌡️  Temperature: {self.llm_settings['temperature']}")
        print(f"⏱️  Timeout: {self.llm_settings['timeout']}s")
        print(f"🔄 Retry Attempts: {self.llm_settings['retry_attempts']}")
        print(f"📊 Log Level: {self.logging_settings['log_level']}")
        
        # Show API key status without revealing the key
        api_key = self.openrouter_api_key
        if api_key:
            masked_key = api_key[:10] + "..." + api_key[-4:]
            print(f"🔑 API Key: ✅ Set ({masked_key})")
        else:
            print("🔑 API Key: ❌ Not Set")
        
        # Show directory status
        print(f"\n📁 Directories:")
        for name, path in self.DIRECTORIES.items():
            exists = "✅" if Path(path).exists() else "❌"
            print(f"   {name}: {path} {exists}")
        
        print("=" * 60)
    
    @classmethod
    def create_sample_env_file(cls) -> None:
        """Create a sample .env file with all required variables"""
        env_content = """# Metadata Enrichment Agent Configuration
# Copy this file to .env and fill in your actual values

# Required: OpenRouter API Key
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Optional: Site Information
SITE_URL=http://localhost:3000
SITE_NAME=Metadata Enrichment Agent

# Optional: LLM Settings
MAX_TOKENS=4000
TEMPERATURE=0.1
TIMEOUT=120

# Optional: Logging
LOG_LEVEL=INFO
"""
        
        env_file_path = Path('.env.sample')
        try:
            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print(f"✅ Sample environment file created: {env_file_path}")
            print("📝 Copy this to '.env' and update with your actual values")
        except Exception as e:
            print(f"❌ Failed to create sample .env file: {e}")


# Singleton instance
_config_instance = None


def get_config() -> Config:
    """
    Get singleton configuration instance
    
    Returns:
        Validated configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def main():
    """Main function for direct execution"""
    print("🔧 Configuration Module - Direct Execution")
    print("=" * 60)
    
    try:
        # Try to create configuration instance
        config = Config()
        
        # Ensure directories exist
        created_dirs = config.ensure_directories()
        print(f"✅ Created/verified {len(created_dirs)} directories")
        
        # Show configuration summary
        config.print_config_summary()
        
        # Test LLM configuration
        llm_config = config.get_llm_config()
        print(f"\n✅ LLM configuration validated successfully")
        print(f"🔧 Config keys: {list(llm_config.keys())}")
        
    except ConfigError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Suggestions:")
        print("1. Set required environment variables:")
        for var, desc in Config._REQUIRED_ENV_VARS.items():
            print(f"   export {var}='your-value-here'  # {desc}")
        print("\n2. Or create a sample .env file:")
        Config.create_sample_env_file()
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()