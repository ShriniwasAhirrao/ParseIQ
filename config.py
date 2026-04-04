import os
import sys

class Config:
    """Configuration settings for the metadata enrichment agent"""
    
    # API Configuration - SECURE: Get from environment variables
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Single model used for all LLM calls (free tier)
    MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"

    # Kept for config compatibility (not used in single-model mode)
    FALLBACK_MODELS = []
    
    # Site information for OpenRouter
    SITE_URL = os.getenv('SITE_URL', 'http://localhost:3000')
    SITE_NAME = os.getenv('SITE_NAME', 'Metadata Enrichment Agent')
    
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
            'ip_address': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        }
    }
    
    # LLM Settings - Updated for better compatibility with OpenRouter
    LLM_SETTINGS = {
        'max_tokens': 4096,  # Enough for full JSON response; lower = faster generation
        'temperature': 0.1,
        'top_p': 0.9,
        'timeout': 240,  # Allow more time for large model responses
        'retry_attempts': 3,
        'retry_delay': 2
    }
    
    # File Processing Settings
    FILE_SETTINGS = {
        'max_file_size_mb': 100,
        'supported_formats': ['.json', '.csv', '.xml', '.xlsx'],
        'encoding': 'utf-8',
        'csv_delimiter_detection': True
    }
    
    # Logging Settings
    LOGGING_SETTINGS = {
        'log_level': 'INFO',
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_to_file': True,
        'log_to_console': True,
        'max_log_files': 10,
        'max_log_size_mb': 50
    }
    
    
    
    @classmethod
    def validate_config(cls) -> dict:
        """Validate configuration and return a dict of issues found."""
        issues = {}
        if not cls.OPENROUTER_API_KEY:
            issues['api_key'] = 'OPENROUTER_API_KEY environment variable is not set'
        if not cls.MODEL_NAME:
            issues['model'] = 'MODEL_NAME is not configured'
        if cls.LLM_SETTINGS.get('max_tokens', 0) <= 0:
            issues['max_tokens'] = 'max_tokens must be positive'
        if cls.LLM_SETTINGS.get('timeout', 0) <= 0:
            issues['timeout'] = 'timeout must be positive'
        return issues

    @classmethod
    def get_api_key(cls):
        """Get API key with fallback for development"""
        # First try environment variable
        api_key = cls.OPENROUTER_API_KEY
        
        # If not found, try to get from user input (for development)
        if not api_key:
            print("\n🔑 OpenRouter API key not found in environment variables.")
            print("You can either:")
            print("1. Set the OPENROUTER_API_KEY environment variable (recommended)")
            print("2. Enter it now (not recommended for production)")
            
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "2":
                api_key = input("Enter your OpenRouter API key: ").strip()
                if not api_key.startswith('sk-or-v1-'):
                    print("⚠️  Warning: API key format may be incorrect")
                    confirm = input("Continue anyway? (y/N): ").strip().lower()
                    if confirm != 'y':
                        print("Exiting...")
                        sys.exit(1)
            else:
                print("Please set the OPENROUTER_API_KEY environment variable and restart.")
                print("\nExample:")
                print("export OPENROUTER_API_KEY='sk-or-v1-your-key-here'")
                sys.exit(1)
        
        return api_key
    
    @classmethod
    def get_llm_config(cls):
        """Get complete LLM configuration dictionary"""
        return {
            'api_key': cls.get_api_key(),
            'base_url': 'https://openrouter.ai/api/v1',
            'model': cls.MODEL_NAME,
            'fallback_models': cls.FALLBACK_MODELS,
            'max_tokens': cls.LLM_SETTINGS['max_tokens'],
            'temperature': cls.LLM_SETTINGS['temperature'],
            'top_p': cls.LLM_SETTINGS['top_p'],
            'timeout': cls.LLM_SETTINGS['timeout'],
            'retry_attempts': cls.LLM_SETTINGS.get('retry_attempts', 3),
            'retry_delay': cls.LLM_SETTINGS.get('retry_delay', 2),
            'site_url': cls.SITE_URL,
            'site_name': cls.SITE_NAME
        }
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of current configuration"""
        print("\n📋 Configuration Summary:")
        print("="*50)
        print(f"🤖 Model: {cls.MODEL_NAME}")
        print(f"🌐 API URL: {cls.OPENROUTER_URL}")
        print(f"🏠 Site URL: {cls.SITE_URL}")
        print(f"🏷️  Site Name: {cls.SITE_NAME}")
        print(f"🔢 Max Tokens: {cls.LLM_SETTINGS['max_tokens']}")
        print(f"🌡️  Temperature: {cls.LLM_SETTINGS['temperature']}")
        print(f"⏱️  Timeout: {cls.LLM_SETTINGS['timeout']}s")
        print(f"🔄 Retry Attempts: {cls.LLM_SETTINGS.get('retry_attempts', 3)}")
        print(f"🔑 API Key: {'✅ Set' if cls.OPENROUTER_API_KEY else '❌ Not Set'}")
        
        if cls.OPENROUTER_API_KEY:
            # Show only first 10 and last 4 characters for security
            masked_key = cls.OPENROUTER_API_KEY[:10] + "..." + cls.OPENROUTER_API_KEY[-4:]
            print(f"🔍 API Key Preview: {masked_key}")
        
        print("="*50)
    
    @classmethod
    def create_prompt_template_path(cls):
        """Create the default prompt template path"""
        return os.path.join('step2_llm_enricher', 'prompt_template.txt')
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            'input',
            'output',
            'logs',
            'debug_output',
            'step2_llm_enricher'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return directories



# If run directly, show configuration summary
if __name__ == "__main__":
    print("🔧 Configuration Module - Direct Execution")
    Config.ensure_directories()
    Config.print_config_summary()
    
    # Test configuration
    try:
        llm_config = Config.get_llm_config()
        print("\n✅ LLM configuration created successfully")
        print(f"🔧 Config keys: {list(llm_config.keys())}")
    except Exception as e:
        print(f"\n❌ Error creating LLM configuration: {e}")