# config.py
from imports import *
from dotenv import load_dotenv 

# Load environment variables from a .env file for local development
load_dotenv()


# AI model and tokenizer, load on demand
def load_ai_components():
    """
    Load the tokenizer and model for use in NLP tasks.
    Returns:
        tokenizer (AutoTokenizer): The loaded tokenizer.
        model (AutoModelForCausalLM): The loaded model.
    """
    try:
        model_id = os.getenv('MODEL_ID', 'vikhyatk/moondream2')
        revision = os.getenv('MODEL_REVISION', '2024-04-02')
        logging.info(f"Loading model {model_id} with revision {revision}")

        # Ensure truncation is appropriately set to handle long texts
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, model_max_length=2048, truncation=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)

        logging.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Failed to load the model components: {e}")
        raise

CACHE_FILE_PATH = 'config_cache.json'
logging.basicConfig(filename='logs/config.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Application General Configuration
APP_NAME = "SEO Analyzer"
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't')

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'seo_analyzer')
}

# Security
SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')

# Path Configurations
LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'logs/app.log')

# SMTP Settings for Email Notifications
SMTP_CONFIG = {
    'server': os.getenv('SMTP_SERVER', 'smtp.example.com'),
    'port': int(os.getenv('SMTP_PORT', 587)),
    'username': os.getenv('SMTP_USERNAME'),
    'password': os.getenv('SMTP_PASSWORD'),
    'use_tls': os.getenv('SMTP_USE_TLS', 'True').lower() in ('true', '1', 't')
}


def get_database_url():
    """Construct the database connection URL."""
    return f"{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

# Function to ensure critical configurations are set
def check_configurations():
    required_configs = [SECRET_KEY, DATABASE_CONFIG['password']]
    if not all(required_configs):
        raise ValueError("One or more required configurations are missing.")
def load_config_from_file(file_path):
    """Loads configuration from a JSON or YAML file."""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format")

def load_config_from_database(db_conn, table_name='configuration'):
    """Loads configuration from a database table."""
    cursor = db_conn.cursor()
    cursor.execute(f"SELECT key, value FROM {table_name}")
    config_dict = {row['0']: row['1'] for row in cursor.fetchall()}
    cursor.close()
    return config_dict

def cache_config(config_dict):
    """Stores the configuration dictionary in a cache file."""
    with open(CACHE_FILE_PATH, 'w') as f:
        json.dump(config_dict, f)

def get_cached_config():
    """Retrieves the cached configuration if available."""
    try:
        with open(CACHE_FILE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def update_config(config_dict, key, value):
    """Updates a specific configuration value in the provided dictionary."""
    config_dict[key] = value
    cache_config(config_dict)  # Update the cache with new config

def validate_config(config_dict):
    """Checks if all required configurations are present and valid."""
    required_keys = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
    missing_keys = [key for key in required_keys if key not in config_dict]
    if missing_keys:
        raise ValueError(f"Missing configuration(s): {', '.join(missing_keys)}")

def load_config():
    """Load the configuration based on source specified in environment or command-line."""
    source = os.getenv('CONFIG_SOURCE', 'file')  # Default to file if not specified
    file_path = os.getenv('CONFIG_FILE_PATH', 'settings.json')
    
    if source == 'file':
        config = load_config_from_file(file_path)
    elif source == 'database':
        db_conn = sqlite3.connect(os.getenv('DB_PATH'))
        config = load_config_from_database(db_conn)
        db_conn.close()
    else:
        raise ValueError("Invalid configuration source")
    
    cache_config(config)  # Cache the loaded configuration
    return config

def check_configurations():
    required_configs = [SECRET_KEY, DATABASE_CONFIG['password']]
    if not all(required_configs):
        raise ValueError("One or more required configurations are missing.")

# Example usage
config = get_cached_config()
if not config:
    config = load_config()
    print("Loaded fresh configuration.")
else:
    print("Loaded configuration from cache.")

# Update and validate configurations as needed
update_config(config, 'NEW_SETTING', 'value')
validate_config(config)

# Check configurations at module load
check_configurations()
