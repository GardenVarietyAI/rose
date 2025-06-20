import os

# Environment prefix
ENV_PREFIX = "ROSE_SERVER_"

# Multiprocessing settings
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["LOKY_CONTEXT_STRATEGY"] = "spawn"

# Service settings
SERVICE_NAME = "ROSE"
SERVICE_VERSION = "0.1.0"
HOST = os.getenv(f"{ENV_PREFIX}HOST", "127.0.0.1")
PORT = int(os.getenv(f"{ENV_PREFIX}PORT", "8004"))
RELOAD = os.getenv(f"{ENV_PREFIX}RELOAD", "true").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv(f"{ENV_PREFIX}LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
WEBHOOK_URL = os.getenv(f"{ENV_PREFIX}WEBHOOK_URL", "http://localhost:8004/v1/webhooks/jobs")

# Data directories
DATA_DIR = os.getenv(f"{ENV_PREFIX}DATA_DIR", "./data")
CHROMA_PERSIST_DIR = os.getenv(f"{ENV_PREFIX}CHROMA_PERSIST_DIR", "./data/chroma")
MODEL_OFFLOAD_DIR = os.getenv(f"{ENV_PREFIX}MODEL_OFFLOAD_DIR", "./data/offload")

# ChromaDB settings
CHROMA_HOST = os.getenv(f"{ENV_PREFIX}CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv(f"{ENV_PREFIX}CHROMA_PORT", "8003"))

# Health check
HEALTH_CHECK_ENABLED = os.getenv(f"{ENV_PREFIX}HEALTH_CHECK_ENABLED", "true").lower() in ("true", "1", "yes")
HEALTH_CHECK_WINDOW_MINUTES = int(os.getenv(f"{ENV_PREFIX}HEALTH_CHECK_WINDOW_MINUTES", "5"))

# Fine-tuning settings
FINE_TUNING_DEFAULT_EPOCHS = int(os.getenv(f"{ENV_PREFIX}FINE_TUNING_DEFAULT_EPOCHS", "3"))
FINE_TUNING_DEFAULT_MAX_LENGTH = int(os.getenv(f"{ENV_PREFIX}FINE_TUNING_DEFAULT_MAX_LENGTH", "512"))
FINE_TUNING_DEFAULT_LEARNING_RATE = float(os.getenv(f"{ENV_PREFIX}FINE_TUNING_DEFAULT_LEARNING_RATE", "5e-6"))
FINE_TUNING_DEFAULT_BATCH_SIZE = os.getenv(f"{ENV_PREFIX}FINE_TUNING_DEFAULT_BATCH_SIZE", "auto")
FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER = os.getenv(
    f"{ENV_PREFIX}FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER", "auto"
)
FINE_TUNING_CHECKPOINT_DIR = os.getenv(f"{ENV_PREFIX}FINE_TUNING_CHECKPOINT_DIR", "data/checkpoints")
FINE_TUNING_CHECKPOINT_INTERVAL = int(os.getenv(f"{ENV_PREFIX}FINE_TUNING_CHECKPOINT_INTERVAL", "10"))
FINE_TUNING_EVAL_BATCH_SIZE = int(os.getenv(f"{ENV_PREFIX}FINE_TUNING_EVAL_BATCH_SIZE", "1"))
FINE_TUNING_MAX_CHECKPOINTS = int(os.getenv(f"{ENV_PREFIX}FINE_TUNING_MAX_CHECKPOINTS", "5"))
FINE_TUNING_STATUS_CHECK_INTERVAL = int(os.getenv(f"{ENV_PREFIX}FINE_TUNING_STATUS_CHECK_INTERVAL", "5"))
FINE_TUNING_MIN_DISK_SPACE_MB = int(os.getenv(f"{ENV_PREFIX}FINE_TUNING_MIN_DISK_SPACE_MB", "500"))
MAX_CONCURRENT_TRAINING = int(os.getenv(f"{ENV_PREFIX}MAX_CONCURRENT_TRAINING", "1"))
MAX_CONCURRENT_EVAL = int(os.getenv(f"{ENV_PREFIX}MAX_CONCURRENT_EVAL", "2"))

# LLM settings
DEFAULT_MODEL = os.getenv(f"{ENV_PREFIX}DEFAULT_MODEL", "qwen-coder")
DEFAULT_LOCAL_MODEL = os.getenv(f"{ENV_PREFIX}DEFAULT_LOCAL_MODEL", "qwen-coder")
MAX_CONTEXT_LENGTH = int(os.getenv(f"{ENV_PREFIX}MAX_CONTEXT_LENGTH", "128000"))
DEFAULT_MAX_TOKENS = int(os.getenv(f"{ENV_PREFIX}DEFAULT_MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv(f"{ENV_PREFIX}DEFAULT_TEMPERATURE", "0.7"))
MAX_CONCURRENT_INFERENCE = int(os.getenv(f"{ENV_PREFIX}MAX_CONCURRENT_INFERENCE", "1"))
MAX_RETRIES = int(os.getenv(f"{ENV_PREFIX}MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv(f"{ENV_PREFIX}RETRY_DELAY", "1.0"))

# Embedding settings
DEFAULT_EMBEDDING_MODEL = os.getenv(f"{ENV_PREFIX}DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MAX_BATCH_SIZE = int(os.getenv(f"{ENV_PREFIX}EMBEDDING_MAX_BATCH_SIZE", "100"))
EMBEDDING_MAX_CONCURRENT_REQUESTS = int(os.getenv(f"{ENV_PREFIX}EMBEDDING_MAX_CONCURRENT", "5"))
ENABLE_EMBEDDING_CACHE = os.getenv(f"{ENV_PREFIX}ENABLE_EMBEDDING_CACHE", "true").lower() in ("true", "1", "yes")
EMBEDDING_CACHE_SIZE = int(os.getenv(f"{ENV_PREFIX}EMBEDDING_CACHE_SIZE", "10000"))

# Response settings
DEFAULT_TIMEOUT_SECONDS = int(os.getenv(f"{ENV_PREFIX}DEFAULT_TIMEOUT", "120"))
MAX_TIMEOUT_SECONDS = int(os.getenv(f"{ENV_PREFIX}MAX_TIMEOUT", "600"))
CHUNK_TIMEOUT_SECONDS = int(os.getenv(f"{ENV_PREFIX}CHUNK_TIMEOUT", "30"))
MAX_OUTPUT_TOKENS = int(os.getenv(f"{ENV_PREFIX}MAX_OUTPUT_TOKENS", "8000"))
WARN_OUTPUT_TOKENS = int(os.getenv(f"{ENV_PREFIX}WARN_OUTPUT_TOKENS", "6000"))
MAX_TOOL_OUTPUT_TOKENS = int(os.getenv(f"{ENV_PREFIX}MAX_TOOL_OUTPUT_TOKENS", "4000"))
MAX_OUTPUT_CHARS = int(os.getenv(f"{ENV_PREFIX}MAX_OUTPUT_CHARS", "15000"))
LARGE_OUTPUT_CHARS = int(os.getenv(f"{ENV_PREFIX}LARGE_OUTPUT_CHARS", "50000"))
ENABLE_TIMEOUT_MONITORING = os.getenv(f"{ENV_PREFIX}ENABLE_TIMEOUT_MONITORING", "true").lower() in ("true", "1", "yes")
ENABLE_COMPLETION_DETECTION = os.getenv(f"{ENV_PREFIX}ENABLE_COMPLETION_DETECTION", "true").lower() in (
    "true",
    "1",
    "yes",
)
ENABLE_OUTPUT_CHUNKING = os.getenv(f"{ENV_PREFIX}ENABLE_OUTPUT_CHUNKING", "true").lower() in ("true", "1", "yes")

# Model timeout overrides
MODEL_TIMEOUT_OVERRIDES = {
    "phi4": 120,
    "qwen-coder": 90,
    "qwen2.5-0.5b": 60,
    "llama": 90,
}

# LLM Model configurations
LLM_MODELS = {
    "qwen2.5-0.5b": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_type": "huggingface",
        "temperature": 0.3,
        "top_p": 0.9,
        "memory_gb": 1.5,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "tinyllama": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_type": "huggingface",
        "temperature": 0.4,
        "top_p": 0.9,
        "memory_gb": 2.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "qwen-coder": {
        "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "model_type": "huggingface",
        "temperature": 0.2,
        "top_p": 0.9,
        "memory_gb": 3.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "phi-2": {
        "model_name": "microsoft/phi-2",
        "model_type": "huggingface",
        "temperature": 0.5,
        "top_p": 0.9,
        "memory_gb": 5.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
    },
    "phi-1.5": {
        "model_name": "microsoft/phi-1_5",
        "model_type": "huggingface",
        "temperature": 0.7,
        "top_p": 0.95,
        "memory_gb": 2.5,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
    },
}

# Embedding model configurations
EMBEDDING_MODELS = {
    "text-embedding-ada-002": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1536,
        "description": "OpenAI's ada-002 model - emulated using BGE model with matching dimensions",
        "format": "OpenAI",
    },
    "nomic-embed-text": {
        "model_name": "nomic-ai/nomic-embed-text-v1",
        "dimensions": 768,
        "description": "Very fast, good all-rounder, GPU/CPU friendly",
        "format": "HuggingFace",
    },
    "bge-small-en-v1.5": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "description": "Tiny and very RAG-optimized, fast and low-memory",
        "format": "HuggingFace",
    },
}

# Fine-tuning models (derived from LLM_MODELS)
FINE_TUNING_MODELS = {
    model_id: config["model_name"]
    for model_id, config in LLM_MODELS.items()
    if config.get("model_type") == "huggingface"
}
