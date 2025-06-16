import os

ENV_PREFIX = "ROSE_SERVER_"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["LOKY_CONTEXT_STRATEGY"] = "spawn"

def get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(f"{ENV_PREFIX}{key}", str(default)))

def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    return float(os.getenv(f"{ENV_PREFIX}{key}", str(default)))

def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(f"{ENV_PREFIX}{key}", str(default)).lower() in ("true", "1", "yes")

def get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.getenv(f"{ENV_PREFIX}{key}", default)

class ServiceConfig:
    """General service configuration."""

    SERVICE_NAME = "ROSE"
    SERVICE_VERSION = "0.1.0"
    HOST = get_env_str("HOST", "0.0.0.0")
    PORT = get_env_int("PORT", 8004)
    RELOAD = get_env_bool("RELOAD", True)
    LOG_LEVEL = get_env_str("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATA_DIR = get_env_str("DATA_DIR", "./data")
    CHROMA_PERSIST_DIR = get_env_str("CHROMA_PERSIST_DIR", "./data/chroma")
    MODEL_OFFLOAD_DIR = get_env_str("MODEL_OFFLOAD_DIR", "./data/offload")
    CHROMA_HOST = get_env_str("CHROMA_HOST", "localhost")
    CHROMA_PORT = get_env_int("CHROMA_PORT", 8003)
    HEALTH_CHECK_ENABLED = get_env_bool("HEALTH_CHECK_ENABLED", True)
    HEALTH_CHECK_WINDOW_MINUTES = get_env_int("HEALTH_CHECK_WINDOW_MINUTES", 5)
    FINE_TUNING_DEFAULT_EPOCHS = get_env_int("FINE_TUNING_DEFAULT_EPOCHS", 3)
    FINE_TUNING_DEFAULT_MAX_LENGTH = get_env_int("FINE_TUNING_DEFAULT_MAX_LENGTH", 512)
    FINE_TUNING_DEFAULT_LEARNING_RATE = get_env_float("FINE_TUNING_DEFAULT_LEARNING_RATE", 5e-6)
    FINE_TUNING_DEFAULT_BATCH_SIZE = get_env_str("FINE_TUNING_DEFAULT_BATCH_SIZE", "auto")
    FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER = get_env_str("FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER", "auto")
    FINE_TUNING_CHECKPOINT_DIR = get_env_str("FINE_TUNING_CHECKPOINT_DIR", "data/checkpoints")
    FINE_TUNING_CHECKPOINT_INTERVAL = get_env_int("FINE_TUNING_CHECKPOINT_INTERVAL", 10)
    FINE_TUNING_MAX_CHECKPOINTS = get_env_int("FINE_TUNING_MAX_CHECKPOINTS", 5)
    FINE_TUNING_STATUS_CHECK_INTERVAL = get_env_int("FINE_TUNING_STATUS_CHECK_INTERVAL", 5)
    FINE_TUNING_MIN_DISK_SPACE_MB = get_env_int("FINE_TUNING_MIN_DISK_SPACE_MB", 500)
    MAX_CONCURRENT_TRAINING = get_env_int("MAX_CONCURRENT_TRAINING", 1)
    MAX_CONCURRENT_EVAL = get_env_int("MAX_CONCURRENT_EVAL", 2)
    @classmethod

    def get_service_info(cls) -> dict:
        """Get service information."""
        return {
            "name": cls.SERVICE_NAME,
            "version": cls.SERVICE_VERSION,
            "host": cls.HOST,
            "port": cls.PORT,
        }

class LLMConfig:
    """Configuration for LLM handling."""

    DEFAULT_MODEL = get_env_str("DEFAULT_MODEL", "qwen-coder")
    DEFAULT_LOCAL_MODEL = get_env_str("DEFAULT_LOCAL_MODEL", "qwen-coder")
    MAX_CONTEXT_LENGTH = get_env_int("MAX_CONTEXT_LENGTH", 128000)
    DEFAULT_MAX_TOKENS = get_env_int("DEFAULT_MAX_TOKENS", 4096)
    DEFAULT_TEMPERATURE = get_env_float("DEFAULT_TEMPERATURE", 0.7)
    MAX_CONCURRENT_INFERENCE = get_env_int("MAX_CONCURRENT_INFERENCE", 1)
    MAX_CONCURRENT_TRAINING = get_env_int("MAX_CONCURRENT_TRAINING", 1)
    MAX_RETRIES = get_env_int("MAX_RETRIES", 3)
    RETRY_DELAY = get_env_float("RETRY_DELAY", 1.0)
    @classmethod

    def get_config_summary(cls) -> dict:
        """Get configuration summary."""
        return {
            "defaults": {
                "model": cls.DEFAULT_MODEL,
                "local_model": cls.DEFAULT_LOCAL_MODEL,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "temperature": cls.DEFAULT_TEMPERATURE,
            },
            "limits": {
                "max_context": cls.MAX_CONTEXT_LENGTH,
                "concurrent_inference": cls.MAX_CONCURRENT_INFERENCE,
                "concurrent_training": cls.MAX_CONCURRENT_TRAINING,
            },
            "retry": {
                "max_retries": cls.MAX_RETRIES,
                "delay": cls.RETRY_DELAY,
            },
        }

class EmbeddingConfig:
    """Configuration for embeddings."""

    DEFAULT_EMBEDDING_MODEL = get_env_str("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    MAX_BATCH_SIZE = get_env_int("EMBEDDING_MAX_BATCH_SIZE", 100)
    MAX_CONCURRENT_REQUESTS = get_env_int("EMBEDDING_MAX_CONCURRENT", 5)
    ENABLE_EMBEDDING_CACHE = get_env_bool("ENABLE_EMBEDDING_CACHE", True)
    EMBEDDING_CACHE_SIZE = get_env_int("EMBEDDING_CACHE_SIZE", 10000)
    @classmethod

    def get_config_summary(cls) -> dict:
        """Get configuration summary."""
        return {
            "default_model": cls.DEFAULT_EMBEDDING_MODEL,
            "batch": {
                "max_size": cls.MAX_BATCH_SIZE,
                "max_concurrent": cls.MAX_CONCURRENT_REQUESTS,
            },
            "cache": {
                "enabled": cls.ENABLE_EMBEDDING_CACHE,
                "size": cls.EMBEDDING_CACHE_SIZE,
            },
        }

class ResponseConfig:
    """Configuration for response handling."""

    DEFAULT_TIMEOUT_SECONDS = get_env_int("DEFAULT_TIMEOUT", 120)
    MAX_TIMEOUT_SECONDS = get_env_int("MAX_TIMEOUT", 600)
    CHUNK_TIMEOUT_SECONDS = get_env_int("CHUNK_TIMEOUT", 30)
    MAX_OUTPUT_TOKENS = get_env_int("MAX_OUTPUT_TOKENS", 8000)
    WARN_OUTPUT_TOKENS = get_env_int("WARN_OUTPUT_TOKENS", 6000)
    MAX_TOOL_OUTPUT_TOKENS = get_env_int("MAX_TOOL_OUTPUT_TOKENS", 4000)
    MAX_OUTPUT_CHARS = get_env_int("MAX_OUTPUT_CHARS", 15000)
    LARGE_OUTPUT_CHARS = get_env_int("LARGE_OUTPUT_CHARS", 50000)
    ENABLE_TIMEOUT_MONITORING = get_env_bool("ENABLE_TIMEOUT_MONITORING", True)
    ENABLE_COMPLETION_DETECTION = get_env_bool("ENABLE_COMPLETION_DETECTION", True)
    ENABLE_OUTPUT_CHUNKING = get_env_bool("ENABLE_OUTPUT_CHUNKING", True)
    MODEL_TIMEOUT_OVERRIDES = {
        "phi4": 120,
        "qwen-coder": 90,
        "qwen2.5-0.5b": 60,
        "llama": 90,
    }
    @classmethod

    def get_timeout_for_model(cls, model: str) -> int:
        """Get timeout for specific model."""
        if model in cls.MODEL_TIMEOUT_OVERRIDES:
            return cls.MODEL_TIMEOUT_OVERRIDES[model]
        for model_prefix, timeout in cls.MODEL_TIMEOUT_OVERRIDES.items():
            if model.startswith(model_prefix):
                return timeout
        return cls.DEFAULT_TIMEOUT_SECONDS
    @classmethod

    def get_config_summary(cls) -> dict:
        """Get configuration summary for logging."""
        return {
            "timeouts": {
                "default": cls.DEFAULT_TIMEOUT_SECONDS,
                "max": cls.MAX_TIMEOUT_SECONDS,
                "chunk": cls.CHUNK_TIMEOUT_SECONDS,
            },
            "limits": {
                "max_output_tokens": cls.MAX_OUTPUT_TOKENS,
                "warn_output_tokens": cls.WARN_OUTPUT_TOKENS,
                "max_tool_output_tokens": cls.MAX_TOOL_OUTPUT_TOKENS,
            },
            "features": {
                "timeout_monitoring": cls.ENABLE_TIMEOUT_MONITORING,
                "completion_detection": cls.ENABLE_COMPLETION_DETECTION,
                "output_chunking": cls.ENABLE_OUTPUT_CHUNKING,
            },
        }

def get_full_config() -> dict:
    """Get full configuration summary."""
    return {
        "service": ServiceConfig.get_service_info(),
        "llms": LLMConfig.get_config_summary(),
        "embeddings": EmbeddingConfig.get_config_summary(),
        "responses": ResponseConfig.get_config_summary(),
        "environment": {
            "prefix": ENV_PREFIX,
        },
    }