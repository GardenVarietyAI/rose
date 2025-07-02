"""Pydantic-based settings for ROSE Server."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for ROSE Server."""

    model_config = SettingsConfigDict(
        env_prefix="ROSE_SERVER_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,  # Make settings immutable
    )

    # Service settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8004, description="Server port")
    reload: bool = Field(default=True, description="Enable auto-reload in development")
    log_level: str = Field(default="INFO", description="Logging level")

    # Data directories
    data_dir: str = Field(default="./data", description="Base data directory")

    # ChromaDB settings
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8003, description="ChromaDB port")
    chroma_persist_dir: str = Field(default="./data/chroma", description="ChromaDB storage path")

    # Fine-tuning settings
    fine_tuning_checkpoint_dir: str = Field(default="data/checkpoints", description="Checkpoint directory")
    fine_tuning_checkpoint_interval: int = Field(default=10, description="Steps between checkpoints")
    fine_tuning_max_checkpoints: int = Field(default=5, description="Maximum checkpoints to keep")
    fine_tuning_eval_batch_size: int = Field(default=1, description="Evaluation batch size")
    training_interval: int = Field(default=30, description="Training job check interval in seconds")

    # Inference settings
    inference_uri: str = Field(default="ws://localhost:8005", description="WebSocket URI for inference worker")
    default_model: str = Field(default="qwen2.5-0.5b", description="Default model for inference")
    inference_timeout: int = Field(default=30, description="Inference timeout in seconds")
    max_concurrent_inference: int = Field(default=1, description="Maximum concurrent inference requests")

    # Webhook settings
    webhook_url: str = Field(
        default="http://localhost:8004/v1/webhooks/jobs", description="Webhook URL for job updates"
    )

    # Derived properties
    @property
    def model_offload_dir(self) -> str:
        """Directory for model offloading."""
        return f"{self.data_dir}/offload"

    @property
    def log_format(self) -> str:
        """Log message format."""
        return "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Global settings instance
settings = Settings()
