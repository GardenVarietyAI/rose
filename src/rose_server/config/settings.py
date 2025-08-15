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
        extra="ignore",
    )

    # Service settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8004, description="Server port")
    reload: bool = Field(default=True, description="Enable auto-reload in development")
    log_level: str = Field(default="INFO", description="Logging level")

    # Data directories
    data_dir: str = Field(default="./data", description="Base data directory")

    # Database settings (non-prefixed for dbmate compatibility)
    database_url: str = Field(default="sqlite:data/rose.db", description="Database URL")
    dbmate_migrations_dir: str = Field(default="db/migrations", description="Migrations directory")
    dbmate_schema_file: str = Field(default="db/schema.sql", description="Schema file location")

    # Fine-tuning settings
    fine_tuning_checkpoint_dir: str = Field(default="data/checkpoints", description="Checkpoint directory")
    fine_tuning_checkpoint_interval: int = Field(default=10, description="Steps between checkpoints")
    fine_tuning_max_checkpoints: int = Field(default=5, description="Maximum checkpoints to keep")
    fine_tuning_eval_batch_size: int = Field(default=1, description="Evaluation batch size")
    fine_tuning_default_epochs: int = Field(default=3, description="Default number of epochs for fine-tuning")
    fine_tuning_default_batch_size: str = Field(default="auto", description="Default batch size for fine-tuning")
    fine_tuning_default_learning_rate_multiplier: str = Field(
        default="auto", description="Default learning rate multiplier"
    )
    fine_tuning_base_learning_rate: float = Field(
        default=2e-5, description="Base learning rate for OpenAI-compatible multiplier"
    )
    fine_tuning_auto_batch_size: int = Field(default=4, description="Batch size when 'auto' is specified")
    fine_tuning_auto_epochs: int = Field(default=3, description="Number of epochs when 'auto' is specified")
    fine_tuning_auto_learning_rate_multiplier: float = Field(
        default=0.1, description="Learning rate multiplier when 'auto' is specified"
    )
    training_interval: int = Field(default=30, description="Training job check interval in seconds")

    # File upload settings
    max_file_upload_size: int = Field(
        default=100 * 1024 * 1024, description="Maximum file upload size in bytes (default: 100MB)"
    )

    # Inference settings
    inference_uri: str = Field(default="ws://localhost:8005", description="WebSocket URI for inference worker")
    inference_timeout: int = Field(default=300, description="Inference timeout in seconds")
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
