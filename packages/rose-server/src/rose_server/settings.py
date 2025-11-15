from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROSE_SERVER_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data directories
    data_dir: str = Field(default="./data", description="Base data directory")

    # Inference settings
    inference_timeout: int = Field(default=300, description="Inference timeout in seconds")
    max_concurrent_inference: int = Field(default=1, description="Maximum concurrent inference requests")

    # Derived properties
    @property
    def models_dir(self) -> str:
        """Directory for storing models."""
        return f"{self.data_dir}/models"

    @property
    def model_offload_dir(self) -> str:
        """Directory for model offloading."""
        return f"{self.data_dir}/offload"
