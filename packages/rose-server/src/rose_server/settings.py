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

    # Vector store settings
    default_chunk_size: int = Field(default=512, description="Default chunk size in tokens for document chunking")
    default_chunk_overlap: int = Field(default=64, description="Default overlap in tokens between chunks")

    # Embedding model settings
    embedding_model_name: str = Field(
        default="Qwen--Qwen3-Embedding-0.6B-GGUF", description="Name of the embedding model directory in models folder"
    )
    embedding_model_quantization: str = Field(
        default="Q8_0", description="Preferred quantization level for embedding model (e.g., Q4_0, Q8_0)"
    )
    embedding_device: str = Field(default="auto", description="Device for embedding inference (auto, cpu, cuda, metal)")
    embedding_dimensions: int = Field(
        default=64, description="Output embedding dimensions (Matryoshka: 32, 64, 128, 256, 512, 768, 1024)"
    )

    # Derived properties
    @property
    def models_dir(self) -> str:
        """Directory for storing models."""
        return f"{self.data_dir}/models"

    @property
    def model_offload_dir(self) -> str:
        """Directory for model offloading."""
        return f"{self.data_dir}/offload"
