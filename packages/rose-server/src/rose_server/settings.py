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

    # Fine-tuning settings
    fine_tuning_checkpoint_dir: str = Field(default="data/checkpoints", description="Checkpoint directory")
    fine_tuning_checkpoint_interval: int = Field(default=10, description="Steps between checkpoints")
    fine_tuning_max_checkpoints: int = Field(default=5, description="Maximum checkpoints to keep")
    fine_tuning_eval_batch_size: int = Field(default=1, description="Evaluation batch size")
    fine_tuning_base_learning_rate: float = Field(
        default=2e-5, description="Base learning rate for OpenAI-compatible multiplier"
    )
    fine_tuning_auto_batch_size: int = Field(default=4, description="Batch size when 'auto' is specified")
    fine_tuning_auto_epochs: int = Field(default=3, description="Number of epochs when 'auto' is specified")
    fine_tuning_auto_learning_rate_multiplier: float = Field(
        default=0.1, description="Learning rate multiplier when 'auto' is specified"
    )

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
