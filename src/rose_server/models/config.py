"""Strongly typed model configuration."""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from rose_server.config.settings import settings
from rose_server.models.store import LanguageModel


class ModelConfig(BaseModel):
    """Configuration for model inference."""

    model_name: str = Field(..., description="The model identifier")
    model_path: Optional[str] = Field(None, description="Path to fine-tuned model")
    base_model: Optional[str] = Field(None, description="Parent model for fine-tuned models")
    quantization: Optional[str] = Field(None, description="Quantization type (e.g., 'int8')")
    lora_target_modules: Optional[List[str]] = Field(None, description="LoRA target modules")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    length_penalty: float = Field(1.0, description="Length penalty")
    max_response_tokens: int = Field(2048, description="Maximum tokens in response")

    @classmethod
    def from_language_model(cls, model: LanguageModel) -> "ModelConfig":
        """Create configuration from a database model.

        Args:
            model: The LanguageModel from the database

        Returns:
            A ModelConfig instance with all relevant settings
        """
        # Start with basic configuration
        config_data = {
            "model_name": model.model_name,
            "temperature": model.temperature or 0.7,
            "top_p": model.top_p or 0.9,
        }

        # Add fine-tuning specific configuration
        if model.is_fine_tuned and model.path:
            config_data["model_path"] = str(Path(settings.data_dir) / model.path)
            config_data["base_model"] = model.parent

        # Add LoRA modules if present
        if model.get_lora_modules():
            config_data["lora_target_modules"] = model.get_lora_modules()

        # Add quantization if specified
        if model.quantization:
            config_data["quantization"] = model.quantization

        return cls(**config_data)
