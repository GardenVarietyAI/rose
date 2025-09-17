from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rose_server.config.settings import settings
from rose_server.models.qwen_configs import get_qwen_config
from rose_server.models.store import LanguageModel


class ModelConfig(BaseModel):
    model_id: str = Field(..., description="Database model ID for caching")
    model_name: str = Field(..., description="The HuggingFace model identifier")
    model_path: Optional[str] = Field(None, description="Path to fine-tuned model")
    base_model: Optional[str] = Field(None, description="Parent model for fine-tuned models")
    kind: Optional[str] = Field(None, description="Model kind for inference engine")
    quantization: Optional[str] = Field(None, description="Quantization type (e.g., 'int8')")
    lora_target_modules: Optional[List[str]] = Field(None, description="LoRA target modules")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    length_penalty: float = Field(1.0, description="Length penalty")
    max_response_tokens: int = Field(2048, description="Maximum tokens in response")
    inference_timeout: float = Field(120.0, description="Timeout for inference in seconds")
    data_dir: str = Field("./data", description="Data directory for models")

    @classmethod
    def from_language_model(cls, model: LanguageModel) -> "ModelConfig":
        qwen_config = get_qwen_config(model.id)
        config_data = {
            "model_id": model.id,
            "model_name": model.model_name,
            "kind": model.kind,
            "temperature": model.temperature if model.temperature is not None else qwen_config.temperature,
            "top_p": model.top_p if model.top_p is not None else qwen_config.top_p,
            "repetition_penalty": qwen_config.repetition_penalty,
            "max_response_tokens": qwen_config.max_response_tokens,
            "inference_timeout": settings.inference_timeout,
            "data_dir": settings.data_dir,
            "lora_target_modules": model.lora_target_modules,
            "quantization": model.quantization,
        }

        if model.is_fine_tuned and model.path:
            config_data["model_path"] = str(Path(settings.data_dir) / model.path)
            config_data["base_model"] = model.parent
        else:
            # For base models, construct the expected path where models are stored
            config_data["model_path"] = str(Path(settings.models_dir) / model.id)

        return cls(**config_data)


class ModelCreateRequest(BaseModel):
    model_name: str  # HuggingFace model name
    kind: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: Optional[int] = None
    lora_target_modules: Optional[List[str]] = None
    quantization: Optional[str] = None


class ModelResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    object: str = "model"
    created_at: int
    owned_by: str
    permissions: Optional[List[str]] = Field(default_factory=lambda: [])
    parent: Optional[str]
    kind: Optional[str] = ""
    model_name: Optional[str]
    lora_target_modules: Optional[List[str]] = Field(default_factory=lambda: [])
    quantization: Optional[str] = None
