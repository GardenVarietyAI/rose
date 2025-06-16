from typing import Dict, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    model_name: str = Field(..., description="HuggingFace model identifier")
    model_type: Literal["huggingface"] = Field(default="huggingface", description="Model type")
    max_response_tokens: int = Field(default=1024, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")


class EmbeddingModelConfig(BaseModel):
    """Configuration for a single embedding model."""

    model_name: str = Field(..., description="HuggingFace model identifier")
    dimensions: int = Field(..., description="Vector dimensions")
    description: str = Field(..., description="Model description")
    format: Literal["OpenAI", "HuggingFace"] = Field(default="HuggingFace", description="API format compatibility")


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
FINE_TUNING_MODELS = {
    model_id: config["model_name"]
    for model_id, config in LLM_MODELS.items()
    if config.get("model_type") == "huggingface"
}


def get_llm_models() -> Dict[str, Dict]:
    """Get all LLM model configurations."""
    return LLM_MODELS.copy()


def get_embedding_models() -> Dict[str, Dict]:
    """Get all embedding model configurations."""
    return EMBEDDING_MODELS.copy()


def get_fine_tunable_models() -> Dict[str, str]:
    """Get mapping of model_id -> model_name for fine-tuning."""
    return FINE_TUNING_MODELS.copy()


def get_model_config(model_id: str) -> Dict:
    """Get configuration for a specific LLM model."""
    if model_id not in LLM_MODELS:
        raise ValueError(f"Unknown model: {model_id}")
    return LLM_MODELS[model_id].copy()
