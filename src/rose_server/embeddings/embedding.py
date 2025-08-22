import asyncio
import logging
from functools import lru_cache
from typing import Any, Dict, List, Union

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from tokenizers import Tokenizer

from rose_server.config.settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = {
    "qwen3-embedding-0.6b": {
        "model_name": "onnx-community/Qwen3-Embedding-0.6B-ONNX",
        "dimensions": 768,
        "description": "Qwen3 embedding model",
        "format": "ONNX",
    },
}


def _register_qwen3_model() -> None:
    """Register qwen3 embedding model with FastEmbed."""
    try:
        TextEmbedding.add_custom_model(
            model=str(EMBEDDING_MODELS["qwen3-embedding-0.6b"]["model_name"]),
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=str(EMBEDDING_MODELS["qwen3-embedding-0.6b"]["model_name"])),
            dim=768,
            model_file="onnx/model.onnx",
            description="Qwen3 embedding model",
        )
    except Exception as e:
        print(f"Failed to register Qwen3 embedding model: {e}")
        pass


_register_qwen3_model()


@lru_cache(maxsize=4)
def _get_model(model_name: str, device: str = "cpu") -> TextEmbedding:
    return TextEmbedding(model=model_name, device=device)


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> Tokenizer:
    if model_name in EMBEDDING_MODELS:
        model_path = str(EMBEDDING_MODELS[model_name]["model_name"])
    else:
        model_path = model_name
    return Tokenizer.from_pretrained(model_path)


def embedding_model() -> TextEmbedding:
    """Get the default embedding model from settings."""
    return _get_model(settings.default_embedding_model, settings.default_embedding_device)


def clear_embedding_cache() -> None:
    """Clear cached models and tokenizers.

    This can be used when memory pressure is detected to free cached
    embedding models and tokenizers.
    """
    _get_model.cache_clear()
    get_tokenizer.cache_clear()


def reload_embedding_model() -> TextEmbedding:
    """Reload the default embedding model, clearing cache first."""
    clear_embedding_cache()
    return embedding_model()


def generate_embeddings(
    texts: Union[str, List[str]],
    model_name: str = "qwen3-embedding-0.6b",
    batch_size: int = 32,
) -> Dict[str, Any]:
    if model_name == "text-embedding-ada-002":
        model_name = "qwen3-embedding-0.6b"

    if isinstance(texts, str):
        texts = [texts]

    model = _get_model(model_name)

    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = list(model.embed(batch))
        all_embeddings.extend(batch_embeddings)

    embedding_tokenizer = get_tokenizer(model_name)

    total_tokens = 0
    for text in texts:
        tokens = embedding_tokenizer.encode(text)
        total_tokens += len(tokens.ids)

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding),
                "index": i,
            }
            for i, embedding in enumerate(all_embeddings)
        ],
        "model": model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


async def generate_embeddings_async(
    texts: Union[str, List[str]],
    model_name: str = "qwen3-embedding-0.6b",
    batch_size: int = 32,
) -> Dict[str, Any]:
    return await asyncio.to_thread(generate_embeddings, texts, model_name, batch_size)
