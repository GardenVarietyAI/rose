import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from tokenizers import Tokenizer

from rose_server.config.settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = {
    "qwen3-embedding-0.6b": {
        "model_name": "qwen3-embedding-0.6b-onnx",
        "dimensions": 1024,
        "description": "Qwen3 embedding model",
        "format": "ONNX",
        "model_path": "data/models/Qwen3-Embedding-0.6B-ONNX",
        "tokenizer_path": "data/models/Qwen3-Embedding-0.6B-ONNX/tokenizer.json",
    },
}


def _register_model() -> None:
    TextEmbedding.add_custom_model(
        model="qwen3-embedding-0.6b-onnx",
        pooling=PoolingType.LAST_TOKEN,
        normalization=True,
        sources=ModelSource(url="localhost"),
        dim=1024,
        model_file="model.onnx",
    )
    logger.info("Registered qwen3-embedding-0.6b-onnx")


_register_model()


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str, device: str = "cpu") -> TextEmbedding:
    if model_name in EMBEDDING_MODELS:
        local_path = Path("data/models/Qwen3-Embedding-0.6B-ONNX")
        return TextEmbedding(
            model_name=str(EMBEDDING_MODELS[model_name]["model_name"]),
            device=device,
            specific_model_path=str(local_path.absolute()),
        )
    else:
        raise ValueError("Unsupported embedding model name given")


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> Tokenizer:
    if model_name in EMBEDDING_MODELS:
        return Tokenizer.from_file("data/models/Qwen3-Embedding-0.6B-ONNX/tokenizer.json")
    else:
        raise ValueError("Missing tokenizer, unsupported embedding model name given")


def get_default_embedding_model() -> TextEmbedding:
    """Get the default embedding model from settings."""
    return get_embedding_model(settings.default_embedding_model, settings.default_embedding_device)


def clear_embedding_cache() -> None:
    """Clear cached models and tokenizers.

    This can be used when memory pressure is detected to free cached
    embedding models and tokenizers.
    """
    get_embedding_model.cache_clear()
    get_tokenizer.cache_clear()


def reload_embedding_model() -> TextEmbedding:
    """Reload the default embedding model, clearing cache first."""
    clear_embedding_cache()
    return get_default_embedding_model()


def generate_embeddings(
    texts: Union[str, List[str]],
    model_name: str = "qwen3-embedding-0.6b",
    batch_size: int = 32,
) -> Dict[str, Any]:
    if model_name == "text-embedding-ada-002":
        model_name = "qwen3-embedding-0.6b"

    if isinstance(texts, str):
        texts = [texts]

    model = get_embedding_model(model_name)

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
