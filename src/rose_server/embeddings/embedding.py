import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Union

import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer

from rose_server.config.settings import settings

EMBEDDING_MODELS = {
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


@lru_cache(maxsize=4)
def _get_model(model_name: str, device: str = "cpu") -> TextEmbedding:
    if model_name in EMBEDDING_MODELS:
        model_path = str(EMBEDDING_MODELS[model_name]["model_name"])
    else:
        model_path = str(model_name)
    return TextEmbedding(model_name=model_path, device=device)


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str) -> Tokenizer:
    if model_name in EMBEDDING_MODELS:
        model_path = str(EMBEDDING_MODELS[model_name]["model_name"])
    else:
        model_path = model_name
    return Tokenizer.from_pretrained(model_path)


def embedding_model() -> TextEmbedding:
    """Get the default embedding model from settings."""
    device = getattr(settings, "default_embedding_device", "cpu")
    return _get_model(settings.default_embedding_model, device)


def clear_embedding_cache() -> None:
    """Clear cached models and tokenizers.

    This can be used when memory pressure is detected to free cached
    embedding models and tokenizers.
    """
    _get_model.cache_clear()
    _get_tokenizer.cache_clear()


def reload_embedding_model() -> TextEmbedding:
    """Reload the default embedding model, clearing cache first."""
    clear_embedding_cache()
    return embedding_model()


def generate_embeddings(
    texts: Union[str, List[str]],
    model_name: str = "bge-small-en-v1.5",
    batch_size: int = 32,
) -> Dict[str, Any]:
    if model_name == "text-embedding-ada-002":
        model_name = "bge-small-en-v1.5"

    if isinstance(texts, str):
        texts = [texts]

    model = _get_model(model_name)

    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = list(model.embed(batch))
        all_embeddings.extend(batch_embeddings)

    embedding_tokenizer = _get_tokenizer(model_name)

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
    model_name: str = "bge-small-en-v1.5",
    batch_size: int = 32,
) -> Dict[str, Any]:
    return await asyncio.to_thread(generate_embeddings, texts, model_name, batch_size)
