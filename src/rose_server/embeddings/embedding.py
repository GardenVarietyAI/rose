import asyncio
from typing import Any, Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

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

# Simple caches for models and tokenizers
_models: Dict[str, SentenceTransformer] = {}
_tokenizers: Dict[str, Tokenizer] = {}


def generate_embeddings(
    texts: Union[str, List[str]],
    model_name: str = "bge-small-en-v1.5",
    batch_size: int = 32,
) -> Dict[str, Any]:
    if model_name == "text-embedding-ada-002":
        model_name = "bge-small-en-v1.5"

    if isinstance(texts, str):
        texts = [texts]

    if model_name not in _models:
        if model_name in EMBEDDING_MODELS:
            model_path = str(EMBEDDING_MODELS[model_name]["model_name"])
        else:
            model_path = str(model_name)
        _models[model_name] = SentenceTransformer(model_path)
    model = _models[model_name]

    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch)
        # Convert 2D array to list of 1D arrays
        all_embeddings.extend(list(batch_embeddings))

    if model_name not in _tokenizers:
        if model_name in EMBEDDING_MODELS:
            model_path = str(EMBEDDING_MODELS[model_name]["model_name"])
        else:
            model_path = model_name
        _tokenizers[model_name] = Tokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    embedding_tokenizer = _tokenizers[model_name]

    total_tokens = 0
    for text in texts:
        tokens = embedding_tokenizer.encode(text)
        total_tokens += len(tokens.ids)

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
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
