from typing import Any, Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Embedding model configurations
EMBEDDING_MODELS = {
    "text-embedding-ada-002": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1536,
        "description": "BGE",
        "format": "HuggingFace",
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

# Simple caches for models and tokenizers
_models: Dict[str, SentenceTransformer] = {}
_tokenizers: Dict[str, AutoTokenizer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Get or load an embedding model."""
    if model_name not in _models:
        if model_name in EMBEDDING_MODELS:
            model_path = str(EMBEDDING_MODELS[model_name]["model_name"])
        else:
            model_path = str(model_name)
        _models[model_name] = SentenceTransformer(model_path)
    return _models[model_name]


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    """Get or load a tokenizer."""
    if model_name not in _tokenizers:
        if model_name in EMBEDDING_MODELS:
            model_path = EMBEDDING_MODELS[model_name]["model_name"]
        else:
            model_path = model_name
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    return _tokenizers[model_name]


def _encode_batch(model: SentenceTransformer, batch: List[str]) -> np.ndarray:
    return model.encode(batch)


def generate_embeddings(
    texts: Union[str, List[str]], model_name: str = "bge-small-en-v1.5", batch_size: int = 32
) -> Dict[str, Any]:
    if isinstance(texts, str):
        texts = [texts]

    model = _get_model(model_name)
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = _encode_batch(model, batch)
        # Convert 2D array to list of 1D arrays
        all_embeddings.extend(list(batch_embeddings))

    embedding_tokenizer = _get_tokenizer(model_name)
    total_tokens = 0
    for text in texts:
        tokens = embedding_tokenizer.encode(text)  # type: ignore
        total_tokens += len(tokens)

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
