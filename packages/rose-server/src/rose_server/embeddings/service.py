from typing import Any, List, Tuple

import numpy as np
from fastembed import TextEmbedding


def generate_embeddings(texts: List[str], embedding_model: TextEmbedding) -> List[np.ndarray]:
    if not texts:
        return []

    return list(embedding_model.embed(texts))


def generate_query_embedding(query: str, embedding_model: TextEmbedding) -> np.ndarray:
    embeddings = list(embedding_model.embed([query]))
    return embeddings[0] if embeddings else np.array([])


def compute_embeddings_with_tokens(
    texts: List[str], model: Any, tokenizer: Any, batch_size: int = 32
) -> Tuple[List[np.ndarray], int]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = list(model.embed(batch))
        all_embeddings.extend(batch_embeddings)

    total_tokens = sum(len(tokenizer.encode(text).ids) for text in texts)
    return all_embeddings, total_tokens
