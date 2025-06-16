from typing import Any, Dict, List, Union

import numpy as np

from rose_server.services import get_embedding_manager


def _encode_batch(model, batch: List[str]) -> List[np.ndarray]:
    return model.encode(batch)

def generate_embeddings(
    texts: Union[str, List[str]], model_name: str = "bge-small-en-v1.5", batch_size: int = 32
) -> Dict[str, Any]:
    if isinstance(texts, str):
        texts = [texts]
    embedding_manager = get_embedding_manager()
    model = embedding_manager.get_model(model_name)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(_encode_batch(model, batch))
    embedding_tokenizer = embedding_manager.get_tokenizer(model_name)
    total_tokens = sum(len(embedding_tokenizer.encode(text)) for text in texts)
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
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }