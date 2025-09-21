from typing import List, Tuple

from rose_server._inference import EmbeddingModel


async def generate_embeddings(texts: List[str], embedding_model: EmbeddingModel) -> Tuple[List[List[float]], int]:
    if not texts:
        return [], 0

    return await embedding_model.encode_batch(texts)  # type: ignore[no-any-return]


async def generate_query_embedding(query: str, embedding_model: EmbeddingModel) -> List[float]:
    return await embedding_model.encode(query)  # type: ignore[no-any-return]


async def compute_embeddings_with_tokens(texts: List[str], model: EmbeddingModel) -> Tuple[List[List[float]], int]:
    return await generate_embeddings(texts, model)
