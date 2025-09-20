from typing import List

from rose_server._inference import EmbeddingModel


async def generate_embeddings(texts: List[str], embedding_model: EmbeddingModel) -> List[List[float]]:
    if not texts:
        return []

    return await embedding_model.encode_batch(texts)


async def generate_query_embedding(query: str, embedding_model: EmbeddingModel) -> List[float]:
    return await embedding_model.encode(query)


async def compute_embeddings_with_tokens(texts: List[str], model: EmbeddingModel) -> tuple[List[List[float]], int]:
    embeddings = await generate_embeddings(texts, model)

    # Rough token estimate (4 chars per token)
    total_tokens = sum(len(text) // 4 for text in texts)
    return embeddings, total_tokens
