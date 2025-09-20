import logging
from typing import List

from rose_server._inference import RerankerModel

logger = logging.getLogger(__name__)


async def score(query: str, response: str, model: RerankerModel) -> float:
    relevance_score = await model.score(query, response)
    logger.debug(f"Query: {query[:50]}, Doc: {response[:50]}, Score: {relevance_score:.4f}")
    return relevance_score


async def score_batch(
    queries: List[str],
    responses: List[str],
    model: RerankerModel,
) -> List[float]:
    if len(queries) != len(responses):
        raise ValueError("Length mismatch")

    if not queries:
        return []

    return await model.score_batch(queries, responses)
