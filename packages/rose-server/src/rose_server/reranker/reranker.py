import logging
from pathlib import Path
from typing import List

from rose_server._inference import RerankerModel
from rose_server.config.settings import settings

logger = logging.getLogger(__name__)


def get_reranker_model() -> RerankerModel:
    model_path = Path(settings.models_dir) / "QuantFactory--Qwen3-Reranker-0.6B-GGUF"

    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {model_path}")

    gguf_file = next((f for f in gguf_files if "Q8_0" in f.name), gguf_files[0])
    tokenizer_file = model_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    model = RerankerModel(str(gguf_file), str(tokenizer_file), "auto")
    logger.info(f"Loaded reranker: {gguf_file.name}")
    return model


async def score(query: str, document: str, model: RerankerModel) -> float:
    return await model.score(query, document)


async def score_batch(queries: List[str], documents: List[str], model: RerankerModel) -> List[float]:
    if len(queries) != len(documents):
        raise ValueError("Length mismatch")

    if not queries:
        return []

    return await model.score_batch(queries, documents)
