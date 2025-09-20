import logging
from pathlib import Path
from typing import List, Tuple

from rose_server._inference import EmbeddingModel
from rose_server.config.settings import settings

logger = logging.getLogger(__name__)


def get_embedding_model() -> EmbeddingModel:
    model_path = (Path(settings.data_dir) / "models" / "Qwen--Qwen3-Embedding-0.6B-GGUF").resolve()

    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {model_path}")

    gguf_file = next((f for f in gguf_files if "Q8_0" in f.name), gguf_files[0])
    tokenizer_file = model_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    model = EmbeddingModel(str(gguf_file.resolve()), str(tokenizer_file.resolve()), "auto")
    logger.info(f"Loaded embeddings: {gguf_file.name}")
    return model


async def encode(text: str, model: EmbeddingModel) -> List[float]:
    return await model.encode(text)  # type: ignore[no-any-return]


async def encode_batch(texts: List[str], model: EmbeddingModel) -> Tuple[List[List[float]], int]:
    if not texts:
        return [], 0

    return await model.encode_batch(texts)  # type: ignore[no-any-return]
