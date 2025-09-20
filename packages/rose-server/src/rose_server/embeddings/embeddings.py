import logging
from pathlib import Path
from typing import List, Tuple

from tokenizers import Tokenizer

from rose_server._inference import EmbeddingModel
from rose_server.config.settings import settings

logger = logging.getLogger(__name__)


def get_tokenizer() -> Tokenizer:
    tokenizer_path = Path(settings.models_dir) / settings.embedding_model_name / "tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    return Tokenizer.from_file(str(tokenizer_path))


def get_embedding_model() -> EmbeddingModel:
    model_path = (Path(settings.models_dir) / settings.embedding_model_name).resolve()

    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {model_path}")

    # Find the specified quantization level
    gguf_file = next((f for f in gguf_files if settings.embedding_model_quantization in f.name), None)

    if not gguf_file:
        available_quants = [f.name for f in gguf_files]
        raise FileNotFoundError(
            f"Quantization {settings.embedding_model_quantization} not found in {model_path}. "
            f"Available: {available_quants}"
        )

    tokenizer_file = model_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    model = EmbeddingModel(str(gguf_file.resolve()), str(tokenizer_file.resolve()), settings.embedding_device)
    logger.info(f"Loaded embeddings: {gguf_file.name} on device: {settings.embedding_device}")
    return model


async def encode(text: str, model: EmbeddingModel) -> List[float]:
    return await model.encode(text)  # type: ignore[no-any-return]


async def encode_batch(texts: List[str], model: EmbeddingModel) -> Tuple[List[List[float]], int]:
    if not texts:
        return [], 0

    return await model.encode_batch(texts)  # type: ignore[no-any-return]
