import logging
from dataclasses import dataclass
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from tokenizers import Tokenizer

from rose_server.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelConfig:
    model_name: str
    dimensions: int
    description: str
    format: str
    model_path: str
    model_filename: str
    tokenizer_path: str


_model_config = EmbeddingModelConfig(
    model_name="qwen3-embedding-0.6b-onnx",
    dimensions=1024,
    description="Qwen3 embedding model",
    format="ONNX",
    model_path=f"{settings.models_dir}/Qwen3-Embedding-0.6B-ONNX",
    model_filename="model.onnx",
    tokenizer_path=f"{settings.models_dir}/Qwen3-Embedding-0.6B-ONNX/tokenizer.json",
)


def _register_model() -> None:
    TextEmbedding.add_custom_model(
        model=_model_config.model_name,
        pooling=PoolingType.LAST_TOKEN,
        normalization=True,
        sources=ModelSource(url="localhost"),
        dim=_model_config.dimensions,
        model_file=_model_config.model_filename,
    )
    logger.info(f"Registered {_model_config.model_name}")


_register_model()


def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(
        model_name=_model_config.model_name,
        device=settings.default_embedding_device,
        specific_model_path=str(Path(_model_config.model_path).absolute()),
    )


def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(_model_config.tokenizer_path)
