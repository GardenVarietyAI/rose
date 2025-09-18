import logging
from pathlib import Path

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from tokenizers import Tokenizer

from rose_server.config.settings import settings

logger = logging.getLogger(__name__)


def get_reranker_session() -> InferenceSession:
    model_path = Path(f"{settings.models_dir}/jinaai--jina-reranker-v2-base-multilingual")
    model_file = model_path / "onnx" / "model_int8.onnx"

    if not model_file.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_file}")

    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    options.inter_op_num_threads = 4
    options.intra_op_num_threads = 4

    session = InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
    logger.info(f"Loaded ONNX reranker from {model_file}")
    return session


def get_reranker_tokenizer() -> Tokenizer:
    model_path = Path(f"{settings.models_dir}/jinaai--jina-reranker-v2-base-multilingual")
    tokenizer_file = model_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    logger.info(f"Loaded reranker tokenizer with {tokenizer.get_vocab_size()} tokens")
    return tokenizer
