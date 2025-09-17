import logging
from pathlib import Path

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from transformers import AutoTokenizer

from rose_server.config.settings import settings

logger = logging.getLogger(__name__)


def get_reranker_session() -> InferenceSession:
    model_path = Path(f"{settings.models_dir}/Qwen3-Reranker-0.6B-ONNX")
    model_file = model_path / "model.onnx"

    if not model_file.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_file}")

    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    options.inter_op_num_threads = 4
    options.intra_op_num_threads = 4

    session = InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
    logger.info(f"Loaded ONNX reranker from {model_file}")
    return session


def get_reranker_tokenizer() -> AutoTokenizer:
    model_path = Path(f"{settings.models_dir}/Qwen3-Reranker-0.6B-ONNX")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    logger.info(f"Loaded reranker tokenizer with {len(tokenizer)} tokens")
    return tokenizer
