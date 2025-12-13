import os
from typing import TypedDict


class ModelConfig(TypedDict):
    id: str
    path: str
    n_gpu_layers: int
    n_ctx: int


HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))

MODELS: dict[str, ModelConfig] = {
    "chat": {
        "id": "Qwen/Qwen3-0.6B-GGUF",
        "path": f"{HF_HOME}/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/*/qwen3-0.6b-q8_0.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 2048,
    },
    "embedding": {
        "id": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "path": f"{HF_HOME}/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/*/qwen3-embedding-0.6b-q8_0.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 2048,
    },
}
