"""Model loading implementation for the inference server - no caching."""

import asyncio
import gc
import logging
from typing import Any, Callable, Dict

import psutil

from rose_core.models import get_tokenizer, load_hf_model

logger = logging.getLogger(__name__)


def get_memory_usage() -> tuple[float, float]:
    """Get current memory usage as (used_gb, percent).

    Returns:
        Tuple of (used memory in GB, percentage used)
    """
    memory = psutil.virtual_memory()
    used_gb = (memory.total - memory.available) / (1024**3)
    percent = memory.percent
    return used_gb, percent


def check_memory_pressure() -> bool:
    """Check if system is under memory pressure.

    Returns:
        True if memory usage is above 80%
    """
    _, percent = get_memory_usage()
    if percent > 80:
        logger.warning(f"High memory usage detected: {percent:.1f}%")
        return True
    return False


def log_memory_status(context: str = "") -> None:
    """Log current memory status with optional context."""
    used_gb, percent = get_memory_usage()
    logger.info(f"Memory usage{f' ({context})' if context else ''}: {used_gb:.1f}GB ({percent:.1f}%)")


class ModelRunner:
    """Manages model loading for inference without caching."""

    def __init__(self) -> None:
        self._load_lock = asyncio.Lock()

    async def load_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model and tokenizer. No caching - loads fresh each time."""
        async with self._load_lock:
            logger.info(f"Loading model: {model_name}")
            log_memory_status(f"before loading {model_name}")

            loader: Callable[..., Any] = model_config.get("loader", load_hf_model)

            try:
                # Use model_path if available (for custom/fine-tuned models), otherwise use model_name
                model_id = model_config.get("model_path") or model_config.get("model_name", model_name)

                # Load the model based on loader type
                if loader.__name__ == "load_peft_model":
                    # PEFT models require model_path
                    model = loader(
                        model_id=model_id,
                        model_path=model_config.get("model_path"),
                        torch_dtype=model_config.get("torch_dtype"),
                    )
                else:
                    # Regular HF models only need model_id
                    model = loader(
                        model_id=model_id,
                        torch_dtype=model_config.get("torch_dtype"),
                    )

                # Load tokenizer
                tokenizer = get_tokenizer(model_id)

                # Return model info
                model_info = {
                    "name": model_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": model_config,
                    "device": str(model.device) if hasattr(model, "device") else "cpu",
                    "dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "unknown",
                }

                log_memory_status(f"after loading {model_name}")

                return model_info

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    def cleanup(self) -> None:
        """Clean up any resources on exit."""
        logger.info("Cleanup called (no-op in non-cached mode)")
        gc.collect()


# Global instance
_runner = ModelRunner()


# Export functions for compatibility
async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer. No caching - loads fresh each time."""
    return await _runner.load_model(model_name, model_config)


def cleanup_models() -> None:
    """Clean up any resources on exit."""
    _runner.cleanup()
