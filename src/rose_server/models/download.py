"""Model downloading functionality using HuggingFace Hub."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import snapshot_download

from rose_core.config.settings import settings

logger = logging.getLogger(__name__)


def get_models_directory() -> Path:
    """Get the local directory for storing downloaded models."""
    models_dir = Path(settings.data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


async def download_model_async(model_name: str, force_download: bool = False) -> Dict[str, str]:
    """Download a model to local storage using snapshot_download.

    Args:
        model_name: HuggingFace model identifier (e.g., "microsoft/phi-2")
        force_download: Force re-download even if model exists locally

    Returns:
        Dict with download status and local path
    """
    models_dir = get_models_directory()

    # Convert model name to directory name (replace / with --)
    safe_model_name = model_name.replace("/", "--")
    local_dir = models_dir / safe_model_name

    # Check if already downloaded
    if local_dir.exists() and not force_download:
        logger.info(f"Model {model_name} already exists at {local_dir}")
        return {
            "status": "exists",
            "message": f"Model {model_name} already downloaded",
            "path": str(local_dir),
        }

    try:
        logger.info(f"Starting download of {model_name} to {local_dir}")

        # Run snapshot_download in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        local_path = await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=model_name,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,  # Download actual files, not symlinks
                resume_download=True,
                token=None,  # Use public models only for now
            ),
        )

        logger.info(f"Successfully downloaded {model_name} to {local_path}")

        return {
            "status": "downloaded",
            "message": f"Model {model_name} successfully downloaded",
            "path": local_path,
        }

    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise


def get_local_model_path(model_name: str) -> Optional[Path]:
    """Get the local path for a downloaded model if it exists."""
    models_dir = get_models_directory()
    safe_model_name = model_name.replace("/", "--")
    local_dir = models_dir / safe_model_name

    if local_dir.exists():
        return local_dir
    return None
