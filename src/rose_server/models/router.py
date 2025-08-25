"""Router for model-related endpoints."""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import aiofiles
import aiofiles.os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from rose_server.config.settings import settings
from rose_server.inference.client import InferenceClient
from rose_server.models.store import (
    create as create_language_model,
    delete as delete_language_model,
    get as get_language_model,
    list_all,
)
from rose_server.schemas.models import ModelCreateRequest

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/models")
async def openai_api_models() -> JSONResponse:
    """OpenAI API-compatible endpoint that lists available models.

    Returns:
        JSON response in OpenAI format with available models
    """
    models = await list_all()
    model_data = []

    # Add language models from database
    for model in models:
        model_data.append(
            {
                "id": model.id,
                "object": "model",
                "created": model.created_at,
                "owned_by": model.owned_by,
                "permission": json.loads(model.permissions) if model.permissions else [],
                "parent": model.parent,
            }
        )
    openai_response = {"object": "list", "data": model_data}
    return JSONResponse(content=openai_response)


@router.get("/models/{model_id}")
async def get_model_details(model_id: str) -> JSONResponse:
    """Get details about a specific model."""
    model = await get_language_model(model_id)
    if model:
        return JSONResponse(
            content={
                "id": model.id,
                "object": "model",
                "created": model.created_at,
                "owned_by": model.owned_by,
                "permission": json.loads(model.permissions) if model.permissions else [],
                "parent": model.parent,
                # Extra field for internal use
                "model_name": model.model_name,
                "lora_target_modules": model.get_lora_modules(),
                "quantization": model.quantization,
            }
        )
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"The model '{model_id}' does not exist",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        },
    )


@router.delete("/models/{model}")
async def delete_model(model: str) -> JSONResponse:
    """Delete a fine-tuned model."""
    model_obj = await get_language_model(model)
    if not model_obj:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    if not model_obj.is_fine_tuned:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": f"Cannot delete base model '{model}'. Only fine-tuned models can be deleted.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_deletable",
                }
            },
        )

    # Delete from database
    await delete_language_model(model)

    # Delete model files if they exist
    if model_obj.path:
        model_path = Path(settings.data_dir) / model_obj.path
        file_path_exists = await aiofiles.os.path.exists(model_path)
        if file_path_exists:
            await asyncio.to_thread(shutil.rmtree, str(model_path))
            logger.info(f"Deleted model files at: {model_path}")

    logger.info(f"Successfully deleted fine-tuned model: {model}")
    return JSONResponse(content={"id": model, "object": "model", "deleted": True})


@router.post("/models/evict")
async def evict_cached_models() -> Dict[str, Any]:
    """Evict all cached models from the inference service to free memory."""
    try:
        client = InferenceClient()
        client.flush_model()

        return {"status": "success", "message": "Models evicted from cache"}

    except Exception as e:
        logger.error(f"Unexpected error during model eviction: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal server error during eviction",
                    "type": "server_error",
                    "code": "eviction_error",
                }
            },
        )


@router.post("/models", status_code=201)
async def create_model(request: ModelCreateRequest) -> Dict[str, Any]:
    """Create a new model configuration."""

    model = await create_language_model(
        model_name=request.model_name,
        name=request.name,
        temperature=request.temperature,
        top_p=request.top_p,
        memory_gb=request.memory_gb,
        timeout=request.timeout,
        lora_modules=request.lora_target_modules,
        quantization=request.quantization,
    )

    logger.info(f"Created model: {model.id} ({model.model_name})")

    return {
        "id": model.id,
        "object": "model",
        "created": model.created_at,
        "owned_by": model.owned_by,
        "permission": json.loads(model.permissions) if model.permissions else [],
        "parent": model.parent,
    }
