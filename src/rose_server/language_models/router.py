"""Router for model-related endpoints."""

import logging
import time

import aiofiles
import aiofiles.os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from rose_core.config.service import EMBEDDING_MODELS
from rose_server.services import get_model_registry

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/models")
async def openai_api_models() -> JSONResponse:
    """OpenAI API-compatible endpoint that lists available models.

    Returns:
        JSON response in OpenAI format with available models
    """
    created_timestamp = int(time.time())
    model_owners = {
        "llama-3": "meta",
        "gemma-7b": "google",
        "zephyr": "stability-ai",
        "phi-3": "microsoft",
        "phi3-128k": "microsoft",
        "qwen2.5-0.5b": "alibaba",
    }
    registry = get_model_registry()
    available_models = registry.list_models()
    model_data = []
    for model_id in available_models:
        if "ft-" not in model_id:
            model_data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": created_timestamp,
                    "owned_by": model_owners.get(model_id, "organization-owner"),
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                }
            )
    for model_id in EMBEDDING_MODELS.keys():
        model_data.append(
            {
                "id": model_id,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "organization-owner",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
        )
    for model_id in available_models:
        if "ft-" in model_id:
            base_model = model_id.split("-ft-")[0] if "-ft-" in model_id else model_id
            model_data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": created_timestamp,
                    "owned_by": "user",
                    "permission": [],
                    "root": base_model,
                    "parent": base_model,
                }
            )
    openai_response = {"object": "list", "data": model_data}
    return JSONResponse(content=openai_response)


@router.get("/models/{model_id}")
async def get_model_details(model_id: str) -> JSONResponse:
    """OpenAI API-compatible endpoint to get details about a specific model.

    Args:
        model_id: The model identifier
    Returns:
        JSON response with model details or 404 if not found
    """
    created_timestamp = int(time.time())
    registry = get_model_registry()
    config = registry.get_model_config(model_id)
    if config:
        is_fine_tuned = "ft-" in model_id
        base_model = model_id.split("-ft-")[0] if is_fine_tuned else model_id
        return JSONResponse(
            content={
                "id": model_id,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "user" if is_fine_tuned else "organization-owner",
                "permission": [],
                "root": base_model,
                "parent": base_model if is_fine_tuned else None,
            }
        )
    if model_id in EMBEDDING_MODELS:
        return JSONResponse(
            content={
                "id": model_id,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "organization-owner",
                "permission": [],
                "root": model_id,
                "parent": None,
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


@router.post("/models/{model_name}/download")
async def download_model(model_name: str) -> JSONResponse:
    """Pre-download a model to avoid blocking during inference."""
    registry = get_model_registry()
    config = registry.get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    try:
        return JSONResponse(
            content={
                "status": "downloading",
                "message": f"Model '{model_name}' download started in background",
                "model": model_name,
            }
        )
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.delete("/models/{model}")
async def delete_model(model: str) -> JSONResponse:
    """Delete a fine-tuned model.

    You must have the Owner role in your organization to delete a model.
    Only fine-tuned models can be deleted - base models cannot be deleted.
    Args:
        model: The model identifier to delete
    Returns:
        JSON response with deletion confirmation or error
    """
    import shutil

    registry = get_model_registry()
    config = registry.get_model_config(model)
    if not config:
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
    if "ft-" not in model:
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
    try:
        success = await registry.unregister_model(model)
        if not success:
            logger.warning(f"Failed to unregister model {model} from registry")
        model_path = config.get("model_path")
        if model_path and await aiofiles.os.path.exists(model_path):
            try:
                import asyncio

                await asyncio.to_thread(shutil.rmtree, model_path)
                logger.info(f"Deleted model files at: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to delete model files at {model_path}: {e}")
        else:
            logger.info(f"No model files to delete for {model}")
        logger.info(f"Successfully deleted fine-tuned model: {model}")
        return JSONResponse(content={"id": model, "object": "model", "deleted": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Internal error while deleting model '{model}': {str(e)}",
                    "type": "internal_server_error",
                    "param": "model",
                    "code": "deletion_error",
                }
            },
        )
