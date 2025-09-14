"""Router for model-related endpoints."""

import asyncio
import logging
import shutil
from pathlib import Path

import aiofiles
import aiofiles.os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from rose_server.config.settings import settings
from rose_server.models.store import (
    create as create_language_model,
    delete as delete_language_model,
    get as get_language_model,
    list_all,
)
from rose_server.schemas.models import ModelCreateRequest, ModelResponse

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/models")
async def index() -> JSONResponse:
    models = await list_all()
    return JSONResponse(content={"object": "list", "data": [model.model_dump() for model in models]})


@router.post("/models", status_code=201)
async def create(request: ModelCreateRequest) -> ModelResponse:
    model = await create_language_model(**request.model_dump())
    logger.info(f"Created model: {model.id} ({model.model_name})")
    return ModelResponse(**model.model_dump())


@router.get("/models/{model_id}")
async def show(model_id: str) -> ModelResponse:
    model = await get_language_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"The model '{model_id}' does not exist")
    return ModelResponse(**model.model_dump())


@router.delete("/models/{model}")
async def remove(model: str) -> JSONResponse:
    success = await delete_language_model(model)
    if success:
        model_obj = await get_language_model(model)
        if model and model_obj.path and model_obj.is_fine_tuned:
            model_path = Path(settings.data_dir) / model_obj.path
            file_path_exists = await aiofiles.os.path.exists(model_path)
            if file_path_exists:
                await asyncio.to_thread(shutil.rmtree, str(model_path))
                logger.info(f"Deleted model files at: {model_path}")
    else:
        raise HTTPException(status_code=403, detail=f"Failed to delete model {model}")

    logger.info(f"Successfully deleted model: {model}")
    return JSONResponse(content={"id": model, "object": "model", "deleted": success})
