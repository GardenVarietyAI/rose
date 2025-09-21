"""Router for model-related endpoints."""

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

import aiofiles
import aiofiles.os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from rose_server.database import get_session
from rose_server.entities.models import LanguageModel
from rose_server.schemas.models import ModelCreateRequest, ModelResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


def _generate_model_id(is_fine_tuned: bool, base_model: str, model_name: str, suffix: str = "") -> str:
    """Generate model ID."""
    if not is_fine_tuned:
        return model_name.replace("/", "--")

    unique_hash = uuid.uuid4().hex[:6]
    suffix_part = suffix if suffix else "default"
    flat_base_model = base_model.replace("/", "--")
    return f"ft:{flat_base_model}:user:{suffix_part}:{unique_hash}"


@router.get("/models")
async def index() -> JSONResponse:
    async with get_session(read_only=True) as session:
        result = await session.execute(select(LanguageModel).order_by(LanguageModel.created_at.desc()))
        models = list(result.scalars().all())

    return JSONResponse(
        content={"object": "list", "data": [ModelResponse(**model.model_dump()).model_dump() for model in models]}
    )


@router.post("/models", status_code=201)
async def create(req: Request, request: ModelCreateRequest) -> ModelResponse:
    is_fine_tuned = request.parent is not None
    owned_by = "user" if is_fine_tuned else "system"

    model_id = _generate_model_id(
        is_fine_tuned=is_fine_tuned,
        base_model=request.parent or request.model_name,
        model_name=request.model_name,
        suffix=request.suffix or "",
    )

    if request.path:
        model_path = request.path
    else:
        model_path = str(Path(req.app.state.settings.models_dir) / model_id)

    model = LanguageModel(
        id=model_id,
        model_name=request.model_name,
        path=model_path,
        kind=request.kind,
        is_fine_tuned=is_fine_tuned,
        temperature=request.temperature,
        top_p=request.top_p,
        timeout=request.timeout,
        owned_by=owned_by,
        parent=request.parent,
        quantization=request.quantization,
        lora_target_modules=request.lora_target_modules if request.lora_target_modules is not None else [],
    )

    async with get_session() as session:
        try:
            session.add(model)
            await session.commit()
            await session.refresh(model)
        except IntegrityError:
            # Model already exists, retrieve and return it
            await session.rollback()
            result = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
            model = result.scalar_one()

    logger.info(f"Created model: {model.id} ({model.model_name})")
    return ModelResponse(**model.model_dump())


@router.get("/models/{model_id}")
async def show(model_id: str) -> ModelResponse:
    async with get_session(read_only=True) as session:
        result = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
        model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(status_code=404, detail=f"The model '{model_id}' does not exist")
    return ModelResponse(**model.model_dump())


@router.delete("/models/{model}")
async def remove(req: Request, model: str) -> JSONResponse:
    async with get_session() as session:
        model_obj = await session.get(LanguageModel, model)
        if not model_obj:
            raise HTTPException(status_code=404, detail=f"The model does not exist: {model}")

        if not model_obj.is_fine_tuned:
            raise HTTPException(status_code=403, detail=f"Cannot delete base model: {model}")

        # Store path before deletion
        model_path = Path(req.app.state.settings.data_dir) / model_obj.path if model_obj.path else None

        await session.delete(model_obj)
        await session.commit()

    if model_path:
        file_path_exists = await aiofiles.os.path.exists(model_path)
        if file_path_exists:
            await asyncio.to_thread(shutil.rmtree, str(model_path))
            logger.info(f"Deleted model files at: {model_path}")

    logger.info(f"Successfully deleted model: {model}")
    return JSONResponse(content={"id": model, "object": "model", "deleted": True})
