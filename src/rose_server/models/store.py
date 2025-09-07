"""Store for LanguageModel entity operations."""

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from rose_server.database import get_session
from rose_server.entities.models import LanguageModel


async def _generate_model_id(is_fine_tuned: bool, base_model: str, model_name: str, suffix: str = "") -> str:
    """Generate model ID."""
    if not is_fine_tuned:
        return model_name.replace("/", "--")

    unique_hash = uuid.uuid4().hex[:6]
    suffix_part = suffix if suffix else "default"
    flat_base_model = base_model.replace("/", "--")
    return f"ft:{flat_base_model}:user:{suffix_part}:{unique_hash}"


async def create(
    model_name: str,
    path: Optional[str] = None,
    parent: Optional[str] = None,
    kind: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout: Optional[int] = None,
    lora_modules: Optional[List[str]] = None,
    suffix: Optional[str] = None,
    quantization: Optional[str] = None,
) -> LanguageModel:
    """Register a new language model."""

    is_fine_tuned = parent is not None
    owned_by = "user" if is_fine_tuned else "system"

    # Generate appropriate ID
    model_id = await _generate_model_id(
        is_fine_tuned=is_fine_tuned,
        base_model=parent or model_name,
        model_name=model_name,
        suffix=suffix or "",
    )

    model = LanguageModel(
        id=model_id,
        model_name=model_name,
        path=path,
        kind=kind,
        is_fine_tuned=is_fine_tuned,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        owned_by=owned_by,
        parent=parent,
        quantization=quantization,
    )

    if lora_modules:
        model.set_lora_modules(lora_modules)

    async with get_session() as session:
        try:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
        except IntegrityError:
            # Model already exists, retrieve and return it
            await session.rollback()
            existing_model = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
            return existing_model.scalar_one()


async def get(model_id: str) -> Optional[LanguageModel]:
    async with get_session(read_only=True) as session:
        result = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
        return result.scalar_one_or_none()


async def list_all() -> List[LanguageModel]:
    """List all models (base + fine-tuned)."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(LanguageModel).order_by(LanguageModel.created_at.desc()))
        return list(result.scalars().all())


async def list_fine_tuned() -> List[LanguageModel]:
    async with get_session(read_only=True) as session:
        result = await session.execute(
            select(LanguageModel).where(LanguageModel.is_fine_tuned is True).order_by(LanguageModel.created_at.desc())
        )
        return list(result.scalars().all())


async def delete(model_id: str) -> bool:
    async with get_session() as session:
        model = await session.get(LanguageModel, model_id)
        if not model:
            return False

        await session.delete(model)
        await session.commit()
        return True
