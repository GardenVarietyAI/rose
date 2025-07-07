"""Store for LanguageModel entity operations."""

import hashlib
import uuid
from typing import List, Optional

from sqlalchemy import select

from rose_server.database import get_session
from rose_server.entities.models import LanguageModel


async def _generate_model_id(is_fine_tuned: bool, base_model: str, suffix: str = "") -> str:
    """Generate model ID based on OpenAI format."""
    # Generate short hash for uniqueness
    unique_hash = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:6]

    if is_fine_tuned:
        # Format: ft:{base_model}:{org}:{suffix}:{hash}
        suffix_part = suffix if suffix else "default"
        return f"ft:{base_model}:user:{suffix_part}:{unique_hash}"
    else:
        # For base models: model-id-{hash}
        return f"model-id-{unique_hash}"


async def create(
    model_name: str,
    path: Optional[str] = None,
    parent: Optional[str] = None,
    name: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    memory_gb: float = 2.0,
    timeout: Optional[int] = None,
    lora_modules: Optional[List[str]] = None,
    owned_by: str = "organization-owner",
    suffix: Optional[str] = None,
) -> LanguageModel:
    """Register a new language model."""
    is_fine_tuned = parent is not None

    # Generate appropriate ID
    model_id = await _generate_model_id(
        is_fine_tuned=is_fine_tuned, base_model=parent or model_name, suffix=suffix or ""
    )

    # Always set owned_by based on model type
    owned_by = "user" if is_fine_tuned else "system"

    model = LanguageModel(
        id=model_id,
        model_name=model_name,
        name=name,
        path=path,
        is_fine_tuned=is_fine_tuned,
        temperature=temperature,
        top_p=top_p,
        memory_gb=memory_gb,
        timeout=timeout,
        owned_by=owned_by,
        parent=parent,
    )

    # Set root to parent for fine-tuned, or self for base models
    model.root = parent or model_id

    # If no name provided, use the model_name
    if not model.name:
        model.name = model.model_name

    if lora_modules:
        model.set_lora_modules(lora_modules)

    async with get_session() as session:
        session.add(model)
        await session.commit()
        await session.refresh(model)
        return model


async def get(model_id: str) -> Optional[LanguageModel]:
    async with get_session(read_only=True) as session:
        result = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
        return result.scalar_one_or_none()


async def list_models(parent: Optional[str] = None) -> List[LanguageModel]:
    async with get_session() as session:
        query = select(LanguageModel)

        if parent:
            query = query.where(LanguageModel.parent == parent)

        query = query.order_by(LanguageModel.created_at.desc())
        result = await session.execute(query)

        return list(result.scalars().all())


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
