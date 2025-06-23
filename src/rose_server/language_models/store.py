"""Store for LanguageModel entity operations."""

from typing import List, Optional

from sqlalchemy import select

from rose_server.database import get_session
from rose_server.entities.language_models import LanguageModel


async def create(
    id: str,
    path: Optional[str] = None,
    base_model: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    name: Optional[str] = None,
) -> LanguageModel:
    """Register a new language model."""
    model = LanguageModel(
        id=id,
        name=name,
        path=path,
        base_model=base_model,
        hf_model_name=hf_model_name,
    )

    async with get_session() as session:
        session.add(model)
        await session.commit()
        await session.refresh(model)
        return model


async def get(model_id: str) -> Optional[LanguageModel]:
    async with get_session(read_only=True) as session:
        result = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
        return result.scalar_one_or_none()


async def list_models(base_model: Optional[str] = None) -> List[LanguageModel]:
    async with get_session() as session:
        query = select(LanguageModel)

        if base_model:
            query = query.where(LanguageModel.base_model == base_model)

        query = query.order_by(LanguageModel.created_at.desc())
        result = await session.execute(query)

        return list(result.scalars().all())


async def list_fine_tuned() -> List[LanguageModel]:
    async with get_session(read_only=True) as session:
        result = await session.execute(
            select(LanguageModel).where(LanguageModel.base_model is not None).order_by(LanguageModel.created_at.desc())
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
