"""Store for LanguageModel entity operations."""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import run_in_session
from ..entities.language_models import LanguageModel


class LanguageModelStore:
    async def create(
        self,
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

        async def _create(session: AsyncSession) -> LanguageModel:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model

        return await run_in_session(_create)

    async def get(self, model_id: str) -> Optional[LanguageModel]:
        async def _get(session: AsyncSession) -> Optional[LanguageModel]:
            result = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
            return result.scalar_one_or_none()

        return await run_in_session(_get)

    async def list(self, base_model: Optional[str] = None) -> List[LanguageModel]:
        async def _list(session: AsyncSession) -> List[LanguageModel]:
            query = select(LanguageModel)
            if base_model:
                query = query.where(LanguageModel.base_model == base_model)
            query = query.order_by(LanguageModel.created_at.desc())
            result = await session.execute(query)
            return list(result.scalars().all())

        return await run_in_session(_list)

    async def list_fine_tuned(self) -> List[LanguageModel]:
        async def _list_fine_tuned(session: AsyncSession) -> List[LanguageModel]:
            result = await session.execute(
                select(LanguageModel)
                .where(LanguageModel.base_model != None)  # noqa: E711
                .order_by(LanguageModel.created_at.desc())
            )
            return list(result.scalars().all())

        return await run_in_session(_list_fine_tuned)

    async def delete(self, model_id: str) -> bool:
        async def _delete(session: AsyncSession) -> bool:
            model = await session.get(LanguageModel, model_id)
            if not model:
                return False
            await session.delete(model)
            await session.commit()
            return True

        return await run_in_session(_delete)
