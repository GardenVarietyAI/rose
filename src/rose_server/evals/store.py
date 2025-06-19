"""Store for Eval entity operations."""

import time
from typing import Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import run_in_session
from ..entities.evals import Eval


class EvalStore:
    """Store for evaluation definition operations."""

    async def create(
        self,
        id: str,
        name: Optional[str] = None,
        data_source_config: Optional[Dict] = None,
        testing_criteria: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> Eval:
        """Create a new evaluation definition."""
        eval = Eval(
            id=id,
            name=name,
            data_source_config=data_source_config or {},
            testing_criteria=testing_criteria or [],
            created_at=int(time.time()),
            meta=metadata,
        )

        async def _create(session: AsyncSession) -> Eval:
            session.add(eval)
            await session.commit()
            await session.refresh(eval)
            return eval

        return await run_in_session(_create)

    async def get(self, eval_id: str) -> Optional[Eval]:
        """Get evaluation definition by ID."""

        async def _get(session: AsyncSession) -> Optional[Eval]:
            result = await session.execute(select(Eval).where(Eval.id == eval_id))  # type: ignore[arg-type]
            return result.scalar_one_or_none()

        return await run_in_session(_get)

    async def list(self, limit: int = 20) -> List[Eval]:
        """List evaluation definitions."""

        async def _list(session: AsyncSession) -> List[Eval]:
            result = await session.execute(select(Eval).order_by(Eval.created_at.desc()).limit(limit))  # type: ignore[attr-defined]
            return list(result.scalars().all())

        return await run_in_session(_list)

    async def delete(self, eval_id: str) -> bool:
        """Delete an evaluation definition."""

        async def _delete(session: AsyncSession) -> bool:
            eval = await session.get(Eval, eval_id)
            if not eval:
                return False
            await session.execute(delete(Eval).where(Eval.id == eval_id))  # type: ignore[arg-type]
            await session.commit()
            return True

        return await run_in_session(_delete)
