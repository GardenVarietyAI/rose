"""Store for EvalRun entity operations."""

import time
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import run_in_session
from ...entities.evals import EvalRun


class EvalRunStore:
    """Store for evaluation run operations."""

    async def create(
        self,
        id: str,
        eval_id: str,
        name: str,
        model: str,
        data_source: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> EvalRun:
        """Create a new evaluation run."""
        eval_run = EvalRun(
            id=id,
            eval_id=eval_id,
            name=name,
            model=model,
            status="queued",
            created_at=int(time.time()),
            data_source=data_source,
            meta=metadata,
        )

        async def _create(session: AsyncSession) -> EvalRun:
            session.add(eval_run)
            await session.commit()
            await session.refresh(eval_run)
            return eval_run

        return await run_in_session(_create)

    async def get(self, run_id: str) -> Optional[EvalRun]:
        """Get evaluation run by ID."""

        async def _get(session: AsyncSession) -> Optional[EvalRun]:
            result = await session.execute(select(EvalRun).where(EvalRun.id == run_id))  # type: ignore[arg-type]
            return result.scalar_one_or_none()

        return await run_in_session(_get)

    async def list(self, eval_id: Optional[str] = None, limit: int = 20) -> List[EvalRun]:
        """List evaluation runs, optionally filtered by eval_id."""

        async def _list(session: AsyncSession) -> List[EvalRun]:
            query = select(EvalRun)
            if eval_id:
                query = query.where(EvalRun.eval_id == eval_id)  # type: ignore[arg-type]
            query = query.order_by(EvalRun.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
            result = await session.execute(query)
            return list(result.scalars().all())

        return await run_in_session(_list)

    async def update_results(self, run_id: str, results: Dict) -> None:
        """Update evaluation run with results."""

        async def _update(session: AsyncSession) -> None:
            run = await session.get(EvalRun, run_id)
            if run:
                run.status = "completed"
                run.results = results
                run.completed_at = int(time.time())
                await session.commit()

        await run_in_session(_update)

    async def update_error(self, run_id: str, error: str) -> None:
        """Update evaluation run with error."""

        async def _update(session: AsyncSession) -> None:
            run = await session.get(EvalRun, run_id)
            if run:
                run.status = "failed"
                run.error_message = error
                run.completed_at = int(time.time())
                await session.commit()

        await run_in_session(_update)

    async def update_status(self, run_id: str, status: str) -> None:
        """Update evaluation run status."""

        async def _update(session: AsyncSession) -> None:
            run = await session.get(EvalRun, run_id)
            if run:
                run.status = status
                if status == "running":
                    run.started_at = int(time.time())
                await session.commit()

        await run_in_session(_update)
