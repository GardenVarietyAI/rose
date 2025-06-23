"""Store for EvalRun entity operations."""

import time
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from rose_server.database import get_session
from rose_server.entities.evals import EvalRun


async def create_eval_run(
    id: str,
    eval_id: str,
    name: str,
    model: str,
    data_source: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
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

    async with get_session() as session:
        session.add(eval_run)
        await session.commit()
        await session.refresh(eval_run)
        return eval_run


async def get_eval_run(run_id: str) -> Optional[EvalRun]:
    """Get evaluation run by ID."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(EvalRun).where(EvalRun.id == run_id))
        return result.scalar_one_or_none()


async def list_eval_runs(eval_id: Optional[str] = None, limit: int = 20) -> List[EvalRun]:
    """List evaluation runs, optionally filtered by eval_id."""
    async with get_session(read_only=True) as session:
        query = select(EvalRun)
        if eval_id:
            query = query.where(EvalRun.eval_id == eval_id)
        query = query.order_by(EvalRun.created_at.desc()).limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())


async def update_eval_run_results(run_id: str, results: Dict[str, Any]) -> None:
    """Update evaluation run with results."""
    async with get_session() as session:
        run = await session.get(EvalRun, run_id)
        if run:
            run.status = "completed"
            run.results = results
            run.completed_at = int(time.time())
            await session.commit()


async def update_eval_run_error(run_id: str, error: str) -> None:
    """Update evaluation run with error."""
    async with get_session() as session:
        run = await session.get(EvalRun, run_id)
        if run:
            run.status = "failed"
            run.error_message = error
            run.completed_at = int(time.time())
            await session.commit()


async def update_eval_run_status(run_id: str, status: str) -> None:
    async with get_session() as session:
        run = await session.get(EvalRun, run_id)
        if run:
            run.status = status
            if status == "running":
                run.started_at = int(time.time())
            await session.commit()
