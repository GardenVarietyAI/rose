"""Database storage for run steps."""

import logging
from typing import List, Optional

from sqlmodel import select

from rose_server.database import RunStep, get_session

logger = logging.getLogger(__name__)


async def get_run_step(run_id: str, step_id: str) -> Optional[RunStep]:
    """Get a specific run step."""
    async with get_session(read_only=True) as session:
        step = await session.get(RunStep, step_id)
        if step and step.run_id == run_id:
            return step
        return None


async def list_run_steps(run_id: str, limit: int = 20, order: str = "desc") -> List[RunStep]:
    """List steps for a run."""
    async with get_session(read_only=True) as session:
        statement = select(RunStep).where(RunStep.run_id == run_id)

        if order == "desc":
            statement = statement.order_by(RunStep.created_at.desc())
        else:
            statement = statement.order_by(RunStep.created_at.asc())

        statement = statement.limit(limit)
        steps = (await session.execute(statement)).scalars().all()

        return list(steps)
