"""Database storage for run steps."""

import logging
import time
from typing import Any, Dict, List, Optional

from sqlmodel import select

from rose_server.database import RunStep, get_session

logger = logging.getLogger(__name__)


async def create_run_step(step: RunStep) -> RunStep:
    """Create a new run step."""
    async with get_session() as session:
        session.add(step)
        await session.commit()
        await session.refresh(step)
        return step


async def update_run_step(
    step_id: str,
    *,
    status: Optional[str] = None,
    step_details: Optional[Dict[str, Any]] = None,
    last_error: Optional[Dict[str, Any]] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> Optional[RunStep]:
    """Update a run step."""
    async with get_session() as session:
        step = await session.get(RunStep, step_id)
        if not step:
            return None

        if status is not None:
            step.status = status

        if step_details is not None:
            step.step_details = step_details

        if last_error is not None:
            step.last_error = last_error

        if usage is not None:
            step.usage = usage

        # Update timestamp fields based on status
        current_time = int(time.time())
        if status == "completed" and not step.completed_at:
            step.completed_at = current_time
        elif status == "failed" and not step.failed_at:
            step.failed_at = current_time
        elif status == "cancelled" and not step.cancelled_at:
            step.cancelled_at = current_time
        elif status == "expired" and not step.expired_at:
            step.expired_at = current_time

        session.add(step)
        await session.commit()
        await session.refresh(step)

        return step


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
