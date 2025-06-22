import time
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select

from rose_server.database import get_session
from rose_server.entities.evals import Eval


async def create_eval(
    id: str,
    name: Optional[str] = None,
    data_source_config: Optional[Dict[str, Any]] = None,
    testing_criteria: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
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

    async with get_session() as session:
        session.add(eval)
        await session.commit()
        await session.refresh(eval)
        return eval


async def get_eval(eval_id: str) -> Optional[Eval]:
    """Get evaluation definition by ID."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(Eval).where(Eval.id == eval_id))
        return result.scalar_one_or_none()


async def list_evals(limit: int = 20) -> List[Eval]:
    """List evaluation definitions."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(Eval).order_by(Eval.created_at.desc()).limit(limit))
        return list(result.scalars().all())


async def delete_eval(eval_id: str) -> bool:
    """Delete an evaluation definition."""
    async with get_session() as session:
        eval = await session.get(Eval, eval_id)
        if not eval:
            return False
        await session.execute(delete(Eval).where(Eval.id == eval_id))
        await session.commit()
        return True
