import json
import uuid
from itertools import batched
from typing import Any

from fastapi import APIRouter, Depends
from htpy.starlette import HtpyResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.dependencies import get_db_session
from rose_server.models.import_events import ImportEvent
from rose_server.schemas.importer import ImportRequest
from rose_server.views.pages.importer import render_import

router = APIRouter(prefix="/v1", tags=["import"])


@router.get("/import")
async def import_page() -> Any:
    return HtpyResponse(render_import())


@router.post("/import/messages")
async def import_messages(request: ImportRequest, session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    import_event = ImportEvent(
        import_source=request.import_source,
        imported_count=0,
        skipped_duplicates=0,
        total_records=len(request.messages),
    )

    values = []
    thread_id_mapping: dict[str, str] = {}

    for external in request.messages:
        meta = external.meta.copy() if external.meta else {}
        meta["import_source"] = request.import_source
        meta["import_at"] = import_event.created_at

        if external.thread_id not in thread_id_mapping:
            thread_id_mapping[external.thread_id] = str(uuid.uuid4())

        values.append(
            {
                "uuid": str(uuid.uuid4()),
                "thread_id": thread_id_mapping[external.thread_id],
                "role": external.role,
                "content": external.content,
                "model": external.model,
                "created_at": external.created_at,
                "import_batch_id": import_event.batch_id,
                "import_external_id": external.import_external_id,
                "meta": json.dumps(meta),
            }
        )

    stmt = text("""
        INSERT OR IGNORE INTO messages (
            uuid, thread_id, role, content, model, created_at,
            import_batch_id, import_external_id, meta
        ) VALUES (
            :uuid, :thread_id, :role, :content, :model, :created_at,
            :import_batch_id, :import_external_id, :meta
        )
    """)

    for chunk in batched(values, 100):
        await session.execute(stmt, list(chunk))

    count_result = await session.execute(
        text("SELECT COUNT(*) FROM messages WHERE import_batch_id = :batch_id"),
        {"batch_id": import_event.batch_id},
    )
    imported_count = count_result.scalar_one()
    skipped_duplicates = import_event.total_records - imported_count

    import_event.imported_count = imported_count
    import_event.skipped_duplicates = skipped_duplicates

    session.add(import_event)

    return {
        "imported": imported_count,
        "skipped_duplicates": skipped_duplicates,
    }
