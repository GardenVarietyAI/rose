import json
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.models.messages import Message
from rose_server.schemas.importer import Conversation, ImportResponse


def parse_jsonl(content: str) -> list[Conversation]:
    lines = content.strip().split("\n")
    conversations: list[Conversation] = []

    for line in lines:
        data = json.loads(line)
        conversations.append(Conversation.model_validate(data))

    return conversations


async def import_conversations(
    conversations: list[Conversation],
    import_source: str,
    session: AsyncSession,
) -> ImportResponse:
    import_at = int(time.time())
    imported_count = 0

    for conv in conversations:
        thread_id = str(uuid.uuid4())
        base_timestamp = import_at

        non_system_messages = [msg for msg in conv.messages if msg.role != "system"]
        for idx, msg in enumerate(non_system_messages):
            message = Message(
                uuid=str(uuid.uuid4()),
                thread_id=thread_id,
                role=msg.role,
                content=msg.content,
                model=None,
                created_at=base_timestamp + idx,
                meta={
                    "import_source": import_source,
                    "import_at": import_at,
                },
            )
            session.add(message)
            imported_count += 1

    return ImportResponse(
        imported_count=imported_count,
        conversations_count=len(conversations),
    )
