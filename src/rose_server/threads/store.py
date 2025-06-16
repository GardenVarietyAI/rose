"""SQLModel-based storage for threads and messages with ChromaDB integration."""
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from openai.types.beta.threads import Text, TextContentBlock
from sqlalchemy import delete, func
from sqlmodel import select

from ..database import (
    Message as MessageDB,
)
from ..database import (
    Thread as ThreadDB,
)
from ..database import (
    current_timestamp,
    run_in_session,
)
from ..schemas.threads import Thread, ThreadMessage

logger = logging.getLogger(__name__)

class ThreadStore:
    """SQLModel-based storage for threads and messages."""

    _instance = None

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super(ThreadStore, cls).__new__(cls)
            cls._instance.chroma_client = None
        return cls._instance

    def set_chroma_client(self, client):
        """Set the ChromaDB client to use for embeddings.

        Args:
            client: ChromaDB client instance
        """
        self.chroma_client = client
        logger.info("ThreadStore configured with ChromaDB client")

    def _to_openai_thread(self, db_thread: ThreadDB) -> Thread:
        """Convert database thread to OpenAI-compatible Thread model."""
        metadata = db_thread.meta if hasattr(db_thread, "meta") and db_thread.meta else {}
        return Thread(
            id=db_thread.id,
            object="thread",
            created_at=db_thread.created_at,
            metadata=metadata,
            tool_resources=db_thread.tool_resources
            if hasattr(db_thread, "tool_resources") and db_thread.tool_resources
            else None,
        )

    def _to_openai_message(self, db_message: MessageDB) -> ThreadMessage:
        """Convert database message to OpenAI-compatible ThreadMessage model."""
        content = db_message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = [TextContentBlock(type="text", text=Text(value=content, annotations=[]))]
        metadata = db_message.meta if hasattr(db_message, "meta") and db_message.meta else {}
        return ThreadMessage(
            id=db_message.id,
            object="thread.message",
            created_at=db_message.created_at,
            thread_id=db_message.thread_id,
            role=db_message.role,
            content=content,
            metadata=metadata,
            assistant_id=db_message.assistant_id,
            run_id=db_message.run_id,
            status="completed",
            completed_at=db_message.created_at,
            incomplete_at=None,
            incomplete_details=None,
            attachments=[],
        )

    async def create_thread(self, metadata: Dict = None) -> Thread:
        """Create a new thread."""
        thread_id = f"thread_{uuid.uuid4().hex}"

        async def operation(session):
            db_thread = ThreadDB(
                id=thread_id,
                created_at=current_timestamp(),
                meta=metadata or {},
                tool_resources={},
            )
            session.add(db_thread)
            await session.commit()
            await session.refresh(db_thread)
            return self._to_openai_thread(db_thread)
        result = await run_in_session(operation)
        logger.info(f"Created thread: {thread_id}")
        return result

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID."""

        async def operation(session):
            db_thread = await session.get(ThreadDB, thread_id)
            if db_thread:
                return self._to_openai_thread(db_thread)
            return None
        return await run_in_session(operation, read_only=True)

    async def update_thread(self, thread_id: str, metadata: Dict) -> Optional[Thread]:
        """Update thread metadata."""

        async def operation(session):
            db_thread = await session.get(ThreadDB, thread_id)
            if not db_thread:
                return None
            if metadata:
                db_thread.meta = metadata
            session.add(db_thread)
            await session.commit()
            await session.refresh(db_thread)
            return self._to_openai_thread(db_thread)
        result = await run_in_session(operation)
        logger.info(f"Updated thread: {thread_id}")
        return result

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all its messages."""

        async def operation(session):
            db_thread = await session.get(ThreadDB, thread_id)
            if not db_thread:
                return False
            message_count_result = await session.execute(
                select(func.count(MessageDB.id)).where(MessageDB.thread_id == thread_id)
            )
            message_count = message_count_result.scalar()
            await session.execute(delete(MessageDB).where(MessageDB.thread_id == thread_id))
            await session.delete(db_thread)
            await session.commit()
            logger.info(f"Deleted thread: {thread_id} and {message_count} messages")
            return True
        return await run_in_session(operation)

    async def list_threads(self, limit: int = 20, order: str = "desc") -> List[Thread]:
        """List all threads."""

        async def operation(session):
            statement = select(ThreadDB)
            if order == "desc":
                statement = statement.order_by(ThreadDB.created_at.desc())
            else:
                statement = statement.order_by(ThreadDB.created_at.asc())
            statement = statement.limit(limit)
            db_threads = (await session.execute(statement)).scalars().all()
            return [self._to_openai_thread(t) for t in db_threads]
        return await run_in_session(operation, read_only=True)

    async def create_message(
        self,
        thread_id: str,
        role: str,
        content: List[Dict],
        metadata: Dict = None,
        assistant_id: str = None,
        run_id: str = None,
        enable_embedding: bool = False,
        embedding_func: Optional[Callable[[str], Dict[str, Any]]] = None,
        collection_name: str = "thread_messages",
    ) -> Optional[ThreadMessage]:
        """Create a new message in a thread with optional vector embedding."""

        async def operation(session):
            db_thread = await session.get(ThreadDB, thread_id)
            if not db_thread:
                return None
            message_id = f"msg_{uuid.uuid4().hex}"
            created_at = current_timestamp()
            formatted_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    formatted_content.append(
                        TextContentBlock(type="text", text=Text(value=item.get("text", ""), annotations=[]))
                    )
                else:
                    formatted_content.append(item)
            content_json = json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in formatted_content])
            model_used = metadata.get("model_used") if metadata else None
            token_count = None
            response_time_ms = None
            if metadata:
                if "token_count" in metadata:
                    try:
                        token_count = int(metadata["token_count"])
                    except (ValueError, TypeError):
                        token_count = None
                if "response_time_ms" in metadata:
                    try:
                        response_time_ms = int(metadata["response_time_ms"])
                    except (ValueError, TypeError):
                        response_time_ms = None
            finish_reason = metadata.get("finish_reason") if metadata else None
            db_message = MessageDB(
                id=message_id,
                thread_id=thread_id,
                role=role,
                content=content_json,
                created_at=created_at,
                assistant_id=assistant_id,
                run_id=run_id,
                file_ids=[],
                model_used=model_used,
                token_count=token_count,
                response_time_ms=response_time_ms,
                finish_reason=finish_reason,
            )
            session.add(db_message)
            session.add(db_thread)
            await session.commit()
            await session.refresh(db_message)
            message = ThreadMessage(
                id=message_id,
                created_at=created_at,
                thread_id=thread_id,
                role=role,
                content=formatted_content,
                metadata=metadata or {},
                assistant_id=assistant_id,
                run_id=run_id,
                object="thread.message",
                status="completed",
                completed_at=created_at,
                incomplete_at=None,
                incomplete_details=None,
                attachments=[],
            )
            return message
        message = await run_in_session(operation)
        if enable_embedding and self.chroma_client and embedding_func:
            await self._embed_message_to_chroma(message, embedding_func, collection_name)
        logger.info(f"Created message: {message.id} in thread: {thread_id}")
        return message

    async def _embed_message_to_chroma(
        self, message: ThreadMessage, embedding_func: Callable[[str], Dict[str, Any]], collection_name: str
    ):
        """Embed a message directly to ChromaDB."""
        try:
            text_content = ""
            for content_item in message.content:
                if content_item.type == "text":
                    text_content += content_item.text.value + "\n"
            if not text_content.strip():
                logger.debug(f"No text content to embed for message {message.id}")
                return
            embedding_result = embedding_func(text_content.strip())
            embedding_vector = embedding_result["data"][0]["embedding"]
            metadata = {
                "thread_id": message.thread_id,
                "message_id": message.id,
                "role": message.role,
                "created_at": message.created_at,
                "text": text_content.strip(),
            }
            if message.assistant_id:
                metadata["assistant_id"] = message.assistant_id
            if message.run_id:
                metadata["run_id"] = message.run_id
            metadata.update(message.metadata)
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except ValueError:
                collection = self.chroma_client.create_collection(
                    name=collection_name, metadata={"purpose": "thread_message_embeddings"}
                )
                logger.info(f"Created ChromaDB collection: {collection_name}")
            collection.add(embeddings=[embedding_vector], ids=[message.id], metadatas=[metadata])
            logger.info(f"Embedded message {message.id} from thread {message.thread_id} to ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to embed message {message.id}: {str(e)}")

    async def get_messages(self, thread_id: str, limit: int = 20, order: str = "desc") -> List[ThreadMessage]:
        """Get messages for a thread."""

        async def operation(session):
            statement = select(MessageDB).where(MessageDB.thread_id == thread_id)
            if order == "desc":
                statement = statement.order_by(MessageDB.created_at.desc())
            else:
                statement = statement.order_by(MessageDB.created_at.asc())
            statement = statement.limit(limit)
            db_messages = (await session.execute(statement)).scalars().all()
            return [self._to_openai_message(m) for m in db_messages]
        return await run_in_session(operation, read_only=True)

    async def get_message(self, thread_id: str, message_id: str) -> Optional[ThreadMessage]:
        """Get a specific message."""

        async def operation(session):
            db_message = await session.get(MessageDB, message_id)
            if db_message and db_message.thread_id == thread_id:
                return self._to_openai_message(db_message)
            return None
        return await run_in_session(operation, read_only=True)

    async def update_message(self, thread_id: str, message_id: str, metadata: Dict) -> Optional[ThreadMessage]:
        """Update message metadata."""

        async def operation(session):
            db_message = await session.get(MessageDB, message_id)
            if not db_message or db_message.thread_id != thread_id:
                return None
            if metadata:
                db_message.meta = metadata
            session.add(db_message)
            await session.commit()
            await session.refresh(db_message)
            return self._to_openai_message(db_message)
        result = await run_in_session(operation)
        logger.info(f"Updated message: {message_id}")
        return result

    async def get_stats(self) -> Dict:
        """Get store statistics."""

        async def operation(session):
            thread_count_result = await session.execute(select(func.count(ThreadDB.id)))
            thread_count = thread_count_result.scalar()
            message_count_result = await session.execute(select(func.count(MessageDB.id)))
            message_count = message_count_result.scalar()
            return {"total_threads": thread_count, "total_messages": message_count, "storage_type": "sqlite"}
        return await run_in_session(operation, read_only=True)