from typing import Any, AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from rose_server import database
from rose_server.dependencies import (
    get_db_session,
    get_openai_client,
    get_readonly_db_session,
    get_spell_checker,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel


class DummyUsage:
    def model_dump(self) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class DummyChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = DummyChoiceMessage(content)
        self.finish_reason = "stop"


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.id = "dummy-completion-id"
        self.choices = [DummyChoice(content)]
        self.usage = DummyUsage()

    def model_dump(self) -> dict[str, object]:
        return {
            "id": self.id,
            "choices": [
                {
                    "message": {"content": self.choices[0].message.content},
                    "finish_reason": self.choices[0].finish_reason,
                }
            ],
            "usage": self.usage.model_dump(),
        }


class DummyChatCompletions:
    async def create(self, *args: Any, **kwargs: Any) -> DummyResponse:
        messages = kwargs.get("messages", [])
        content = ""
        if messages and isinstance(messages, list):
            last = messages[-1]
            if isinstance(last, dict):
                content = str(last.get("content", ""))
        return DummyResponse(content)


class DummyChat:
    def __init__(self) -> None:
        self.completions = DummyChatCompletions()


class DummyOpenAI:
    def __init__(self) -> None:
        self.chat = DummyChat()


test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
    connect_args={
        "check_same_thread": False,
    },
)

test_async_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with database.get_session(test_async_session_factory) as session:
        yield session


async def override_readonly_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with database.get_session(test_async_session_factory, read_only=True) as session:
        yield session


def override_openai_client() -> DummyOpenAI:
    return DummyOpenAI()


def override_spell_checker() -> None:
    return None


@pytest.fixture
async def test_db() -> AsyncGenerator[None, None]:
    """Create test database tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
def client(test_db: None) -> Generator[TestClient, None, None]:
    """Create test client with in-memory database."""

    from rose_server.app import create_app
    from starlette.routing import _DefaultLifespan

    app = create_app()

    app.router.lifespan_context = _DefaultLifespan(app.router)

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_readonly_db_session] = override_readonly_db_session
    app.dependency_overrides[get_openai_client] = override_openai_client
    app.dependency_overrides[get_spell_checker] = override_spell_checker

    with TestClient(app) as test_client:
        yield test_client
