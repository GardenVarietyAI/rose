from typing import Any, AsyncGenerator, Generator

import httpx
import pytest
from fastapi.testclient import TestClient
from rose_server import database
from rose_server.dependencies import (
    get_db_session,
    get_llama_client,
    get_readonly_db_session,
    get_spell_checker,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel


class DummyLlamaClient:
    async def get(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("GET", f"http://llama.local/{url}")
        if url == "models":
            return httpx.Response(
                200,
                request=request,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen3-0.6B-Q8_0.gguf",
                            "object": "model",
                            "created": 0,
                            "owned_by": "system",
                        }
                    ],
                },
            )
        return httpx.Response(404, request=request, json={"error": {"message": "not found"}})

    async def post(self, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", f"http://llama.local/{url}")
        if url != "chat/completions":
            return httpx.Response(404, request=request, json={"error": {"message": "not found"}})

        payload = kwargs.get("json") or {}
        messages = payload.get("messages") or []
        content = ""
        if messages and isinstance(messages, list) and isinstance(messages[-1], dict):
            content = str(messages[-1].get("content", ""))

        return httpx.Response(
            200,
            request=request,
            json={
                "id": "dummy-completion-id",
                "object": "chat.completion",
                "model": "/tmp/Qwen3-0.6B-Q8_0.gguf",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


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


def override_llama_client() -> DummyLlamaClient:
    return DummyLlamaClient()


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
    app.dependency_overrides[get_llama_client] = override_llama_client
    app.dependency_overrides[get_spell_checker] = override_spell_checker

    with TestClient(app) as test_client:
        yield test_client
