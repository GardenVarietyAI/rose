from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from rose_server import database
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
    connect_args={
        "check_same_thread": False,
    },
)

test_async_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


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

    app.state.engine = test_engine
    app.state.db_session_maker = test_async_session_factory
    app.state.get_db_session = lambda read_only=False: database.get_session(test_async_session_factory, read_only)
    app.state.chat_model = None

    with TestClient(app) as test_client:
        yield test_client
