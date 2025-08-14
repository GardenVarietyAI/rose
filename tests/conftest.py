# mypy: ignore-errors

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from rose_server import database
from rose_server.app import create_app

# Create in-memory engine for testing
test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
    connect_args={
        "check_same_thread": False,
    },
)

# Create test session factory
test_async_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def test_db():
    """Create test database tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
def client(test_db, monkeypatch):
    """Create test client with in-memory database."""
    # Monkey patch the database session factory
    monkeypatch.setattr(database, "async_session_factory", test_async_session_factory)
    monkeypatch.setattr(database, "engine", test_engine)

    app = create_app()
    with TestClient(app) as test_client:
        # Create the default test model via API
        response = test_client.post(
            "/v1/models",
            json={
                "model_name": "qwen2.5-0.5b",
                "temperature": 0.7,
                "top_p": 0.9,
                "memory_gb": 1.0,
            },
        )
        assert response.status_code == 201
        yield test_client
