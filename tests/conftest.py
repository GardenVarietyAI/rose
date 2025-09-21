# mypy: ignore-errors

import pytest
from fastapi.testclient import TestClient
from rose_server import database
from rose_server.connect import _VecConnection
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
    connect_args={
        "check_same_thread": False,
        "factory": _VecConnection,
    },
)

test_async_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def test_db():
    """Create test database tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        # Create vec0 table
        from sqlalchemy import text

        await conn.execute(
            text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec0 USING vec0(
                document_id TEXT PRIMARY KEY,
                embedding float[64]
            )
        """)
        )
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
def client(test_db):
    """Create test client with in-memory database."""
    from rose_server._inference import InferenceServer
    from rose_server.app import create_app, get_embedding_model, get_tokenizer
    from rose_server.settings import Settings
    from starlette.routing import _DefaultLifespan

    app = create_app()

    # Skip the lifespan
    app.router.lifespan_context = _DefaultLifespan(app.router)

    settings = Settings()
    app.state.settings = settings
    app.state.engine = test_engine
    app.state.db_session_maker = test_async_session_factory
    app.state.get_db_session = lambda read_only=False: database.get_session(test_async_session_factory, read_only)
    app.state.inference_server = InferenceServer("auto")

    try:
        app.state.embedding_model = get_embedding_model(
            models_dir=settings.models_dir,
            embedding_model_name=settings.embedding_model_name,
            embedding_model_quantization=settings.embedding_model_quantization,
            embedding_device=settings.embedding_device,
            embedding_dimensions=settings.embedding_dimensions,
        )
        app.state.embedding_tokenizer = get_tokenizer(
            models_dir=settings.models_dir,
            embedding_model_name=settings.embedding_model_name,
        )
    except Exception:
        app.state.embedding_model = None
        app.state.embedding_tokenizer = None

    app.state.reranker_model = None
    app.state.tokenizer = None

    with TestClient(app) as test_client:
        # Create the default test model via API
        response = test_client.post(
            "/v1/models",
            json={
                "model_name": "qwen2.5-0.5b",
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        assert response.status_code == 201
        yield test_client
