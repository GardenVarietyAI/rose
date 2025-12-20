from collections.abc import AsyncIterator
from typing import Optional, Protocol, cast

import httpx
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

from rose_server.settings import Settings


class HasLlamaClient(Protocol):
    llama_client: httpx.AsyncClient


class HasSettings(Protocol):
    settings: Settings


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.get_db_session() as session:
        yield session


async def get_readonly_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.get_db_session(read_only=True) as session:
        yield session


def get_llama_client(request: Request) -> httpx.AsyncClient:
    state = cast(HasLlamaClient, request.app.state)
    return state.llama_client


def get_settings(request: Request) -> Settings:
    state = cast(HasSettings, request.app.state)
    return getattr(state, "settings", Settings())


def get_spell_checker(request: Request) -> Optional[SymSpell]:
    return getattr(request.app.state, "spell_checker", None)
