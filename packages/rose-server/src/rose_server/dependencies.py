from collections.abc import AsyncIterator
from typing import Optional, Protocol, cast

import httpx
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell


class HasLlamaClient(Protocol):
    llama_client: httpx.AsyncClient


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.get_db_session() as session:
        yield session


async def get_readonly_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.get_db_session(read_only=True) as session:
        yield session


def get_llama_client(request: Request) -> httpx.AsyncClient:
    state = cast(HasLlamaClient, request.app.state)
    return state.llama_client


def get_spell_checker(request: Request) -> Optional[SymSpell]:
    return getattr(request.app.state, "spell_checker", None)
