from collections.abc import AsyncIterator
from typing import Optional, Protocol, cast

from fastapi import Request
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell


class HasOpenAIClient(Protocol):
    openai_client: AsyncOpenAI


class HasJinja2Templates(Protocol):
    templates: Jinja2Templates


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.get_db_session() as session:
        yield session


async def get_readonly_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.get_db_session(read_only=True) as session:
        yield session


def get_openai_client(request: Request) -> AsyncOpenAI:
    state = cast(HasOpenAIClient, request.app.state)
    return state.openai_client


def get_spell_checker(request: Request) -> Optional[SymSpell]:
    return getattr(request.app.state, "spell_checker", None)


def get_templates(request: Request) -> Jinja2Templates:
    state = cast(HasJinja2Templates, request.app.state)
    return state.templates
