"""SQLModel-based storage for assistants with clean OpenAI compatibility."""

import logging
import uuid
from typing import List, Optional, Union

from openai.types.beta.assistant_tool import CodeInterpreterTool, FileSearchTool, FunctionTool
from openai.types.shared_params import FunctionDefinition
from sqlalchemy import delete
from sqlmodel import select

from rose_server.database import current_timestamp, get_session
from rose_server.entities.assistants import (
    Assistant as AssistantDB,
    AssistantTool,
)
from rose_server.schemas.assistants import AssistantCreateRequest, AssistantResponse, AssistantUpdateRequest

logger = logging.getLogger(__name__)


def _to_openai_assistant(db: AssistantDB, tools: List[AssistantTool]) -> AssistantResponse:
    assistant_tools: List[Union[CodeInterpreterTool, FileSearchTool, FunctionTool]] = []
    tool_resources = {}

    for t in tools:
        match t.tool_type:
            case "code_interpreter":
                assistant_tools.append(CodeInterpreterTool(type="code_interpreter"))
            case "file_search":
                assistant_tools.append(FileSearchTool(type="file_search"))
                if t.file_ids:
                    tool_resources["file_search"] = {"vector_store_ids": t.file_ids}
            case "function":
                cfg = t.tool_config or {}
                assistant_tools.append(
                    FunctionTool(
                        type="function",
                        function=FunctionDefinition(
                            name=cfg.get("name", ""),
                            description=cfg.get("description", ""),
                            parameters=cfg.get("parameters", {}),
                        ),
                    )
                )

    return AssistantResponse(
        id=db.id,
        object="assistant",
        created_at=db.created_at,
        name=db.name,
        description=db.description,
        model=db.model,
        instructions=db.instructions,
        tools=assistant_tools,
        tool_resources=tool_resources,
        metadata={},
        temperature=db.temperature,
        top_p=db.top_p,
        response_format=None,
    )


async def create_assistant(request: AssistantCreateRequest) -> AssistantResponse:
    """Create a new assistant."""
    assistant_id = f"asst_{uuid.uuid4().hex}"
    async with get_session() as session:
        db_assistant = AssistantDB(
            id=assistant_id,
            created_at=current_timestamp(),
            name=request.name,
            description=request.description,
            model=request.model,
            instructions=request.instructions,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 1.0,
        )
        session.add(db_assistant)

        tools = []
        for tool in request.tools:
            if tool.type == "code_interpreter":
                db_tool = AssistantTool(assistant_id=assistant_id, tool_type="code_interpreter")
            elif tool.type == "file_search":
                file_ids = []
                if request.tool_resources and "file_search" in request.tool_resources:
                    file_ids = request.tool_resources["file_search"].get("vector_store_ids", [])
                db_tool = AssistantTool(
                    assistant_id=assistant_id, tool_type="file_search", file_ids=file_ids if file_ids else None
                )
            elif tool.type == "function":
                db_tool = AssistantTool(
                    assistant_id=assistant_id,
                    tool_type="function",
                    tool_config={
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    },
                )
            else:
                continue

            session.add(db_tool)
            tools.append(db_tool)

        await session.commit()
        await session.refresh(db_assistant)

        if tools:
            for tool in tools:
                await session.refresh(tool)

        return _to_openai_assistant(db_assistant, tools)


async def get_assistant(assistant_id: str) -> Optional[AssistantResponse]:
    """Get an assistant by ID."""
    async with get_session(read_only=True) as session:
        db_assistant = await session.get(AssistantDB, assistant_id)
        if not db_assistant:
            return None
        statement = select(AssistantTool).where(AssistantTool.assistant_id == assistant_id)
        tools = (await session.execute(statement)).scalars().all()
        return _to_openai_assistant(db_assistant, tools)


async def list_assistants(limit: int = 20, order: str = "desc") -> List[AssistantResponse]:
    """List assistants."""
    async with get_session(read_only=True) as session:
        statement = select(AssistantDB)
        if order == "desc":
            statement = statement.order_by(AssistantDB.created_at.desc())
        else:
            statement = statement.order_by(AssistantDB.created_at.asc())
        statement = statement.limit(limit)
        db_assistants = (await session.execute(statement)).scalars().all()
        assistant_ids = [a.id for a in db_assistants]
        all_tools = []
        if assistant_ids:
            tools_statement = select(AssistantTool).where(AssistantTool.assistant_id.in_(assistant_ids))
            all_tools = (await session.execute(tools_statement)).scalars().all()
        tools_by_assistant = {}
        for tool in all_tools:
            if tool.assistant_id not in tools_by_assistant:
                tools_by_assistant[tool.assistant_id] = []
            tools_by_assistant[tool.assistant_id].append(tool)
        return [_to_openai_assistant(a, tools_by_assistant.get(a.id, [])) for a in db_assistants]


async def update_assistant(assistant_id: str, request: AssistantUpdateRequest) -> Optional[AssistantResponse]:
    """Update an assistant."""
    async with get_session() as session:
        db_assistant = await session.get(AssistantDB, assistant_id)
        if not db_assistant:
            return None
        if request.model is not None:
            db_assistant.model = request.model
        if request.name is not None:
            db_assistant.name = request.name
        if request.description is not None:
            db_assistant.description = request.description
        if request.instructions is not None:
            db_assistant.instructions = request.instructions
        if request.temperature is not None:
            db_assistant.temperature = request.temperature
        if request.top_p is not None:
            db_assistant.top_p = request.top_p
        if request.tools is not None:
            await session.execute(delete(AssistantTool).where(AssistantTool.assistant_id == assistant_id))
            tools = []
            for tool in request.tools:
                if tool.type == "code_interpreter":
                    db_tool = AssistantTool(assistant_id=assistant_id, tool_type="code_interpreter")
                elif tool.type == "file_search":
                    file_ids = []
                    if request.tool_resources and "file_search" in request.tool_resources:
                        file_ids = request.tool_resources["file_search"].get("vector_store_ids", [])
                    db_tool = AssistantTool(
                        assistant_id=assistant_id, tool_type="file_search", file_ids=file_ids if file_ids else None
                    )
                elif tool.type == "function":
                    db_tool = AssistantTool(
                        assistant_id=assistant_id,
                        tool_type="function",
                        tool_config={
                            "name": tool.function.name,
                            "description": tool.function.description,
                            "parameters": tool.function.parameters,
                        },
                    )
                else:
                    continue
                session.add(db_tool)
                tools.append(db_tool)
        else:
            tools_statement = select(AssistantTool).where(AssistantTool.assistant_id == assistant_id)
            tools = (await session.execute(tools_statement)).scalars().all()
        session.add(db_assistant)
        await session.commit()
        await session.refresh(db_assistant)
        if tools:
            for tool in tools:
                await session.refresh(tool)
        return _to_openai_assistant(db_assistant, tools)


async def delete_assistant(assistant_id: str) -> bool:
    """Delete an assistant and all associated tools."""
    async with get_session() as session:
        db_assistant = await session.get(AssistantDB, assistant_id)
        if db_assistant:
            await session.execute(delete(AssistantTool).where(AssistantTool.assistant_id == assistant_id))
            await session.delete(db_assistant)
            await session.commit()
            logger.info(f"Deleted assistant: {assistant_id}")
            return True
        return False
