"""SQLModel-based storage for assistants with clean OpenAI compatibility."""
import logging
import uuid
from typing import List, Optional
from openai.types.beta.assistant_tool import CodeInterpreterTool, FileSearchTool, FunctionTool
from openai.types.shared_params import FunctionDefinition
from sqlalchemy import delete
from sqlmodel import select
from ..database import current_timestamp, run_in_session
from ..entities.assistants import Assistant as AssistantDB
from ..entities.assistants import AssistantTool
from ..schemas.assistants import Assistant, CreateAssistantRequest, UpdateAssistantRequest
logger = logging.getLogger(__name__)

class AssistantStore:
    """SQLModel-based storage for assistants."""

    def _to_openai_assistant(self, db_assistant: AssistantDB, tools: List[AssistantTool]) -> Assistant:
        """Convert database assistant to OpenAI-compatible Assistant model."""
        openai_tools = []
        tool_resources = {}
        for tool in tools:
            if tool.tool_type == "code_interpreter":
                openai_tools.append({"type": "code_interpreter"})
            elif tool.tool_type == "file_search":
                openai_tools.append({"type": "file_search"})
                if tool.file_ids:
                    tool_resources["file_search"] = {"vector_store_ids": tool.file_ids}
            elif tool.tool_type == "function":
                function_config = tool.tool_config or {}
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function_config.get("name", ""),
                            "description": function_config.get("description", ""),
                            "parameters": function_config.get("parameters", {}),
                        },
                    }
                )
        assistant_tools = []
        for tool_dict in openai_tools:
            if tool_dict["type"] == "code_interpreter":
                assistant_tools.append(CodeInterpreterTool(type="code_interpreter"))
            elif tool_dict["type"] == "file_search":
                assistant_tools.append(FileSearchTool(type="file_search"))
            elif tool_dict["type"] == "function":
                func = tool_dict["function"]
                assistant_tools.append(
                    FunctionTool(
                        type="function",
                        function=FunctionDefinition(
                            name=func["name"], description=func["description"], parameters=func["parameters"]
                        ),
                    )
                )
        return Assistant(
            id=db_assistant.id,
            object="assistant",
            created_at=db_assistant.created_at,
            name=db_assistant.name,
            description=db_assistant.description,
            model=db_assistant.model,
            instructions=db_assistant.instructions,
            tools=assistant_tools,
            tool_resources=tool_resources,
            metadata={},
            temperature=db_assistant.temperature,
            top_p=db_assistant.top_p,
            response_format=None,
        )

    async def create_assistant(self, request: CreateAssistantRequest) -> Assistant:
        """Create a new assistant."""
        assistant_id = f"asst_{uuid.uuid4().hex}"

        async def operation(session):
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
            return self._to_openai_assistant(db_assistant, tools)
        result = await run_in_session(operation)
        logger.info(f"Created assistant: {assistant_id}")
        return result

    async def get_assistant(self, assistant_id: str) -> Optional[Assistant]:
        """Get an assistant by ID."""

        async def operation(session):
            db_assistant = await session.get(AssistantDB, assistant_id)
            if not db_assistant:
                return None
            statement = select(AssistantTool).where(AssistantTool.assistant_id == assistant_id)
            tools = (await session.execute(statement)).scalars().all()
            return self._to_openai_assistant(db_assistant, tools)
        return await run_in_session(operation, read_only=True)

    async def list_assistants(self, limit: int = 20, order: str = "desc") -> List[Assistant]:
        """List assistants."""

        async def operation(session):
            statement = select(AssistantDB)
            if order == "desc":
                statement = statement.order_by(AssistantDB.created_at.desc())
            else:
                statement = statement.order_by(AssistantDB.created_at.asc())
            statement = statement.limit(limit)
            db_assistants = (await session.execute(statement)).scalars().all()
            assistant_ids = [a.id for a in db_assistants]
            tools_statement = select(AssistantTool).where(AssistantTool.assistant_id.in_(assistant_ids))
            all_tools = (await session.execute(tools_statement)).scalars().all()
            tools_by_assistant = {}
            for tool in all_tools:
                if tool.assistant_id not in tools_by_assistant:
                    tools_by_assistant[tool.assistant_id] = []
                tools_by_assistant[tool.assistant_id].append(tool)
            return [self._to_openai_assistant(a, tools_by_assistant.get(a.id, [])) for a in db_assistants]
        return await run_in_session(operation, read_only=True)

    async def update_assistant(self, assistant_id: str, request: UpdateAssistantRequest) -> Optional[Assistant]:
        """Update an assistant."""

        async def operation(session):
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
            return self._to_openai_assistant(db_assistant, tools)
        result = await run_in_session(operation)
        logger.info(f"Updated assistant: {assistant_id}")
        return result

    async def delete_assistant(self, assistant_id: str) -> bool:
        """Delete an assistant and all associated tools."""

        async def operation(session):
            db_assistant = await session.get(AssistantDB, assistant_id)
            if db_assistant:
                await session.execute(delete(AssistantTool).where(AssistantTool.assistant_id == assistant_id))
                await session.delete(db_assistant)
                await session.commit()
                logger.info(f"Deleted assistant: {assistant_id}")
                return True
            return False
        return await run_in_session(operation)

def get_assistant_store() -> AssistantStore:
    """Get the assistant store instance."""
    return AssistantStore()