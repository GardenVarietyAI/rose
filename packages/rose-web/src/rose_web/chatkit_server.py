from typing import Any, AsyncIterator

from agents import Agent, RunConfig, Runner
from agents.models.openai_provider import OpenAIProvider
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.store import AttachmentStore, Store
from chatkit.types import ThreadMetadata, ThreadStreamEvent, UserMessageItem
from openai import AsyncOpenAI

from rose_web.settings import get_settings


class RoseChatKitServer(ChatKitServer[Any]):
    def __init__(
        self,
        data_store: Store,
        file_store: AttachmentStore | None = None,
    ):
        super().__init__(data_store, file_store)
        settings = get_settings()
        self.default_model = settings.default_model
        self.openai_client = AsyncOpenAI(
            base_url=settings.openai_api_url,
            api_key=settings.openai_api_key,
        )

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: Any,
    ) -> AsyncIterator[ThreadStreamEvent]:
        model = self.default_model
        if input_user_message and input_user_message.inference_options.model:
            model = input_user_message.inference_options.model

        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )

        agent = Agent[AgentContext[Any]](
            model=model,
            name="Assistant",
            instructions="You are a helpful assistant. Be concise.",
        )

        openai_provider = OpenAIProvider(openai_client=self.openai_client)

        agent_input = []
        if input_user_message:
            agent_input.extend(await simple_to_agent_input(input_user_message))

        previous_response_id = thread.metadata.get("previous_response_id")
        result = Runner.run_streamed(
            agent,
            agent_input,
            context=agent_context,
            run_config=RunConfig(model_provider=openai_provider, tracing_disabled=True),
            previous_response_id=previous_response_id,
        )

        async for event in stream_agent_response(agent_context, result):
            yield event

        thread.metadata["previous_response_id"] = result.last_response_id
        await self.store.save_thread(thread, context)
