import os
from typing import Any, Dict

from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled


class CalculatorActor:
    """Agent specialized in mathematical calculations and problem solving."""

    def __init__(self, base_url: str = "http://localhost:8004/v1", model: str = "qwen2.5-0.5b") -> None:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = "dummy"
        set_tracing_disabled(True)
        set_default_openai_api("responses")

        instructions = """You are a precise mathematical assistant.

        When given a mathematical problem or expression:
        1. Break down complex problems step-by-step
        2. Show your calculations clearly
        3. Provide the final answer prominently
        4. Use proper mathematical notation when helpful
        5. Double-check your arithmetic

        Be concise but thorough in your explanations."""

        self.agent = Agent(
            name="Calculator",
            model=model,
            instructions=instructions,
        )

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the agent with the given query."""
        try:
            result = Runner.run_sync(self.agent, query)
            return {"response": result.final_output, "success": True}
        except Exception as e:
            return {"response": f"Error processing query: {str(e)}", "success": False}
