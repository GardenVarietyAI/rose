from openai.types.beta.assistant_tool import AssistantTool
from openai.types.beta.function_tool import FunctionTool
from openai.types.beta.threads.required_action_function_tool_call import RequiredActionFunctionToolCall
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput as OpenAIToolOutput

Tool = AssistantTool
ToolFunction = FunctionTool
ToolCall = RequiredActionFunctionToolCall
ToolOutput = OpenAIToolOutput
BUILTIN_TOOLS = {
    "code_interpreter": {
        "type": "code_interpreter",
        "description": "Code interpreter tool (not implemented - use function calling instead)",
    },
    "retrieval": {
        "type": "retrieval",
        "description": "Searches through attached documents using vector similarity",
        "supported": True,
    },
    "file_search": {
        "type": "file_search",
        "description": "Enhanced version of retrieval for OpenAI v2 API",
        "supported": True,
    },
}
