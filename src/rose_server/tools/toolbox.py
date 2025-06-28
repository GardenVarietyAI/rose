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
        "description": "Executes Python code in a subprocess",
        "supported": True,
    },
    "file_search": {
        "type": "file_search",
        "description": "Searches through attached documents using vector similarity",
        "supported": True,
    },
    "web_search": {
        "type": "web_search",
        "description": "Searches the web for current information using DuckDuckGo",
        "supported": True,
    },
}
