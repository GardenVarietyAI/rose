import sys
import types

# Stub external dependencies not available during tests
jinja2 = types.ModuleType("jinja2")
jinja2.Environment = object
jinja2.FileSystemLoader = object
sys.modules["jinja2"] = jinja2

openai = types.ModuleType("openai")
types_mod = types.ModuleType("types")
beta_mod = types.ModuleType("beta")
for cls_name in ["CodeInterpreterTool", "FileSearchTool", "FunctionTool"]:
    setattr(beta_mod, cls_name, type(cls_name, (), {}))
types_mod.beta = beta_mod
openai.types = types_mod
sys.modules["openai"] = openai
sys.modules["openai.types"] = types_mod
sys.modules["openai.types.beta"] = beta_mod

sys.path.append("src")

from rose_server.events.event_types.generation import (  # noqa: E402
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
)
from rose_server.events.formatters.responses import (  # noqa: E402
    ResponsesFormatter,
)


def test_function_call_output():
    formatter = ResponsesFormatter()
    events = [
        ResponseStarted(model_name="m", input_tokens=1),
        TokenGenerated(
            model_name="m",
            token="<tool_call><tool>foo</tool><args><a>1</a></args></tool_call>",
            token_id=1,
            position=0,
        ),
        ResponseCompleted(
            model_name="m",
            response_id="resp",
            total_tokens=1,
            finish_reason="stop",
        ),
    ]
    result = formatter.format_complete_response(events)
    assert result["output"][0]["type"] == "function_call"
    assert result["output"][0]["name"] == "foo"
