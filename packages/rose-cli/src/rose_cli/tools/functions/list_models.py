from typing import Any

from agents import RunContextWrapper, function_tool

from rose_cli.utils import get_client


@function_tool
def list_models(ctx: RunContextWrapper[Any]) -> str:
    """List all available models from the ROSE API."""
    try:
        client = get_client()

        response = client.models.list()
        models = []
        for model in response.data:
            model_info = f"• {model.id}"
            if hasattr(model, "parent") and model.parent:
                model_info += f" (fine-tuned from {model.parent})"
            else:
                model_info += f" (by {model.owned_by})"
            models.append(model_info)

        if models:
            return f"Available models ({len(models)} total):\n" + "\n".join(models)
        else:
            return "No models found"
    except Exception as e:
        return f"Error listing models: {str(e)}"
