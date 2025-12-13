from typing import Any

from fastapi import APIRouter
from rose_server.llms import MODELS

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models() -> dict[str, Any]:
    models = [
        {
            "id": config["id"],
            "object": "model",
            "created": 1234567890,
            "owned_by": "system",
        }
        for config in MODELS.values()
    ]
    return {"object": "list", "data": models}
