import httpx
from fastapi import APIRouter, Depends, HTTPException, Response

from rose_server.dependencies import get_llama_client

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(llama_client: httpx.AsyncClient = Depends(get_llama_client)) -> Response:
    try:
        upstream = await llama_client.get("models")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"LLM server unavailable: {str(e)}")

    content_type = upstream.headers.get("content-type", "application/json")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=content_type)
