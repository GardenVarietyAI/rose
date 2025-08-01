"""Router module for embeddings API endpoints."""

from typing import List, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from rose_server.embeddings import generate_embeddings

router = APIRouter(prefix="/v1")


@router.post("/embeddings")
async def openai_api_embeddings(
    input: Union[str, List[str]],
    model: str = "text-embedding-ada-002",
) -> JSONResponse:
    """Generate embeddings.

    Args:
        request: The embeddings request containing input texts and model
    Returns:
        JSON response in OpenAI format with embeddings
    """
    try:
        response = generate_embeddings(texts=input, model_name=model)
        return JSONResponse(content=response)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": {"message": f"An error occurred: {str(e)}", "type": "server_error"}}
        )
