"""Router module for embeddings API endpoints."""
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from rose_server.embeddings import generate_embeddings
from rose_server.schemas.embeddings import EmbeddingsRequest
router = APIRouter()
@router.post("/v1/embeddings")

async def openai_api_embeddings(request: EmbeddingsRequest = Body(...)) -> JSONResponse:
    """OpenAI API-compatible endpoint for generating embeddings.

    Args:
        request: The embeddings request containing input texts and model
    Returns:
        JSON response in OpenAI format with embeddings
    """
    try:
        response = generate_embeddings(texts=request.input, model_name=request.model)
        return JSONResponse(content=response)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request_error"}})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"An error occurred: {str(e)}", "type": "server_error"}},
        )