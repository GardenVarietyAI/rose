"""Router module for embeddings API endpoints."""

from fastapi import APIRouter, HTTPException

from rose_server.embeddings import generate_embeddings_async
from rose_server.schemas.embeddings import EmbeddingRequest, EmbeddingResponse

router = APIRouter(prefix="/v1/embeddings")


@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings.

    Args:
        request: The embeddings request containing input texts and model
    Returns:
        JSON response in OpenAI format with embeddings
    """
    try:
        response = await generate_embeddings_async(texts=request.input, model_name=request.model)
        return EmbeddingResponse(**response)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": {"message": f"An error occurred: {str(e)}", "type": "server_error"}}
        )
