import atexit
import json
import logging
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rose_core.config.service import MAX_CONCURRENT_INFERENCE
from rose_inference.generation.cache import cleanup_models
from rose_inference.generation.process import process_inference_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ROSE Inference Server")


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle inference WebSocket connections."""
    await websocket.accept()

    try:
        # Receive the request
        request = await websocket.receive_text()
        request_data = json.loads(request)

        # Process the inference request
        await process_inference_request(websocket, request_data)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error handling inference: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass  # Connection might be closed


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "max_concurrent_inference": MAX_CONCURRENT_INFERENCE,
    }


def main() -> None:
    """Entry point for the inference server."""
    # Register cleanup handler
    atexit.register(cleanup_models)

    # Run server
    logger.info("Starting inference server on ws://localhost:8005")
    logger.info(f"Max concurrent inferences: {MAX_CONCURRENT_INFERENCE}")

    uvicorn.run(app, host="localhost", port=8005, log_level="info")


if __name__ == "__main__":
    main()
