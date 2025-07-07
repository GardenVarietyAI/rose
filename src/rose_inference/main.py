import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rose_core.config.settings import settings
from rose_inference.runner import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    # Initialize runner on startup
    app.state.runner = Runner()

    # Start the worker task
    app.state.runner.worker_task = asyncio.create_task(app.state.runner.start_worker())
    logger.info("Inference runner initialized with queue worker")

    yield

    # Cleanup on shutdown
    if app.state.runner.worker_task:
        app.state.runner.worker_task.cancel()
        try:
            await app.state.runner.worker_task
        except asyncio.CancelledError:
            pass

    app.state.runner.model_cache.evict()
    logger.info("Inference runner cleaned up")


app = FastAPI(title="ROSE Inference Server", lifespan=lifespan)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle inference WebSocket connections."""
    # Check auth if enabled
    if settings.auth_enabled:
        token = os.getenv("ROSE_API_KEY")
        # Headers are normalized to lowercase by Starlette
        auth_header = websocket.headers.get("authorization", "")
        if token and auth_header != f"Bearer {token}":
            await websocket.close(code=1008, reason="Unauthorized")
            return

    await websocket.accept()

    try:
        # Receive the request
        request = await websocket.receive_text()
        request_data = json.loads(request)

        # Check if this is a control request
        if request_data.get("type") == "control":
            action = request_data.get("action")

            if action == "evict_models":
                result = app.state.runner.evict_models()
                await websocket.send_json(result)

            elif action == "cache_status":
                result = app.state.runner.get_status()
                await websocket.send_json(result)

            else:
                await websocket.send_json({"status": "error", "message": f"Unknown control action: {action}"})
            return

        # Process the inference request
        async for event in app.state.runner.run_inference(
            model_name=request_data["model_name"],
            model_config=request_data["config"],
            generation_kwargs=request_data["generation_kwargs"],
            messages=request_data.get("messages"),
            prompt=request_data.get("prompt", ""),
            stream_id=request_data.get("stream_id"),
        ):
            await websocket.send_json(event)

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
        "max_concurrent_inference": settings.max_concurrent_inference,
    }


def main() -> None:
    """Entry point for the inference server."""

    # Run server
    logger.info("Starting inference server on ws://localhost:8005")
    logger.info(f"Max concurrent inferences: {settings.max_concurrent_inference}")

    uvicorn.run(app, host="localhost", port=8005, log_level="info")


if __name__ == "__main__":
    main()
