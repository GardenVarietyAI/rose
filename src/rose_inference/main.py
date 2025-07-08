"""Simplified WebSocket inference server."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rose_inference.inference import InferenceHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    app.state.handler = InferenceHandler()
    logger.info("Inference handler initialized")
    yield
    # Cleanup
    app.state.handler.evict_cache()
    logger.info("Inference handler cleaned up")


app = FastAPI(title="ROSE Inference Server", lifespan=lifespan)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle inference WebSocket connections."""
    # Check auth if enabled
    auth_enabled = os.getenv("ROSE_SERVER_AUTH_ENABLED", "false").lower() == "true"
    if auth_enabled:
        token = os.getenv("ROSE_API_KEY")
        auth_header = websocket.headers.get("authorization", "")
        if token and auth_header != f"Bearer {token}":
            await websocket.close(code=1008, reason="Unauthorized")
            return

    await websocket.accept()

    try:
        # Receive request
        request = await websocket.receive_text()
        request_data = json.loads(request)

        # Handle control requests
        if request_data.get("type") == "control":
            action = request_data.get("action")

            if action == "evict_models":
                result = app.state.handler.evict_cache()
                await websocket.send_json(result)
            elif action == "cache_status":
                result = app.state.handler.get_status()
                await websocket.send_json(result)
            else:
                await websocket.send_json({"status": "error", "message": f"Unknown action: {action}"})
            return

        # Handle inference request
        async for event in app.state.handler.run_inference(
            config=request_data["config"],
            generation_kwargs=request_data["generation_kwargs"],
            messages=request_data.get("messages"),
            prompt=request_data.get("prompt", ""),
        ):
            await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error handling inference: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok"}


def main() -> None:
    """Entry point for the inference server."""
    logger.info("Starting inference server on ws://localhost:8005")
    uvicorn.run(app, host="localhost", port=8005, log_level="info")


if __name__ == "__main__":
    main()
