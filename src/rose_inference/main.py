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
    app.state.active_connections = 0
    app.state.max_connections = int(os.getenv("ROSE_INFERENCE_MAX_CONNECTIONS", "10"))

    logger.info(f"Inference handler initialized (max_connections={app.state.max_connections})")
    yield

    # Cleanup
    app.state.handler.evict_cache()
    logger.info("Inference handler cleaned up")


app = FastAPI(title="ROSE Inference Server", lifespan=lifespan)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle persistent inference WebSocket connections."""
    # Check connection limit before accepting
    if app.state.active_connections >= app.state.max_connections:
        await websocket.close(code=1013, reason="Server busy - too many connections")
        return

    # Check auth if enabled
    auth_enabled = os.getenv("ROSE_SERVER_AUTH_ENABLED", "false").lower() == "true"
    if auth_enabled:
        token = os.getenv("ROSE_API_KEY")
        auth_header = websocket.headers.get("authorization", "")
        if token and auth_header != f"Bearer {token}":
            await websocket.close(code=1008, reason="Unauthorized")
            return

    await websocket.accept()
    app.state.active_connections += 1
    logger.info(f"Client connected (active_connections={app.state.active_connections})")

    try:
        # Process requests on this connection sequentially
        async for message in websocket.iter_text():
            request_data = json.loads(message)

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
                continue

            # Handle inference request
            try:
                async for event in app.state.handler.run_inference(
                    config=request_data["config"],
                    generation_kwargs=request_data["generation_kwargs"],
                    messages=request_data.get("messages"),
                    prompt=request_data.get("prompt", ""),
                ):
                    await websocket.send_json(event)
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await websocket.send_json({"type": "error", "error": str(e)})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app.state.active_connections -= 1
        logger.info(f"Client cleanup (active_connections={app.state.active_connections})")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "active_connections": app.state.active_connections,
        "max_connections": app.state.max_connections,
        "cache_status": app.state.handler.get_status(),
    }


def main() -> None:
    """Entry point for the inference server."""
    logger.info("Starting inference server on ws://localhost:8005")
    uvicorn.run(app, host="localhost", port=8005, log_level="info")


if __name__ == "__main__":
    main()
