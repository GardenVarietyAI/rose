"""Simplified WebSocket inference server."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rose_inference.cache import ModelCache
from rose_inference.generator import generate_stream, generate_with_logprobs
from rose_inference.loader import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    app.state.cache = ModelCache()
    app.state.active_connections = 0
    app.state.max_connections = int(os.getenv("ROSE_INFERENCE_MAX_CONNECTIONS", "10"))

    logger.info(f"Inference server initialized (max_connections={app.state.max_connections})")
    yield

    # Cleanup
    app.state.cache.evict()
    logger.info("Inference server cleaned up")


app = FastAPI(title="ROSE Inference Server", lifespan=lifespan)


@app.websocket("/inference")
async def inference_endpoint(websocket: WebSocket) -> None:
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
        # Process inference requests on this connection sequentially
        async for message in websocket.iter_text():
            request_data = json.loads(message)

            try:
                config = request_data["config"]
                model_id = config.get("model_id")
                model_name = config.get("model_name")

                if not model_id or not model_name:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": f"Missing required config: model_id={model_id}, model_name={model_name}",
                        }
                    )
                    continue

                # Get or load model
                model_info = app.state.cache.get(model_id)
                if model_info is None:
                    logger.info(f"Cache miss, loading model: {model_name}")
                    model_info = await load_model(config)
                    app.state.cache.set(model_id, model_info)

                # Choose generation function based on logprobs parameter
                generation_kwargs = request_data["generation_kwargs"]
                if generation_kwargs.get("logprobs", False):
                    # Use non-streaming generation with logprobs
                    generator_fn = generate_with_logprobs
                else:
                    # Use regular streaming generation
                    generator_fn = generate_stream

                # Generate
                async for event in generator_fn(
                    model=model_info["model"],
                    tokenizer=model_info["tokenizer"],
                    messages=request_data.get("messages"),
                    prompt=request_data.get("prompt", ""),
                    generation_kwargs=generation_kwargs,
                    config=config,
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
    return {"status": "ok"}


@app.post("/control/evict")
async def evict_cache() -> Dict[str, str]:
    """Evict all cached models."""
    app.state.cache.evict()
    return {"status": "evicted", "message": "Model cache cleared"}


def main() -> None:
    """Entry point for the inference server."""
    host = os.getenv("ROSE_INFERENCE_HOST", "0.0.0.0")
    port = int(os.getenv("ROSE_INFERENCE_PORT", "8005"))
    logger.info(f"Starting inference server on ws://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
