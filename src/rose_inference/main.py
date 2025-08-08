"""Inference Server"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rose_inference.cache import CachedModelRegistry
from rose_inference.generator import ModelGenerationParams, stream, with_logprobs
from rose_inference.loader import ModelLoaderParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    app.state.cache = CachedModelRegistry()
    app.state.active_connections = 0
    app.state.max_connections = int(os.getenv("ROSE_INFERENCE_MAX_CONNECTIONS", "10"))

    logger.info(f"Inference server initialized (max_connections={app.state.max_connections})")
    yield

    # Cleanup
    app.state.cache.evict()
    logger.info("Inference server cleaned up")


app = FastAPI(title="ROSE Inference Server", lifespan=lifespan)


@app.websocket("/inference")
async def inference(websocket: WebSocket) -> None:
    """Handle persistent inference WebSocket connections."""
    # Check connection limit before accepting
    if app.state.active_connections >= app.state.max_connections:
        await websocket.close(code=1013, reason="Server busy: too many connections")
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
        async for message in websocket.iter_text():
            request_data = json.loads(message)

            try:
                config = request_data["config"]
                generation_kwargs = request_data.get("generation_kwargs", {}) or {}

                load_params = ModelLoaderParams(
                    model_id=config.get("model_id"),
                    model_name=config.get("model_name"),
                    model_path=config.get("model_path"),
                    torch_dtype=config.get("torch_dtype"),
                    data_dir=config.get("data_dir", "./data"),
                )

                if not load_params.model_id or not load_params.model_name:
                    await websocket.send_json(
                        {"type": "error", "error": f"Missing config for model {load_params.model_id}"}
                    )
                    continue

                # Get or load model
                cached = await app.state.cache.get_or_load(load_params)
                generate = stream if not generation_kwargs.get("logprobs", False) else with_logprobs

                # Generate
                async with cached.use():
                    async for event in generate(
                        ModelGenerationParams(
                            model=cached.model,
                            tokenizer=cached.tokenizer,
                            messages=request_data.get("messages"),
                            prompt=request_data.get("prompt", ""),
                            generation_kwargs=generation_kwargs,
                        ),
                    ):
                        await websocket.send_json(event)
            except Exception:
                logger.exception("Error processing request")
                await websocket.send_json({"type": "error", "error": "internal error"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("Error processing request")
        logger.exception("WebSocket error")
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
    await app.state.cache.evict()
    return {"status": "evicted", "message": "Model cache cleared"}


def main() -> None:
    """Entry point for the inference server."""
    host = os.getenv("ROSE_INFERENCE_HOST", "0.0.0.0")
    port = int(os.getenv("ROSE_INFERENCE_PORT", "8005"))
    logger.info(f"Starting inference server on ws://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
