"""WebSocket server for streaming inference."""

import asyncio
import atexit
import json
import logging

import websockets

from rose_core.config.service import MAX_CONCURRENT_INFERENCE
from rose_inference.generation.generate.cache import cleanup_models
from rose_inference.generation.process import process_inference_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def handle_inference(websocket):
    """Handle a single inference WebSocket connection."""
    try:
        # Receive the request
        request = await websocket.recv()
        request_data = json.loads(request)

        # Process the inference request
        await process_inference_request(websocket, request_data)

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error handling inference: {e}")
        try:
            await websocket.send(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass  # Connection might be closed


async def run_inference_server():
    """Run the inference WebSocket server."""
    host = "localhost"
    port = 8005

    logger.info(f"Starting inference server on ws://{host}:{port}")
    logger.info(f"Max concurrent inferences: {MAX_CONCURRENT_INFERENCE}")

    async with websockets.serve(handle_inference, host, port, ping_interval=30, ping_timeout=120):
        await asyncio.Future()  # Run forever


def main():
    """Entry point for the inference server."""
    # Register cleanup handler
    atexit.register(cleanup_models)

    # Run server
    asyncio.run(run_inference_server())


if __name__ == "__main__":
    main()
