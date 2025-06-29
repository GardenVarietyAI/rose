"""Worker that handles both WebSocket inference and polled training jobs."""

import asyncio
import json
import logging
import os

import websockets

from .client import get_client
from .fine_tuning.process import process_training_job
from .inference.process import process_inference_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"


async def poll_training_jobs():
    """Poll for training jobs using the existing pattern."""
    client = get_client()
    logger.info("Started polling for training jobs")

    while True:
        try:
            jobs = client.get_queued_jobs("training", limit=1)
            if jobs:
                job = jobs[0]
                logger.info(f"Running training job {job['id']}")
                # Run in thread to not block the event loop
                await asyncio.get_event_loop().run_in_executor(None, process_training_job, job["id"], job["payload"])
        except Exception as e:
            logger.error(f"Training job failed: {e}")

        await asyncio.sleep(5)


async def handle_inference_connection(websocket):
    """Handle a single inference WebSocket connection."""
    try:
        # Receive the request
        request = await websocket.recv()
        request_data = json.loads(request)

        # Process the inference request
        await process_inference_request(websocket, request_data)

    except websockets.exceptions.ConnectionClosed:
        logger.info("Inference connection closed")
    except Exception as e:
        logger.error(f"Error handling inference connection: {e}")


async def handle_inference_stream():
    """Run the WebSocket inference server."""
    host = "localhost"
    port = 8005

    logger.info(f"Starting inference WebSocket server on ws://{host}:{port}")

    async with websockets.serve(
        handle_inference_connection,
        host,
        port,
        ping_interval=30,  # Send ping every 30 seconds
        ping_timeout=120,  # Wait 120 seconds for pong
    ):
        await asyncio.Future()  # Run forever


async def run_worker():
    """Run both inference server and training poller."""
    logger.info("Worker started")

    await asyncio.gather(handle_inference_stream(), poll_training_jobs())


def main():
    """Entry point for the worker process."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
