"""WebSocket client for inference worker."""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import websockets

from rose_core.config.settings import settings
from rose_server.events.event_types import InputTokensCounted, ResponseCompleted, TokenGenerated

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for connecting to the inference worker via WebSocket."""

    def __init__(self, uri: Optional[str] = None):
        self.uri = uri or settings.inference_uri

    async def stream_inference(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        prompt: str,
        generation_kwargs: Dict[str, Any],
        response_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream inference results from the worker."""
        start_time = asyncio.get_event_loop().time()
        total_tokens = 0

        try:
            # Build headers for auth
            token = os.getenv("ROSE_API_KEY") or ""
            headers = {"Authorization": f"Bearer {token}"}

            async with websockets.connect(
                self.uri,
                additional_headers=headers,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=120,  # Wait 120 seconds for pong
                open_timeout=10,  # Wait 10 seconds for connection
            ) as websocket:
                # Send inference request
                request: Dict[str, Any] = {
                    "model_name": model_name,
                    "config": model_config,
                    "generation_kwargs": generation_kwargs,
                }

                # Send either messages or prompt, not both
                if messages:
                    request["messages"] = messages
                else:
                    request["prompt"] = prompt

                await websocket.send(json.dumps(request))
                logger.debug(f"Sent inference request for model {model_name}")

                # Stream responses with timeout
                asyncio.get_event_loop().time()
                async for message in websocket:
                    data = json.loads(message)

                    if data["type"] == "input_tokens_counted":
                        # Yield input tokens counted event
                        yield InputTokensCounted(
                            model_name=model_name,
                            response_id=response_id,
                            input_tokens=data["input_tokens"],
                        )

                    elif data["type"] == "token":
                        total_tokens += 1
                        yield TokenGenerated(
                            model_name=model_name,
                            token=data["token"],
                            token_id=total_tokens,
                            position=data["position"],
                            logprob=None,
                        )

                    elif data["type"] == "complete":
                        completion_time = asyncio.get_event_loop().time() - start_time
                        # Extract token counts from completion message
                        input_tokens = data.get("input_tokens", 0)
                        output_tokens = data.get("output_tokens", total_tokens)
                        total = data.get("total_tokens", input_tokens + output_tokens)

                        yield ResponseCompleted(
                            model_name=model_name,
                            response_id=response_id,
                            total_tokens=total,
                            finish_reason="stop",
                            output_tokens=output_tokens,
                            completion_time=completion_time,
                        )
                        break

                    elif data["type"] == "error":
                        logger.error(f"Inference error: {data['error']}")
                        raise RuntimeError(f"Inference failed: {data['error']}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected during inference")
            # Return a cancelled completion event
            completion_time = asyncio.get_event_loop().time() - start_time
            yield ResponseCompleted(
                model_name=model_name,
                response_id=response_id,
                total_tokens=total_tokens,
                finish_reason="cancelled",
                output_tokens=total_tokens,
                completion_time=completion_time,
            )
            return
        except OSError as e:
            logger.error(f"Could not connect to inference worker at {self.uri}: {e}")
            raise RuntimeError("Unable to connect to inference worker. Please ensure the inference service is running.")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise RuntimeError("Unable to connect to inference worker. Please ensure the inference service is running.")
