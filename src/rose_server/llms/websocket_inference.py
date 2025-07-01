"""WebSocket client for inference worker."""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import websockets

from rose_server.events import ResponseCompleted, TokenGenerated

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for connecting to the inference worker via WebSocket."""

    def __init__(self, uri: str = "ws://localhost:8005"):
        self.uri = uri

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

        try:
            async with websockets.connect(
                self.uri,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=120,  # Wait 120 seconds for pong
            ) as websocket:
                # Send inference request
                request = {
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

                # Stream responses
                start_time = asyncio.get_event_loop().time()
                total_tokens = 0

                async for message in websocket:
                    data = json.loads(message)

                    if data["type"] == "token":
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
                        yield ResponseCompleted(
                            model_name=model_name,
                            response_id=response_id,
                            total_tokens=data.get("total_tokens", total_tokens),
                            finish_reason="stop",
                            output_tokens=data.get("total_tokens", total_tokens),
                            completion_time=completion_time,
                        )
                        break

                    elif data["type"] == "error":
                        logger.error(f"Inference error: {data['error']}")
                        raise RuntimeError(f"Inference failed: {data['error']}")

        except OSError as e:
            logger.error(f"Could not connect to inference worker at {self.uri}: {e}")
            raise
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}")
