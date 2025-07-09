"""Client for inference worker with WebSocket for streaming and HTTP for control."""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import websockets

from rose_server.config.settings import settings
from rose_server.events.event_types import (
    InputTokensCounted,
    ResponseCompleted,
    TokenGenerated,
)

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for connecting to the inference worker via WebSocket and HTTP."""

    def __init__(self, uri: Optional[str] = None):
        self.uri = uri or settings.inference_uri
        # Extract base URL for HTTP endpoints
        self.base_url = self.uri.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")
        # Build auth headers once
        token = os.getenv("ROSE_API_KEY") or ""
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def evict_models(self) -> Dict[str, Any]:
        """Evict all cached models from the inference service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/control/evict",
                    headers=self.headers,
                    timeout=5.0,
                )
                response.raise_for_status()
                result: Dict[str, Any] = response.json()

                logger.info(f"Model eviction response: {result}")
                return result

        except httpx.TimeoutException:
            logger.error("Timeout waiting for eviction response")
            return {"status": "timeout", "message": "Eviction request timed out"}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error evicting models: {e}")
            return {"status": "error", "message": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            logger.error(f"Failed to evict models: {e}")
            return {"status": "error", "message": str(e)}

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
        start_time = asyncio.get_running_loop().time()
        total_tokens = 0

        try:
            # Update URI to use /inference endpoint
            inference_uri = self.uri + "/inference"

            async with websockets.connect(
                inference_uri,
                additional_headers=self.headers,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=120,  # Wait 120 seconds for pong
                open_timeout=10,  # Wait 10 seconds for connection
            ) as websocket:
                # Send inference request
                request: Dict[str, Any] = {
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
                timeout = settings.inference_timeout
                deadline = asyncio.get_running_loop().time() + timeout

                async for message in websocket:
                    # Check if we've exceeded the timeout
                    if asyncio.get_running_loop().time() > deadline:
                        logger.error(f"Inference timeout after {timeout}s")
                        raise asyncio.TimeoutError(f"Inference timeout after {timeout}s")
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
                        completion_time = asyncio.get_running_loop().time() - start_time
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

        except asyncio.TimeoutError as e:
            logger.error(f"Inference timeout: {e}")
            # Return a timeout completion event
            completion_time = asyncio.get_running_loop().time() - start_time
            yield ResponseCompleted(
                model_name=model_name,
                response_id=response_id,
                total_tokens=total_tokens,
                finish_reason="timeout",
                output_tokens=total_tokens,
                completion_time=completion_time,
            )
            return
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected during inference")
            # Return a cancelled completion event
            completion_time = asyncio.get_running_loop().time() - start_time
            yield ResponseCompleted(
                model_name=model_name,
                response_id=response_id,
                total_tokens=total_tokens,
                finish_reason="cancelled",
                output_tokens=total_tokens,
                completion_time=completion_time,
            )
            return
        except (OSError, ValueError) as e:
            logger.error(f"Could not connect to inference worker at {self.uri}: {e}")
            raise RuntimeError("Unable to connect to inference worker. Please ensure the inference service is running.")
        except RuntimeError:
            # Re-raise RuntimeError to preserve the original error message
            raise
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise RuntimeError(f"Inference error: {str(e)}")
