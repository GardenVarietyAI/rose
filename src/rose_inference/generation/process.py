"""Subprocess-per-request inference orchestrator."""

import asyncio
import json
import logging
import sys
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from rose_core.config.settings import settings

logger = logging.getLogger(__name__)

# Semaphore for controlling concurrent inferences (default: 1 = sequential)
inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inference)


async def process_inference_request(websocket: WebSocket, request_data: Dict[str, Any]) -> None:
    """Process inference request in isolated subprocess."""
    stream_id = str(uuid.uuid4())[:8]
    proc: Optional[asyncio.subprocess.Process] = None

    async with inference_semaphore:
        try:
            logger.info(f"[{stream_id}] Starting subprocess inference")

            # Check for empty input
            messages = request_data.get("messages")
            prompt = request_data.get("prompt", "")

            if not messages and not prompt:
                logger.info(f"[{stream_id}] Empty input received")
                await websocket.send_json({"type": "complete", "total_tokens": 0})
                return

            # Add stream_id to request
            request_data["stream_id"] = stream_id

            # Launch subprocess
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "rose_inference.generation.runner",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (OSError, PermissionError) as e:
                logger.error(f"[{stream_id}] Failed to create subprocess: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": f"Failed to create inference process: {e}",
                    }
                )
                return

            logger.info(f"[{stream_id}] Subprocess started with PID {proc.pid}")

            # Send request via stdin and close
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps(request_data).encode())
                    await proc.stdin.drain()
                    proc.stdin.close()
                else:
                    raise RuntimeError("Process stdin not available")
            except (BrokenPipeError, ConnectionError) as e:
                logger.error(f"[{stream_id}] Failed to send request to subprocess: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "Failed to communicate with inference process",
                    }
                )
                return

            # Create tasks for streaming
            async def stream_stdout() -> bool:
                """Stream stdout to websocket."""
                if not proc.stdout:
                    return False
                async for line in proc.stdout:
                    # Check if client is still connected
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.info(f"[{stream_id}] Client disconnected, stopping stream")
                        return False

                    try:
                        data = json.loads(line)
                        await websocket.send_json(data)
                    except json.JSONDecodeError:
                        logger.warning(f"[{stream_id}] Invalid JSON: {line.decode()}")
                    except Exception as e:
                        logger.error(f"[{stream_id}] Error forwarding: {e}")
                        return False
                return True

            async def stream_stderr() -> None:
                """Stream stderr for real-time logging."""
                if not proc.stderr:
                    return
                async for line in proc.stderr:
                    logger.info(f"[{stream_id}] Subprocess: {line.decode().strip()}")

            # Run streams with timeout
            try:
                stdout_task = asyncio.create_task(stream_stdout())
                stderr_task = asyncio.create_task(stream_stderr())

                # Wait for stdout completion with timeout (stderr can continue)
                try:
                    await asyncio.wait_for(stdout_task, timeout=settings.inference_timeout)
                except asyncio.TimeoutError:
                    logger.error(f"[{stream_id}] Inference timeout after {settings.inference_timeout}s")
                    stdout_task.cancel()
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"type": "error", "error": "Inference timeout"})
                    raise

                # Check if stdout completed successfully
                if not stdout_task.result():
                    raise Exception("Stream interrupted")

                # Give stderr a bit more time to finish logging
                try:
                    await asyncio.wait_for(stderr_task, timeout=5)
                except asyncio.TimeoutError:
                    stderr_task.cancel()

            except asyncio.TimeoutError:
                raise

            # Wait for process completion
            returncode = await asyncio.wait_for(proc.wait(), timeout=10)

            if returncode != 0:
                logger.error(f"[{stream_id}] Subprocess failed with code {returncode}")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "error", "error": "Inference failed"})

        except WebSocketDisconnect:
            logger.info(f"[{stream_id}] WebSocket disconnected during inference")
        except asyncio.CancelledError:
            logger.info(f"[{stream_id}] Inference cancelled")
            raise
        except Exception as e:
            logger.error(f"[{stream_id}] Error: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "error", "error": str(e)})
                except Exception:
                    pass
        finally:
            # Clean up subprocess if still running
            if proc and proc.returncode is None:
                logger.info(f"[{stream_id}] Terminating subprocess")
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning(f"[{stream_id}] Force killing subprocess")
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass  # Process already gone
                except ProcessLookupError:
                    pass  # Process already gone
