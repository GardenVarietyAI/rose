"""API client utilities for worker processes."""

import logging
import time
from typing import Any, Dict, Optional

import httpx

from rose_core.config.service import HOST, PORT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class ServiceClient:
    """HTTP client for worker→server communication."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url or f"http://{HOST}:{PORT}"
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with retry logic."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                    logger.warning(f"Request failed with {e.response.status_code}, retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    last_error = e
                    continue
                raise
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Connection error: {e}, retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    last_error = e
                    continue
                raise
        raise last_error

    def update_job_status(self, job_id: int, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Update job status in the API."""
        try:
            self._request_with_retry(
                "PATCH",
                f"/v1/jobs/{job_id}",
                json={"status": status, "result": result},
            )
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")

    def post_webhook(
        self,
        event: str,
        object_type: str,
        job_id: int,
        object_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Post a webhook event to the server."""
        payload = {
            "event": event,
            "object": object_type,
            "job_id": job_id,
            "object_id": object_id,
            "created_at": int(time.time()),
            "data": data or {},
        }

        try:
            self._request_with_retry("POST", "/v1/webhooks/jobs", json=payload)
        except httpx.HTTPStatusError as e:
            logger.warning(f"Webhook '{event}' failed with status {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.warning(f"Webhook '{event}' failed: {e}")


# Global client instance
_client: Optional[ServiceClient] = None


def get_client() -> ServiceClient:
    """Get or create the global service client."""
    global _client
    if _client is None:
        _client = ServiceClient()
    return _client


def update_job_status(job_id: int, status: str, result: Optional[Dict[str, Any]] = None) -> None:
    """Update job status using the global client."""
    get_client().update_job_status(job_id, status, result)


def post_webhook(
    event: str,
    object_type: str,
    job_id: int,
    object_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Post webhook using the global client."""
    get_client().post_webhook(event, object_type, job_id, object_id, data)
