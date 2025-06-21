"""API client utilities for worker processes."""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from rose_core.config.service import HOST, PORT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class ServiceClient:
    """HTTP client for worker→server communication."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url or f"http://{HOST}:{PORT}"
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self) -> "ServiceClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self._client.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Make an HTTP request."""
        response = self._client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def update_job_status(self, job_id: int, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Update job status in the API."""
        self._request(
            "PATCH",
            f"/v1/jobs/{job_id}",
            json={"status": status, "result": result},
        )

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

        self._request("POST", "/v1/webhooks/jobs", json=payload)

    def get_queued_jobs(self, job_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get queued jobs of a specific type."""
        response = self._request(
            "GET",
            "/v1/jobs",
            params={
                "type": job_type,
                "status": "queued",
                "limit": limit,
            },
        )
        data = response.json()
        return data.get("data", [])

    def get_job_details(self, job_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific job."""
        response = self._request("GET", f"/v1/jobs/{job_id}")
        return response.json()

    def check_fine_tuning_job_status(self, ft_job_id: str) -> str:
        """Check if a fine-tuning job has been cancelled."""
        response = self._request("GET", f"/v1/fine_tuning/jobs/{ft_job_id}")
        data = response.json()
        if data.get("status") in ["cancelled", "failed"]:
            return data["status"]  # type: ignore[no-any-return]
        return "running"


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
