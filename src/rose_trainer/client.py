"""API client utilities for worker processes."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from rose_trainer.types import ModelConfig

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class ServiceClient:
    """HTTP client for worker→server communication."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = os.getenv("ROSE_SERVER_URL", "http://127.0.0.1:8004")
        self.timeout = timeout
        token = os.getenv("ROSE_API_KEY") or ""
        headers = {"Authorization": f"Bearer {token}"}

        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

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
        self._request("PATCH", f"/v1/jobs/{job_id}", json={"status": status, "result": result})

    def get_model(self, model_id: str) -> ModelConfig:
        """Get model information from the API."""
        try:
            response = self._request("GET", f"/v1/models/{model_id}")
            result: Dict[str, Any] = response.json()
            model: ModelConfig = ModelConfig.model_validate(result)
            return model
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def post_webhook(
        self,
        event: str,
        object_type: str,
        job_id: int,
        object_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Post a webhook event to the server."""
        self._request(
            "POST",
            "/v1/webhooks/jobs",
            json={
                "event": event,
                "object": object_type,
                "job_id": job_id,
                "object_id": object_id,
                "created_at": int(time.time()),
                "data": data or {},
            },
        )

    def get_queued_jobs(self, job_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get queued jobs of a specific type."""
        response = self._request("GET", "/v1/jobs", params={"type": job_type, "status": "queued", "limit": limit})
        data: Dict[str, Any] = response.json()
        jobs: List[Dict[str, Any]] = data.get("data", [])
        return jobs

    def get_job_details(self, job_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific job."""
        response = self._request("GET", f"/v1/jobs/{job_id}")
        result: Dict[str, Any] = response.json()
        return result

    def check_fine_tuning_job_status(self, ft_job_id: str) -> str:
        """Check if a fine-tuning job has been cancelled."""
        response = self._request("GET", f"/v1/fine_tuning/jobs/{ft_job_id}")
        data = response.json()
        status: str = data.get("status")
        if status in ["cancelled", "failed"]:
            return status
        return "running"

    def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Create a chat completion via the API."""
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        if seed is not None:
            request_data["seed"] = seed

        # Use custom timeout for long generations
        old_timeout = self._client.timeout
        self._client.timeout = httpx.Timeout(timeout)
        try:
            response = self._request("POST", "/v1/chat/completions", json=request_data)
            result: Dict[str, Any] = response.json()
            return result
        finally:
            self._client.timeout = old_timeout

    def get_file_content(self, file_id: str) -> bytes:
        """Download file content via the API."""
        response = self._request("GET", f"/v1/files/{file_id}/content")
        if response.status_code == 404:
            raise FileNotFoundError(f"File '{file_id}' not found")
        return response.content
