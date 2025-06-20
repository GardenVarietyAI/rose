import logging
import time
from typing import Any, Dict, Optional

import httpx

from .config.service import HOST, PORT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0


class WebhookClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url or f"http://{HOST}:{PORT}"
        self.timeout = timeout

    def post_event(
        self,
        event: str,
        object_type: str,
        job_id: int,
        object_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Post a webhook event to the server.

        Args:
            event: Event type (e.g., "job.running", "job.completed")
            object_type: Object type ("training" or "eval")
            job_id: Queue job ID
            object_id: Object-specific ID (fine-tuning job ID or eval ID)
            data: Optional event data
        """
        payload = {
            "event": event,
            "object": object_type,
            "job_id": job_id,
            "object_id": object_id,
            "created_at": int(time.time()),
            "data": data or {},
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/v1/webhooks/jobs", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.warning("Webhook '%s' failed with status %d: %s", event, e.response.status_code, e.response.text)
        except Exception as e:
            logger.warning("Webhook '%s' failed: %s", event, e)


default_client = WebhookClient()


def post_webhook(
    event: str,
    object_type: str,
    job_id: int,
    object_id: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Convenience function using the default webhook client."""
    default_client.post_event(event, object_type, job_id, object_id, data)
