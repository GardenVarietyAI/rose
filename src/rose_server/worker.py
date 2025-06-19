from __future__ import annotations

import gc
import logging
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx
import torch
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from .config import ServiceConfig
from .fine_tuning.training import HFTrainer

logger = logging.getLogger(__name__)
BASE_URL = "http://localhost:8004"
HTTP_TIMEOUT = 10
POLL_INTERVAL = 5
TRAINING_JOB_TIMEOUT = 7_200


def _now() -> int:
    return int(time.time())


def _session() -> httpx.Client:
    return httpx.Client(timeout=HTTP_TIMEOUT)


def _post_webhook(event: str, obj: str, job_id: int, object_id: str, data: Dict | None = None) -> None:
    payload = {
        "event": event,
        "object": obj,
        "job_id": job_id,
        "object_id": object_id,
        "created_at": _now(),
        "data": data or {},
    }
    try:
        with _session() as client:
            client.post(f"{BASE_URL}/v1/webhooks/jobs", json=payload)
    except Exception as exc:
        logger.warning("Webhook '%s' failed: %s", event, exc)


def _device_memory_log(label: str) -> None:
    if torch.backends.mps.is_available():
        alloc = torch.mps.current_allocated_memory() / 1024**3
        resv = torch.mps.driver_allocated_memory() / 1024**3
        logger.info("MPS %s - allocated %.2f GB | reserved %.2f GB", label, alloc, resv)
    elif torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        logger.info("CUDA %s - allocated %.2f GB | reserved %.2f GB", label, alloc, resv)


def _hard_memory_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()


def process_training_job_sync(job_id: int, payload: Dict[str, Any]) -> Optional[Dict]:
    """
    Run a fine-tune synchronously.

    Returns:
        dict on success, None on cancelled / failure.
    """
    logger.info("Processing training job %s: %s", job_id, payload)
    _device_memory_log("before-job")
    trainer = HFTrainer()
    ft_job_id: str = payload.get("job_id", "")
    if not ft_job_id:
        logger.error("Missing 'job_id' in training payload")
        return None
    train_file = Path(ServiceConfig.DATA_DIR, "uploads", payload["training_file"]).resolve()

    def check_cancel() -> Optional[str]:
        try:
            with _session() as client:
                r = client.get(f"{BASE_URL}/v1/fine_tuning/jobs/{ft_job_id}")
                if r.status_code == 200:
                    status = r.json().get("status")
                    if status in {"cancelled", "cancelling"}:
                        return "cancelling"
                    if status == "pausing":
                        return "pausing"
        except Exception as exc:
            logger.debug("Cancel-poll failed: %s", exc)
        return None

    def progress(level: str, msg: str, data: Dict | None = None) -> None:
        getattr(logger, level if level in ("warning", "error") else "info")("Training %s - %s", ft_job_id, msg)
        # Map levels to valid OpenAI API values
        valid_level = "warn" if level == "warning" else level if level in ("info", "error") else "info"
        _post_webhook(
            "job.progress", "training", job_id, ft_job_id, {"level": valid_level, "message": msg, **(data or {})}
        )

    _post_webhook("job.running", "training", job_id, ft_job_id)
    hyper = payload.get("hyperparameters", {})
    if "suffix" in payload:
        hyper["suffix"] = payload["suffix"]
    try:
        result = trainer.train(
            ft_job_id,
            payload["model"],
            train_file,
            hyper,
            check_cancel,
            progress,
        )
    except Exception as exc:
        logger.exception("Training crashed")
        _post_webhook(
            "job.failed", "training", job_id, ft_job_id, {"error": {"message": str(exc), "code": "job_error"}}
        )
        raise
    finally:
        if hasattr(trainer, "cleanup") and callable(getattr(trainer, "cleanup", None)):
            try:
                trainer.cleanup()
            except AttributeError:
                pass
        _hard_memory_cleanup()
        _device_memory_log("after-cleanup")
    if result.get("success"):
        data = {
            "fine_tuned_model": result["model_name"],
            "trained_tokens": int(result.get("tokens_processed", 0)),
            "final_loss": result.get("final_loss"),
            "steps": result.get("steps"),
        }
        _post_webhook("job.completed", "training", job_id, ft_job_id, data)
        return {
            "model_id": result["model_name"],
            "status": "completed",
            "trained_tokens": data["trained_tokens"],
        }
    if result.get("cancelled"):
        _post_webhook("job.cancelled", "training", job_id, ft_job_id)
    elif result.get("paused"):
        pass
    else:
        _post_webhook(
            "job.failed",
            "training",
            job_id,
            ft_job_id,
            {"error": {"message": result.get("error", "Training failed"), "code": "training_failed"}},
        )
    return None


def _schedule_poller(scheduler: BackgroundScheduler) -> None:
    """Register a polling job that fetches new work every POLL_INTERVAL seconds."""

    processors: Dict[str, Callable[[int, Dict], Optional[Dict]]] = {
        "training": process_training_job_sync,
    }

    def poll() -> None:
        with _session() as client:
            try:
                r = client.get(f"{BASE_URL}/v1/jobs", params={"status": "queued", "type": "training", "limit": 10})
                for job in r.json().get("data", []):
                    jid = str(job["id"])
                    if scheduler.get_job(jid):
                        continue
                    job_detail = client.get(f"{BASE_URL}/v1/jobs/{jid}").json()
                    jtype = job_detail["type"]
                    if jtype in processors:
                        executor = "training" if jtype == "training" else "default"
                        scheduler.add_job(
                            processors[jtype],
                            args=[int(jid), job_detail["payload"]],
                            id=jid,
                            executor=executor,
                        )
                        logger.info("Scheduled %s job %s", jtype, jid)
            except Exception as exc:
                logger.error("Polling error: %s", exc)

    scheduler.add_job(poll, "interval", seconds=POLL_INTERVAL, id="poller")
    poll()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info(
        "Training Worker starting - max training=%d",
        ServiceConfig.MAX_CONCURRENT_TRAINING,
    )

    executors = {
        "default": ThreadPoolExecutor(max_workers=1),
        "training": ThreadPoolExecutor(max_workers=ServiceConfig.MAX_CONCURRENT_TRAINING),
    }
    scheduler = BackgroundScheduler(executors=executors)
    scheduler.start()
    _schedule_poller(scheduler)

    # graceful shutdown
    stop_flag: Dict[str, bool] = {"quit": False}

    def _graceful(*_: Any) -> None:
        stop_flag["quit"] = True

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)
    logger.info("Worker ready - polling every %ds", POLL_INTERVAL)

    try:
        while not stop_flag["quit"]:
            time.sleep(1)
    finally:
        logger.info("Shutting down...")
        scheduler.shutdown(wait=True)
        logger.info("Goodbye")


if __name__ == "__main__":
    main()
