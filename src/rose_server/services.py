import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Services:
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, instance: Any) -> None:
        cls._registry[name] = instance
        logger.debug(f"Registered service: {name}")

    @classmethod
    def get(cls, name: str) -> Any:
        if name not in cls._registry:
            raise KeyError(f"Service '{name}' not registered. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_services(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    async def shutdown(cls) -> None:
        job_store = cls._registry.get("job_store")
        if job_store and hasattr(job_store, "shutdown"):
            await job_store.shutdown()
            logger.info("APScheduler shutdown completed")
        logger.info("All services shutdown completed")

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        logger.debug("Services registry cleared")


def get_file_store():
    return Services.get("file_store")


def get_fine_tuning_store():
    """Get the fine-tuning store service."""
    return Services.get("fine_tuning_store")


def get_tokenizer_service():
    """Get the tokenizer service."""
    return Services.get("tokenizer_service")


def get_embedding_manager():
    """Get the embedding manager service."""
    return Services.get("embedding_manager")


def get_vector_store_store():
    """Get the vector store manager service."""
    return Services.get("vector_store_store")


def get_job_store():
    """Get the job store service."""
    return Services.get("job_store")


def get_chromadb_manager():
    """Get the ChromaDB manager service."""
    return Services.get("chromadb_manager")


def get_model_registry():
    """Get the model registry service."""
    return Services.get("model_registry")
