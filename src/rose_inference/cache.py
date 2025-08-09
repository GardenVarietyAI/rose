import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from rose_inference.loader import ModelLoaderParams, load_model, load_tokenizer, unload_model

logger = logging.getLogger(__name__)


class CachedModel:
    def __init__(
        self,
        model_id: str,
        model: Any,
        tokenizer: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
    ):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.ttl = ttl_seconds if ttl_seconds is not None else int(os.getenv("ROSE_INFERENCE_CACHE_SECONDS", "300"))

        self.active_uses = 0
        self._lock = asyncio.Lock()
        self._evicted = False
        self._expiry_task: Optional[asyncio.Task[None]] = None

    def _schedule_expiry(self) -> None:
        """Schedule eviction after TTL if idle."""
        if self._expiry_task and not self._expiry_task.done():
            self._expiry_task.cancel()
        self._expiry_task = asyncio.create_task(self._expire_after_ttl())

    async def _expire_after_ttl(self) -> None:
        try:
            await asyncio.sleep(self.ttl)
            if self.active_uses == 0:
                await self.evict()
        except asyncio.CancelledError:
            pass

    @asynccontextmanager
    async def use(self) -> AsyncGenerator[Any, Any]:
        """Mark model as in use, cancel expiry timer, then release when done."""
        if self.is_evicted:
            raise RuntimeError(f"CachedModel {self.model_id} is evicted")

        async with self._lock:
            self.active_uses += 1
            if self._expiry_task:
                self._expiry_task.cancel()
                self._expiry_task = None
        try:
            yield
        finally:
            async with self._lock:
                self.active_uses = max(0, self.active_uses - 1)
                if self.active_uses == 0:
                    self._schedule_expiry()

    async def evict(self) -> None:
        if self._evicted:
            return

        if self._expiry_task and not self._expiry_task.done():
            self._expiry_task.cancel()
            self._expiry_task = None

        model = self.model
        tokenizer = self.tokenizer
        self.model = None
        self.tokenizer = None

        try:
            tokenizer.close()  # type: ignore[union-attr]
        except Exception:
            logger.debug("tokenizer.close() failed", exc_info=True)

        asyncio.create_task(asyncio.to_thread(unload_model, model))
        self._evicted = True

    @property
    def is_evicted(self) -> bool:
        return self.model is None


class CachedModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, CachedModel] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def add(self, model: CachedModel) -> CachedModel:
        self._models[model.model_id] = model
        return self._models[model.model_id]

    async def get_or_load(self, load_params: ModelLoaderParams) -> CachedModel:
        self._purge()
        if (m := self._models.get(load_params.model_id)) and not m.is_evicted:
            return m
        lock = self._locks.setdefault(load_params.model_id, asyncio.Lock())
        async with lock:
            self._purge()
            if (m := self._models.get(load_params.model_id)) and not m.is_evicted:
                return m
            logger.info("Cache miss, loading model: %s", load_params.model_name)
            model = await asyncio.to_thread(load_model, load_params)
            tokenizer = await asyncio.to_thread(
                load_tokenizer, model_name=load_params.model_name, data_dir=load_params.data_dir
            )
            return self.add(CachedModel(load_params.model_id, model, tokenizer))

    def get(self, model_id: str) -> Optional[CachedModel]:
        # Purge stale entries first
        self._purge()
        model = self._models.get(model_id)
        return model if model and not model.is_evicted else None

    def all(self) -> list[CachedModel]:
        # Purge stale entries before returning
        self._purge()
        return list(self._models.values())

    async def remove(self, model_id: str) -> None:
        if model := self._models.pop(model_id, None):
            await model.evict()

    async def evict(self) -> None:
        for m in self._models.values():
            await m.evict()
        self._models.clear()

    def _purge(self) -> None:
        dead_keys = [k for k, m in self._models.items() if m.is_evicted]
        for k in dead_keys:
            self._models.pop(k, None)
