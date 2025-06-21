# rose_server/evals/data_source_loader.py
"""Dataset loader used by the evaluation runner."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from rose_worker.client import get_client

logger = logging.getLogger(__name__)

_SAMPLE = Dict[str, str]  # {"input": "...", "expected": "..."}


def _take(seq: Sequence[_SAMPLE], k: Optional[int]) -> List[_SAMPLE]:
    """Return a copy of *seq* limited to *k* elements (or all if *k* is None)."""
    return list(seq if k is None else seq[:k])


class DataSourceLoader:
    """Load evaluation datasets from stored files or inline content"""

    def load_dataset(
        self,
        data_source_config: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> List[_SAMPLE]:
        """Resolve the correct backend based on *data_source_config*."""
        src_type = data_source_config.get("type")
        if src_type == "stored_completions":
            file_id = data_source_config.get("completion_tag_suffix")
            if not file_id:
                raise ValueError("stored_completions requires 'completion_tag_suffix'")
            return self._from_file(file_id, max_samples)

        if src_type == "file_content":
            return self._from_content(data_source_config.get("content", []), max_samples)

        raise ValueError(f"Unsupported data source type: {src_type!r}")

    @staticmethod
    def _extract_item(raw: Dict[str, Any]) -> _SAMPLE:
        """Extract item from ROSE's standard format."""
        item = raw["item"]
        return {
            "input": item["input"],
            "expected": item["expected"],
        }

    def _from_content(
        self,
        content: Sequence[Dict[str, Any]],
        max_samples: Optional[int],
    ) -> List[_SAMPLE]:
        samples = [self._extract_item(obj) for obj in content]
        out = _take(samples, max_samples)
        logger.info("Loaded %d sample(s) from inline content", len(out))
        return out

    def _from_file(self, file_id: str, max_samples: Optional[int]) -> List[_SAMPLE]:
        """Download file via API, then parse JSONL or JSON-array."""
        try:
            raw_bytes = get_client().get_file_content(file_id)
        except FileNotFoundError:
            raise

        if not raw_bytes:
            raise FileNotFoundError(f"File '{file_id}' returned empty content")

        text = raw_bytes.decode("utf-8").strip()
        samples: List[_SAMPLE] = []

        for ln, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                samples.append(self._extract_item(json.loads(line)))
            except json.JSONDecodeError:
                raise
            if max_samples and len(samples) >= max_samples:
                break

        out = _take(samples, max_samples)
        logger.info("Loaded %d sample(s) from file '%s'", len(out), file_id)
        return out
