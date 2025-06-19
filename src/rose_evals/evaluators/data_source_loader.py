# rose_server/evals/data_source_loader.py
"""Dataset loader used by the evaluation runner."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import httpx

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
        """Normalise one raw JSON object to the canonical schema."""
        item = raw.get("item", raw)  # allow nested `item`
        return {
            "input": item.get("input", ""),
            "expected": item.get("expected", item.get("ground_truth", "")),
        }

    def _from_content(
        self,
        content: Sequence[Dict[str, Any]],
        max_samples: Optional[int],
    ) -> List[_SAMPLE]:
        samples = [_DataSourceLoader._extract_item(obj) for obj in content]
        out = _take(samples, max_samples)
        logger.info("Loaded %d sample(s) from inline content", len(out))
        return out

    def _from_file(self, file_id: str, max_samples: Optional[int]) -> List[_SAMPLE]:
        """Download file via API, then parse JSONL or JSON-array."""
        base_url = os.getenv("ROSE_BASE_URL", "http://localhost:8004")

        with httpx.Client() as client:
            response = client.get(f"{base_url}/v1/files/{file_id}/content")
            if response.status_code == 404:
                raise FileNotFoundError(f"File '{file_id}' not found")
            response.raise_for_status()
            raw_bytes = response.content

        if not raw_bytes:
            raise FileNotFoundError(f"File '{file_id}' returned empty content")

        text = raw_bytes.decode("utf-8").strip()
        samples: List[_SAMPLE] = []

        # Heuristic: JSONL → contains newlines not wrapped by [] / JSON array otherwise
        if text.startswith("["):
            try:
                data_list = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"File {file_id} is not valid JSON") from exc
            samples = [self._extract_item(obj) for obj in data_list]

        else:  # JSONL
            for ln, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    samples.append(self._extract_item(json.loads(line)))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping line %d in %s: %s", ln, file_id, exc)
                if max_samples and len(samples) >= max_samples:
                    break

        out = _take(samples, max_samples)
        logger.info("Loaded %d sample(s) from file '%s'", len(out), file_id)
        return out


# make helper accessible for unit tests without exporting the whole class
_DataSourceLoader = DataSourceLoader
