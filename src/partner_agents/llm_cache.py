"""On-disk cache of model replies behind ``LLM_CACHE_DIR`` (ADR-018, HLD §9).

One JSON file per key, written atomically. A missing or corrupt file is a cache miss.
The eval runner reuses this class with its own directory (ADR-014).
"""

import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


def cache_key(*parts: Any) -> str:
    """A stable sha256 over ``parts`` (anything JSON-serialisable, else ``str()``)."""
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    async def get(self, key: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._read, self._path(key))

    async def put(self, key: str, value: dict[str, Any]) -> None:
        await asyncio.to_thread(self._write, self._path(key), value)

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
