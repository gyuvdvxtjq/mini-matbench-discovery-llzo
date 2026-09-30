"""Jsonl checkpoint store: the unit of resumability.

Every record is keyed by (model, protocol, params-hash, material_id), so
switching models or protocol parameters never collides with cached results:
a changed parameter simply produces a new key and a fresh computation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def params_hash(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def make_key(model: str, protocol: str, params: dict[str, Any], material_id: str) -> str:
    return f"{model}|{protocol}|{params_hash(params)}|{material_id}"


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["_key"]] = record

    def get_ok(self, key: str) -> dict[str, Any] | None:
        record = self.records.get(key)
        if record and record.get("status") == "ok":
            return record
        return None

    def append(self, key: str, record: dict[str, Any]) -> None:
        record = {"_key": key, **record}
        self.records[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            )
