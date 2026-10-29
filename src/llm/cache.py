"""Disk cache for LLM calls, keyed on model + messages + settings.

One JSON file per call under .cache/llm/<aa>/<hash>.json, so reruns are cheap
and the cache survives crashes. Timing runs must bypass it (use_cache=False).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import ROOT

CACHE_DIR = ROOT / ".cache" / "llm"


def make_key(payload: dict[str, Any]) -> str:
    """Stable hash of everything that affects the model output."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def get(key: str) -> dict[str, Any] | None:
    p = _path(key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def put(key: str, value: dict[str, Any]) -> None:
    p = _path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
