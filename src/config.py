"""Load project configuration and the active hardware profile.

Usage:
    from src.config import load_config
    cfg = load_config()            # profile from $SUPPORT_PROFILE, default "home"
    cfg = load_config("lab")
    cfg.route("fast")              # -> {"model": ..., "tag": ..., "think": ..., "num_ctx": ...}
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# The global env also has TensorFlow + Keras 3, which transformers 4.57 cannot
# import. We only use PyTorch, so tell transformers to skip TensorFlow.
os.environ.setdefault("USE_TF", "0")
# Windows without Developer Mode has no symlinks; HF falls back to copies. Fine, just noisy.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
MODELS_CACHE = ROOT / "models" / "cache"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """All configuration for one run: models, thresholds, datasets and the profile."""

    profile: dict[str, Any]
    models: dict[str, Any]
    thresholds: dict[str, Any]
    datasets: dict[str, Any]

    @property
    def profile_name(self) -> str:
        return self.profile["name"]

    def route(self, name: str) -> dict[str, Any]:
        """Return the route settings merged with the model registry entry."""
        route = dict(self.profile["routes"][name])
        route.update({"model_key": route["model"], **self.models["llm"][route["model"]]})
        return route

    def embedder(self) -> dict[str, Any]:
        emb = dict(self.profile["embedding"])
        return {**emb, **self.models["embedders"][emb["model"]]}

    def reranker(self, name: str | None = None) -> dict[str, Any]:
        rr = dict(self.profile["reranker"])
        key = name or rr["model"]
        return {**rr, "model": key, **self.models["rerankers"][key]}

    def hardware_stamp(self) -> dict[str, Any]:
        """Hardware + model facts recorded in every results file."""
        return {
            "profile": self.profile_name,
            "hardware": self.profile["hardware"],
            "routes": {k: {"tag": self.route(k)["tag"], "quantization": self.route(k).get("quantization"),
                           "think": self.route(k)["think"], "num_ctx": self.route(k)["num_ctx"]}
                       for k in self.profile["routes"]},
            "embedder": self.embedder()["hf_id"],
            "reranker": self.reranker()["hf_id"],
        }


def load_config(profile: str | None = None) -> Config:
    """Load configs/*.yaml plus configs/profiles/<profile>.yaml."""
    load_dotenv(ROOT / ".env")
    profile = profile or os.environ.get("SUPPORT_PROFILE", "home")
    os.environ.setdefault("HF_HOME", str(MODELS_CACHE))
    return Config(
        profile=_read_yaml(CONFIG_DIR / "profiles" / f"{profile}.yaml"),
        models=_read_yaml(CONFIG_DIR / "models.yaml"),
        thresholds=_read_yaml(CONFIG_DIR / "thresholds.yaml"),
        datasets=_read_yaml(CONFIG_DIR / "datasets.yaml"),
    )
