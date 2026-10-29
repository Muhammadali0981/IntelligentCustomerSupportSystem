"""Sentence-transformers embedder, configured from the active profile.

Queries use the model's query instruction prompt (Qwen3-Embedding); documents do not.
Vectors are L2-normalised, so dot product = cosine similarity.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import MODELS_CACHE, Config


class Embedder:
    def __init__(self, cfg: Config, device: str | None = None):
        e = cfg.embedder()
        self.hf_id: str = e["hf_id"]
        self.query_prompt: str | None = e.get("query_prompt_name")
        self.batch_size: int = e["batch_size"]
        self.device = device or e["query_device"]
        self.model = SentenceTransformer(self.hf_id, device=self.device, cache_folder=str(MODELS_CACHE))

    def encode_docs(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        kw = {"prompt_name": self.query_prompt} if self.query_prompt else {}
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress, **kw)

    def free(self) -> None:
        """Release GPU memory after bulk encoding."""
        import torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
