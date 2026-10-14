"""Dense text embeddings with an on-disk cache.

Policy clauses are embedded once and cached by content hash, so restarts and repeated
evaluations cost no API calls; only new queries are embedded at runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import config
from .llm import LLMError, post_with_retry


class GeminiEmbedder:
    name = "gemini"
    batch_size = 50

    def __init__(self, model: str | None = None, dimensions: int = 768, cache_path: Path | None = None):
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimensions = dimensions
        self.cache_path = cache_path or config.ARTIFACTS_DIR / "embedding_cache.json"
        self._cache: dict[str, list[float]] = self._load_cache()

    def _load_cache(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text())
        except (OSError, ValueError):
            return {}

    def _key(self, text: str, task: str) -> str:
        return hashlib.sha256(f"{self.model}|{self.dimensions}|{task}|{text}".encode()).hexdigest()

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t, task) not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            body = {"requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                 "taskType": task, "outputDimensionality": self.dimensions}
                for t in batch
            ]}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
            data = post_with_retry(url, body, {"x-goog-api-key": self.api_key}, "gemini-embeddings")
            vectors = [e["values"] for e in data.get("embeddings", [])]
            if len(vectors) != len(batch):
                raise LLMError("Embedding response size mismatch")
            for text, vec in zip(batch, vectors, strict=True):
                self._cache[self._key(text, task)] = vec
        if missing:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))
        matrix = np.array([self._cache[self._key(t, task)] for t in texts], dtype=float)
        # Truncated (Matryoshka) embeddings must be re-normalised before cosine similarity.
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def get_embedder() -> GeminiEmbedder | None:
    """Return an embedder when one is configured, else None (lexical retrieval only)."""
    if config.RETRIEVAL_MODE == "lexical" or not os.getenv("GEMINI_API_KEY"):
        return None
    return GeminiEmbedder()
