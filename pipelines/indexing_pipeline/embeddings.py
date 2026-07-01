from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBEDDING_PROVIDER = None  # Uses Hugging Face Inference API by default
DEFAULT_BATCH_SIZE = 8


def _load_env_value(name: str, env_path: Path = ENV_FILE_PATH) -> str | None:
    """Read a secret from the process environment first, then a local .env file."""
    if value := os.getenv(name):
        return value
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            resolved = value.strip().strip('"').strip("'")
            return resolved or None

    return None


class HuggingFaceInferenceEmbeddingProvider:
    """Generate dense embeddings via the Hugging Face Hub Inference API or a selected inference provider."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        *,
        provider: str | None = DEFAULT_EMBEDDING_PROVIDER,
        token: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float | None = None,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for inference-based embeddings. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.provider = provider
        self.batch_size = max(1, batch_size)
        resolved_token = token or _load_env_value("HF_TOKEN")

        self._client = InferenceClient(
            provider=provider,
            token=resolved_token,
            timeout=timeout,
        )

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Return a L2-normalized float32 array of shape (num_texts, embedding_dim)."""
        text_list = [str(text).strip() for text in texts if str(text).strip()]
        if not text_list:
            raise ValueError("Cannot embed an empty list of texts")

        embeddings: list[np.ndarray] = []
        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            batch_embeddings = self._client.feature_extraction(batch, model=self.model_name)
            if not isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = np.asarray(batch_embeddings, dtype="float32")
            else:
                batch_embeddings = batch_embeddings.astype("float32")

            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)
            embeddings.append(batch_embeddings)

        vectors = np.vstack(embeddings)
        faiss.normalize_L2(vectors)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string and return a L2-normalized vector."""
        return self.embed_texts([query])
