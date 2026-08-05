"""
rag.embedder — singleton wrapper around the all-MiniLM-L6-v2 model.

The model is loaded once on first call and cached in the module-level
``_MODEL`` variable.  Subsequent calls reuse the same object, avoiding the
~350 ms cold-start penalty on every embedding request.

Model choice rationale
----------------------
* all-MiniLM-L6-v2 produces 384-dimensional vectors.
* It runs entirely on CPU, has no API key dependency, and its weights are
  cached locally by HuggingFace on first download (~90 MB).
* Cosine similarity on 384-dim vectors is fast enough for our expected
  corpus size (thousands of chunks, not millions).
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

_MODEL: SentenceTransformer | None = None
_MODEL_LOCK = threading.Lock()


def get_embedder() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer instance, loading it on first call.

    Thread-safe: uses a module-level lock so that two concurrent requests
    cannot both trigger the ~350 ms model load simultaneously.

    Returns:
        The loaded SentenceTransformer model.

    Raises:
        RuntimeError: If the model fails to load (e.g. disk full, bad cache).
    """
    global _MODEL  # noqa: PLW0603

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        # Double-checked locking: another thread may have loaded it while we
        # waited for the lock.
        if _MODEL is not None:
            return _MODEL

        logger.info("Loading embedding model '%s' (first call)...", _MODEL_NAME)
        try:
            _MODEL = SentenceTransformer(_MODEL_NAME)
            logger.info("Embedding model loaded — dim=%d.", _EMBEDDING_DIM)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{_MODEL_NAME}': {exc}"
            ) from exc

    return _MODEL


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """
    Embed a sequence of strings and return a 2-D float32 array.

    Args:
        texts: One or more strings to embed.  Empty strings are allowed but
            will produce a zero vector in the output.

    Returns:
        A numpy array of shape ``(len(texts), 384)`` and dtype ``float32``.
        Each row is the L2-normalised embedding of the corresponding input.

    Raises:
        ValueError: If *texts* is empty.
    """
    if not texts:
        raise ValueError("embed_texts requires at least one text string.")

    model = get_embedder()
    # normalize_embeddings=True → cosine sim == dot product, which pgvector
    # uses internally for vector_cosine_ops — results are consistent.
    embeddings: np.ndarray = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return embeddings.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """
    Convenience wrapper — embed a single string and return a 1-D array.

    Args:
        text: The string to embed.

    Returns:
        A numpy array of shape ``(384,)`` and dtype ``float32``.
    """
    return embed_texts([text])[0]
