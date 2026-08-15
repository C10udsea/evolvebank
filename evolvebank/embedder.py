"""Local embedding model wrapper.

Strategy retrieval needs to compare "how similar is this new task to past
strategies" -- that is a vector similarity search. We embed strategy texts
with a small local sentence-transformers model (~100MB), so no extra API
key is needed.

In China, set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is
unreachable.
"""

import os

import numpy as np

_MODEL_NAME = os.environ.get("EVOLVEBANK_EMBEDDER", "BAAI/bge-small-en-v1.5")
_model = None  # lazy singleton: loading takes a few seconds


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts -> (n, d) float32 array, L2-normalized."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    vecs = _get_model().encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single text -> (d,) float32 array, L2-normalized."""
    return embed([text])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (already normalized -> dot product)."""
    return float(np.dot(a, b))
