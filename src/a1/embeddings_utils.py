"""
Pure embedding utility functions with no external dependencies.

Provides:
- _pseudo_embed: Deterministic pseudo-embeddings for testing/fallback
- _stringify_item: Convert any item to JSON string for embedding
"""

import hashlib
import json
from typing import Any

import numpy as np

__all__ = ["_stringify_item", "_pseudo_embed"]


def _stringify_item(x: Any) -> str:
    """Convert any item to a JSON string for embedding."""
    try:
        return json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o)), sort_keys=True)
    except Exception:
        return str(x)


def _pseudo_embed(text: str, dim: int = 512) -> np.ndarray:
    """
    Deterministic pseudo-embedding based on SHA256 digest.

    Returns normalized vector in R^dim. Not a real semantic embedding,
    but deterministic and fast for testing/fallback when no API key.

    Args:
        text: Text to embed
        dim: Embedding dimension (default: 512)

    Returns:
        Normalized numpy array of shape (dim,)
    """
    # Use multiple hashes with different salts to get enough entropy
    vals = []
    for offset in range((dim + 31) // 32):  # 32 floats per hash
        salt = f"{offset}:{text}"
        h = hashlib.sha256(salt.encode("utf-8")).digest()
        for i in range(min(32, dim - len(vals))):
            byte = h[i % len(h)]
            vals.append((byte / 255.0) * 2.0 - 1.0)  # Map to [-1, 1]

    # Convert to numpy and normalize
    vec = np.array(vals[:dim], dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec
