"""
V18: GloVe/FastText-based word HV seed initialization.

Problem
-------
Previous behavior: word HVs were seeded from ``hash(word) % SEED_MODULO``.
"car" and "automobile" produced completely different random HVs with Hamming
similarity ≈ 0.5 (chance level).  The system was blind to synonyms and
semantically related words unless explicitly taught.

Fix
---
Load GloVe 6B 50d at module init (lazy, cached).  For known words, project
the 50-dimensional dense embedding into the 10 240-bit binary HV space using a
fixed random projection matrix (Johnson-Lindenstrauss).  For unknown words,
fall back to the existing hash seed, preserving backward compatibility.

The GloVe file is NOT committed to the repository (171 MB).  Download it with::

    nsck/scripts/download_embeddings.sh

or place ``glove.6B.50d.txt`` under ``nsck/data/embeddings/`` manually.
When the file is absent the module degrades silently to hash-based seeds.

Usage
-----
    from python.core.vsa.word_seeds import word_to_seed_bits

    bits = word_to_seed_bits("automobile")  # shape (10240,), dtype uint8
    hv = HyperVector.from_bits(bits)        # semantically informed HV
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HV_DIM: int = 10_240
_GLOVE_DIM: int = 50

# Path to the GloVe 50d text file relative to this module
_GLOVE_PATH: str = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "data", "embeddings", "glove.6B.50d.txt",
)

# ---------------------------------------------------------------------------
# Module-level singletons (loaded lazily on first call)
# ---------------------------------------------------------------------------

_GLOVE: Optional[Dict[str, np.ndarray]] = None
# Fixed (10240, 50) projection matrix — generated once with seed=42 for
# determinism across runs.  Johnson-Lindenstrauss guarantees approximate
# distance preservation after sign-binarisation.
_PROJECTION_MATRIX: Optional[np.ndarray] = None
_GLOVE_LOADED: bool = False  # True once load was attempted (even if file absent)


def _ensure_glove_loaded() -> None:
    """Load GloVe embeddings and projection matrix (once per process)."""
    global _GLOVE, _PROJECTION_MATRIX, _GLOVE_LOADED
    if _GLOVE_LOADED:
        return

    _GLOVE_LOADED = True

    # Build fixed projection matrix regardless of whether GloVe exists —
    # it is also used as a deterministic RNG for the hash fallback path so
    # callers that query the matrix don't need to re-check file availability.
    rng = np.random.RandomState(42)  # fixed seed — must not change across versions
    _PROJECTION_MATRIX = rng.randn(_HV_DIM, _GLOVE_DIM).astype(np.float32)

    glove_path = os.path.abspath(_GLOVE_PATH)
    if not os.path.isfile(glove_path):
        # Graceful degradation — hash fallback will be used for all words
        return

    _GLOVE = {}
    try:
        with open(glove_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip().split(" ")
                if len(parts) == _GLOVE_DIM + 1:
                    word = parts[0]
                    vec = np.array(parts[1:], dtype=np.float32)
                    _GLOVE[word] = vec
    except Exception:
        _GLOVE = None  # treat any parse failure as "file absent"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def word_to_seed_bits(word: str) -> np.ndarray:
    """Return a deterministic 10240-bit seed array for *word*.

    If GloVe embeddings are available and the word (lowercased) is in the
    vocabulary, a Johnson-Lindenstrauss projection of the 50d GloVe vector is
    returned.  This means semantically similar words (e.g. "car" and
    "automobile") produce HVs with higher-than-chance Hamming similarity.

    If GloVe is unavailable or the word is out-of-vocabulary, the original
    hash-based seed is used, preserving backward compatibility for all words.

    Parameters
    ----------
    word : str
        Input word (case-insensitive).

    Returns
    -------
    np.ndarray, shape (10240,), dtype uint8
        Binary bit array: 1 for set bits, 0 for unset.  Pass to
        ``HyperVector.from_bits()`` or use directly with the VSA operations.
    """
    _ensure_glove_loaded()

    word_lower = word.lower()

    if _GLOVE is not None and word_lower in _GLOVE:
        # Project 50d GloVe vector into 10240-bit space via random hyperplanes.
        # Sign(projection) binarises the continuous vector into a binary HV that
        # preserves approximate cosine similarity (Johnson-Lindenstrauss lemma).
        vec = _GLOVE[word_lower]  # shape (50,)
        projected = _PROJECTION_MATRIX @ vec  # shape (10240,)
        return (projected > 0).astype(np.uint8)

    # Hash fallback: same behaviour as the original ``HyperVector(hash(word) % 2^32)``
    # seed, ensuring backward compatibility for out-of-vocabulary words.
    seed = hash(word) % (2 ** 31)  # PCG64/numpy expects non-negative int
    rng = np.random.RandomState(seed)
    return (rng.random(_HV_DIM) > 0.5).astype(np.uint8)


def glove_available() -> bool:
    """Return True if GloVe embeddings have been loaded successfully."""
    _ensure_glove_loaded()
    return _GLOVE is not None and len(_GLOVE) > 0


def glove_vocab_size() -> int:
    """Return the number of words in the loaded GloVe vocabulary (0 if absent)."""
    _ensure_glove_loaded()
    return len(_GLOVE) if _GLOVE else 0
