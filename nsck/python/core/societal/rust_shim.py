"""Optional Rust-accelerated societal backend shim for NSCK V5.

Tries to import LivingHvStore, SocietalHnswRs, SpectralRgRs from the
compiled hypervec_rs extension.  Falls back to None if unavailable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_USE_RUST_SOCIETAL = False
LivingHvStoreRs = None
SocietalHnswRsClass = None
SpectralRgRsClass = None

try:
    import hypervec_rs as _hypervec_rs  # type: ignore[import]

    LivingHvStoreRs = getattr(_hypervec_rs, "LivingHvStore", None)
    SocietalHnswRsClass = getattr(_hypervec_rs, "SocietalHnswRs", None)
    SpectralRgRsClass = getattr(_hypervec_rs, "SpectralRgRs", None)
    if LivingHvStoreRs is not None:
        _USE_RUST_SOCIETAL = True
except ImportError:
    pass


def get_rust_store() -> Optional[object]:
    """Return a new LivingHvStore instance if the Rust backend is available."""
    if LivingHvStoreRs is not None:
        return LivingHvStoreRs()
    return None


def cosine_similarity_matrix_rs(
    vectors: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute NxN cosine similarity matrix via Rust if available.

    Parameters
    ----------
    vectors:
        Float32 array of shape (N, D).

    Returns
    -------
    (N, N) float32 similarity matrix, or ``None`` if Rust is unavailable.
    """
    if SpectralRgRsClass is None:
        return None
    try:
        row_list = [row.astype(float).tolist() for row in vectors]
        result = SpectralRgRsClass.cosine_similarity_matrix(row_list)
        return np.array(result, dtype=np.float32)
    except Exception:
        return None
