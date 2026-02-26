"""
FHRR: Fourier Holographic Reduced Representations
===================================================
Complex-valued VSA vectors for differentiable operations.

FHRR is an alternative to binary HyperVectors that represents each dimension as
a unit-magnitude complex number (phasor).  Key properties:

- **Binding** = element-wise complex multiplication (exact inverse via conjugate).
- **Bundling** = element-wise sum of phasors, then re-normalise to unit circle.
- **Similarity** = cosine of magnitude vectors; ``gradient_similarity`` uses the
  real/imag decomposition for gradient-friendly computation.
- **Scalar encoding** via golden-ratio–based phase offsets gives quasi-orthogonal
  vectors for distinct scalar values.

Integration points:
- ``FHRRVector`` and ``FHRRMemory`` are standalone and do not depend on the Rust
  backend.  They can be used alongside or instead of ``HyperVector`` for tasks
  that require differentiable or invertible VSA operations.
- ``CognitiveEngine`` can use FHRR for plan-step encoding when gradient-based
  optimisation of symbolic sequences is needed.
"""
from __future__ import annotations

import sys
import os
import math
from typing import List, Tuple, Optional

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
# Public alias — part of the deterministic scalar encoding contract
GOLDEN_RATIO = _GOLDEN_RATIO


class FHRRVector:
    """
    FHRR phasor vector: each component is a unit-magnitude complex number.
    Binding = element-wise multiplication; bundling = sum + normalize.
    """

    def __init__(self, dim: int = 1024, seed: Optional[int] = None):
        self.dim = dim
        if seed is not None:
            rng = np.random.default_rng(seed)
            phases = rng.uniform(-np.pi, np.pi, dim)
        else:
            phases = np.random.uniform(-np.pi, np.pi, dim)
        self.phasors = np.exp(1j * phases).astype(np.complex128)

    @staticmethod
    def from_phasors(phasors: np.ndarray) -> "FHRRVector":
        """Create from existing phasor array."""
        v = FHRRVector.__new__(FHRRVector)
        v.phasors = np.asarray(phasors, dtype=np.complex128)
        v.dim = len(v.phasors)
        return v

    def bind(self, other: "FHRRVector") -> "FHRRVector":
        """Element-wise complex multiplication (circular convolution in freq domain)."""
        return FHRRVector.from_phasors(self.phasors * other.phasors)

    def unbind(self, other: "FHRRVector") -> "FHRRVector":
        """Element-wise complex conjugate multiplication (inverse bind)."""
        return FHRRVector.from_phasors(self.phasors * np.conj(other.phasors))

    def bundle(self, others: List["FHRRVector"]) -> "FHRRVector":
        """Element-wise sum of phasors, then normalize phases to unit circle."""
        total = self.phasors.copy()
        for o in others:
            total = total + o.phasors
        # Normalize each component to unit magnitude
        magnitudes = np.abs(total)
        magnitudes[magnitudes == 0] = 1.0
        normalized = total / magnitudes
        return FHRRVector.from_phasors(normalized)

    def similarity(self, other: "FHRRVector") -> float:
        """Cosine similarity of phasor magnitudes."""
        a = np.abs(self.phasors)
        b = np.abs(other.phasors)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def gradient_similarity(self, other: "FHRRVector") -> float:
        """Differentiable cosine similarity using real/imag parts."""
        a = np.concatenate([self.phasors.real, self.phasors.imag])
        b = np.concatenate([other.phasors.real, other.phasors.imag])
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def to_gradient_input(self, requires_grad: bool = True) -> np.ndarray:
        """Return phasors as float array [real, imag] concatenated."""
        return np.concatenate([self.phasors.real, self.phasors.imag]).astype(np.float64)

    @staticmethod
    def from_gradient_output(arr: np.ndarray, dim: int) -> "FHRRVector":
        """Reconstruct from [real, imag] float array."""
        real = arr[:dim]
        imag = arr[dim:2 * dim]
        phasors = (real + 1j * imag).astype(np.complex128)
        # Normalize to unit phasors
        magnitudes = np.abs(phasors)
        magnitudes[magnitudes == 0] = 1.0
        return FHRRVector.from_phasors(phasors / magnitudes)

    def encode_scalar(self, x: float) -> "FHRRVector":
        """Deterministic phasor encoding of scalar: phase = 2π * frac(x * φ)."""
        frac = (x * _GOLDEN_RATIO) % 1.0
        phases = np.array(
            [2 * np.pi * ((frac * (i + 1)) % 1.0) for i in range(self.dim)],
            dtype=np.float64,
        )
        return FHRRVector.from_phasors(np.exp(1j * phases))

    @staticmethod
    def encode_symbol(name: str, dim: int = 1024) -> "FHRRVector":
        """Hash-based deterministic phasor encoding of a symbol name."""
        import hashlib
        h = hashlib.sha256(name.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        phases = rng.uniform(-np.pi, np.pi, dim)
        v = FHRRVector.__new__(FHRRVector)
        v.dim = dim
        v.phasors = np.exp(1j * phases).astype(np.complex128)
        return v


class FHRRMemory:
    """Associative memory using FHRR phasors."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self._composite: Optional[FHRRVector] = None
        self._key_hvs: dict = {}

    def _make_key_hv(self, key: str) -> FHRRVector:
        if key not in self._key_hvs:
            self._key_hvs[key] = FHRRVector.encode_symbol(key, self.dim)
        return self._key_hvs[key]

    def store(self, key: str, vector: FHRRVector):
        """Bind key_hv with vector and bundle into composite memory."""
        key_hv = self._make_key_hv(key)
        bound = key_hv.bind(vector)
        if self._composite is None:
            self._composite = bound
        else:
            self._composite = self._composite.bundle([bound])

    def retrieve(self, key: str) -> FHRRVector:
        """Unbind key from composite memory trace."""
        if self._composite is None:
            return FHRRVector(self.dim)
        key_hv = self._make_key_hv(key)
        return self._composite.unbind(key_hv)

    def cleanup(
        self,
        query: FHRRVector,
        codebook: List[Tuple[str, FHRRVector]],
    ) -> Tuple[str, float]:
        """Find closest match in codebook."""
        best_name = ""
        best_sim = -1.0
        for name, hv in codebook:
            sim = query.similarity(hv)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return (best_name, best_sim)
