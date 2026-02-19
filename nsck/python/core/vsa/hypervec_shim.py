"""Compatibility shim for the ``hypervec_rs`` Rust extension.

Tries to load the compiled Rust extension first.  When it is not available
(e.g. in CI or on a machine without the compiled binary) it falls back
transparently to the pure-Python implementation in ``hypervec_py``.

Modules throughout the codebase should import via::

    import python.core.vsa.hypervec_shim as hypervec_rs

This guarantees they get the fastest available backend without crashing
when the Rust build is missing.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import types
from typing import Any, Optional

from python.core.vsa.hypervec_py import HyperVectorPy as _FallbackHV

_USE_RUST = False
_ext: types.ModuleType | None = None

try:
    import hypervec_rs as _ext  # type: ignore[no-redef]  # compiled Rust extension
    _USE_RUST = True
except ImportError:
    pass

if not _USE_RUST:
    if multiprocessing.current_process().name == "MainProcess":
        print(">> [VSA] Using Python Fallback (via shim)")


def _hv_repr_bytes(hv) -> bytes:
    return repr(hv).encode("utf-8")


def _install_compat_methods(HV: Any) -> None:
    """Add helper methods that may be missing from the Rust class."""
    import numpy as _np

    # --- bits property (u64 blocks ↔ int8 numpy array) ---
    if not hasattr(HV, "bits"):
        @property
        def _bits_property(self):
            """Unpack 160 u64 ints into a 10240-element int8 numpy array."""
            state = self.__getstate__()  # list of 160 u64 ints
            arr = _np.zeros(10240, dtype=_np.int8)
            for word_idx, word in enumerate(state):
                for bit_pos in range(64):
                    if word & (1 << bit_pos):
                        arr[word_idx * 64 + bit_pos] = 1
            return arr

        try:
            HV.bits = _bits_property
        except (TypeError, AttributeError):
            pass  # Rust class may not allow property assignment

    # --- from_bits classmethod ---
    if not hasattr(HV, "from_bits"):
        @classmethod
        def from_bits(cls, bits_array):
            """Create a HyperVector from a numpy int8 bit array."""
            bits_arr = _np.asarray(bits_array, dtype=_np.int8)
            num_u64 = len(bits_arr) // 64
            state = [0] * num_u64
            for word_idx in range(num_u64):
                word = 0
                for bit_pos in range(64):
                    if bits_arr[word_idx * 64 + bit_pos]:
                        word |= (1 << bit_pos)
                state[word_idx] = word
            obj = cls.__new__(cls)
            obj.__setstate__(state)
            return obj

        try:
            HV.from_bits = from_bits
        except (TypeError, AttributeError):
            pass

    # --- weighted_bundle ---
    if not hasattr(HV, "weighted_bundle"):
        def weighted_bundle(self, other, weight: float, seed: Optional[int] = None):
            w = max(0.0, min(1.0, float(weight)))
            k = 7
            self_n = max(1, int(round(w * k)))
            other_n = max(1, k - self_n)

            out = self
            for _ in range(self_n - 1):
                out = out.bundle(self)
            for _ in range(other_n):
                out = out.bundle(other)
            return out

        setattr(HV, "weighted_bundle", weighted_bundle)

    # --- lsh_hash ---
    if not hasattr(HV, "lsh_hash"):
        def lsh_hash(self, seed: int, n_bits: int) -> int:
            h = hashlib.blake2b(_hv_repr_bytes(self), digest_size=8)
            sig64 = int.from_bytes(h.digest(), "little")
            n = int(n_bits)
            if n >= 64:
                return sig64
            return sig64 & ((1 << n) - 1)

        setattr(HV, "lsh_hash", lsh_hash)

    # --- permute (Phase 8: Temporal Encoding) ---
    if not hasattr(HV, "permute"):
        import numpy as _np

        def permute(self, shift: int):
            """Circular bitwise permutation for temporal encoding."""
            dim = 10240
            shift_norm = shift % dim
            if shift_norm == 0:
                return self  # identity

            # For Rust HVs, get the u64 block state via pickle interface
            try:
                state = self.__getstate__()  # Returns list of 160 u64 ints
            except Exception:
                # Should not happen — indicates a broken HV object
                return self

            num_u64 = dim // 64
            total_bits = dim

            s = ((shift_norm % total_bits) + total_bits) % total_bits
            word_shift = s // 64
            bit_shift = s % 64

            new_bits = [0] * num_u64
            if bit_shift == 0:
                for i in range(num_u64):
                    src = (i + num_u64 - word_shift) % num_u64
                    new_bits[i] = state[src]
            else:
                complement = 64 - bit_shift
                mask = (1 << 64) - 1
                for i in range(num_u64):
                    src_hi = (i + num_u64 - word_shift) % num_u64
                    src_lo = (i + num_u64 - word_shift - 1) % num_u64
                    new_bits[i] = ((state[src_hi] << bit_shift) | (state[src_lo] >> complement)) & mask

            result = object.__new__(type(self))
            result.__setstate__(new_bits)  # type: ignore[attr-defined]
            return result

        setattr(HV, "permute", permute)

    if not hasattr(HV, "permute_inverse"):
        def permute_inverse(self, shift: int):
            """Inverse permutation: permute(-shift)."""
            return self.permute(-shift)

        setattr(HV, "permute_inverse", permute_inverse)
    
    # --- cosine_similarity (noise-robust comparison) ---
    if not hasattr(HV, "cosine_similarity"):
        import numpy as _np
        
        def cosine_similarity(self, other):
            """
            Cosine similarity on bipolar representation (more robust to noise).
            
            Converts binary {0,1} to bipolar {-1,+1} then computes cosine.
            This is more robust to bit flips than raw Hamming distance.
            
            Returns:
                float in [-1, 1]: 1 = identical, 0 = orthogonal, -1 = opposite
            """
            # Get bits arrays
            try:
                # For Rust HV: use the bits property
                if hasattr(self, 'bits'):
                    a_bits = self.bits
                    b_bits = other.bits
                else:
                    # Fallback: reconstruct from state
                    a_state = self.__getstate__()
                    b_state = other.__getstate__()
                    a_bits = _np.zeros(10240, dtype=_np.int8)
                    b_bits = _np.zeros(10240, dtype=_np.int8)
                    for word_idx, word in enumerate(a_state):
                        for bit_pos in range(64):
                            if word & (1 << bit_pos):
                                a_bits[word_idx * 64 + bit_pos] = 1
                    for word_idx, word in enumerate(b_state):
                        for bit_pos in range(64):
                            if word & (1 << bit_pos):
                                b_bits[word_idx * 64 + bit_pos] = 1
            except Exception:
                # Fallback to Hamming if bits extraction fails
                return self.similarity(other)
            
            # Convert to bipolar: 0 → -1, 1 → +1
            a_bipolar = (a_bits.astype(_np.float32) - 0.5) * 2
            b_bipolar = (b_bits.astype(_np.float32) - 0.5) * 2
            
            # Cosine similarity
            dot_product = _np.dot(a_bipolar, b_bipolar)
            norm_a = _np.linalg.norm(a_bipolar)
            norm_b = _np.linalg.norm(b_bipolar)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
        
        setattr(HV, "cosine_similarity", cosine_similarity)
    
    # --- similarity_robust (unified interface) ---
    if not hasattr(HV, "similarity_robust"):
        def similarity_robust(self, other, method='cosine'):
            """
            Noise-robust similarity with configurable method  .
            
            Args:
                other: HyperVector to compare against
                method: 'hamming' (legacy) or 'cosine' (recommended)
            
            Returns:
                float in [0, 1]: similarity score
            """
            if method == 'cosine':
                # Cosine returns [-1, 1], normalize to [0, 1]
                cos_sim = self.cosine_similarity(other)
                return (cos_sim + 1.0) / 2.0
            else:
                # Use legacy Hamming
                return self.similarity(other)
        
        setattr(HV, "similarity_robust", similarity_robust)


if _USE_RUST and _ext is not None:
    _install_compat_methods(_ext.HyperVector)
    HyperVector = _ext.HyperVector
    
    # PATCH: Align permute direction with Python convention
    # Python: permute(1) -> np.roll(-1) (roll left, positions move left)
    # Rust:   permute(1) -> rotate right (positions move right)
    # Fix: Negate shift parameter to match Python's np.roll direction
    
    _rust_permute = HyperVector.permute
    def _aligned_permute(self, shift: int):
        return _rust_permute(self, -shift)
    HyperVector.permute = _aligned_permute
    
    _rust_permute_inverse = HyperVector.permute_inverse
    def _aligned_permute_inverse(self, shift: int):
        return _rust_permute_inverse(self, -shift)
    HyperVector.permute_inverse = _aligned_permute_inverse
    
    # --- Expose all Rust-accelerated classes ---
    SemanticMemoryConcurrent = _ext.SemanticMemoryConcurrent
    EpisodicMemoryConcurrent = _ext.EpisodicMemoryConcurrent
    Episode = _ext.Episode
    CognitiveWorkerPool = _ext.CognitiveWorkerPool
    HyperVectorRegistry = _ext.HyperVectorRegistry
    ActivationAccumulator = _ext.ActivationAccumulator
    PersistentStorage = _ext.PersistentStorage
    AsyncCognitiveRuntime = _ext.AsyncCognitiveRuntime

    # Rust parallel free-functions
    parallel_similarity_search = _ext.parallel_similarity_search
    batch_parallel_similarity_search = _ext.batch_parallel_similarity_search
    parallel_bundle = _ext.parallel_bundle
    run_semantic_search_async = _ext.run_semantic_search_async

    __backend__ = "Rust"
    if multiprocessing.current_process().name == "MainProcess":
        print(">> [VSA] Using Rust Accelerator (hypervec_rs) [Patched Direction]")
else:
    # The Python fallback already has native bits, from_bits, permute, etc.
    # Do NOT call _install_compat_methods — it would overwrite the instance
    # attribute 'bits' with a read-only property designed for the Rust u64 layout.
    HyperVector = _FallbackHV

    # Python fallback stubs for Rust-only classes
    # These are lightweight stand-ins so code can import without crashing
    SemanticMemoryConcurrent = None
    EpisodicMemoryConcurrent = None
    Episode = None
    CognitiveWorkerPool = None
    HyperVectorRegistry = None
    ActivationAccumulator = None
    PersistentStorage = None
    AsyncCognitiveRuntime = None
    parallel_similarity_search = None
    batch_parallel_similarity_search = None
    parallel_bundle = None
    run_semantic_search_async = None

    __backend__ = "Python"

__all__ = [
    "HyperVector",
    "SemanticMemoryConcurrent",
    "EpisodicMemoryConcurrent",
    "Episode",
    "CognitiveWorkerPool",
    "HyperVectorRegistry",
    "ActivationAccumulator",
    "PersistentStorage",
    "AsyncCognitiveRuntime",
    "parallel_similarity_search",
    "batch_parallel_similarity_search",
    "parallel_bundle",
    "run_semantic_search_async",
]
