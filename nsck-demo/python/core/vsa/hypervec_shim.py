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
from typing import Optional

_USE_RUST = False

try:
    import hypervec_rs as _ext  # compiled Rust extension
    _USE_RUST = True
except ImportError:
    _ext = None

if _ext is None:
    # Fall back to the pure-Python implementation.
    from python.core.vsa.hypervec_py import HyperVectorPy as _FallbackHV  # noqa: E402

    if multiprocessing.current_process().name == "MainProcess":
        print(">> [VSA] Using Python Fallback (via shim)")


def _hv_repr_bytes(hv) -> bytes:
    return repr(hv).encode("utf-8")


def _install_compat_methods(HV):
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

            result = type(self).__new__(type(self))
            result.__setstate__(new_bits)
            return result

        setattr(HV, "permute", permute)

    if not hasattr(HV, "permute_inverse"):
        def permute_inverse(self, shift: int):
            """Inverse permutation: permute(-shift)."""
            return self.permute(-shift)

        setattr(HV, "permute_inverse", permute_inverse)


if _USE_RUST:
    _install_compat_methods(_ext.HyperVector)
    HyperVector = _ext.HyperVector
    
    # PATCH: Align permute direction with Python (Left-Shift for positive)
    # Python: permute(1) -> np.roll(-1) (Left)
    # Rust:   permute(1) -> Right Shift
    # Fix: Negate shift called on Rust
    
    _rust_permute = HyperVector.permute
    def _aligned_permute(self, shift: int):
        return _rust_permute(self, -shift)
    HyperVector.permute = _aligned_permute
    
    _rust_permute_inverse = HyperVector.permute_inverse
    def _aligned_permute_inverse(self, shift: int):
        # permute_inverse(s) = permute(-s)
        # We want Python behavior: permute_inverse(1) -> permute(-1) -> np.roll(1) (Right)
        # Rust behavior: permute_inverse(1) -> permute(-1) -> ROT_RIGHT(-1) = ROT_LEFT(1) (Left)
        # So we also need to negate here to swap back to Right.
        return _rust_permute_inverse(self, -shift)
    HyperVector.permute_inverse = _aligned_permute_inverse
    
    if multiprocessing.current_process().name == "MainProcess":
        print(">> [VSA] Using Rust Accelerator (hypervec_rs) [Patched Direction]")
else:
    _install_compat_methods(_FallbackHV)
    HyperVector = _FallbackHV

__all__ = ["HyperVector"]
