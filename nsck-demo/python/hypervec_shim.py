"""Compatibility shim for the ``hypervec_rs`` Rust extension.

Tries to load the compiled Rust extension first.  When it is not available
(e.g. in CI or on a machine without the compiled binary) it falls back
transparently to the pure-Python implementation in ``hypervec_py``.

Modules throughout the codebase should import via::

    import hypervec_shim as hypervec_rs

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
    from hypervec_py import HyperVectorPy as _FallbackHV  # noqa: E402

    if multiprocessing.current_process().name == "MainProcess":
        print(">> [VSA] Using Python Fallback (via shim)")


def _hv_repr_bytes(hv) -> bytes:
    return repr(hv).encode("utf-8")


def _install_compat_methods(HV):
    """Add helper methods that may be missing from the Rust class."""

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


if _USE_RUST:
    _install_compat_methods(_ext.HyperVector)
    HyperVector = _ext.HyperVector
    if multiprocessing.current_process().name == "MainProcess":
        print(">> [VSA] Using Rust Accelerator (hypervec_rs)")
else:
    _install_compat_methods(_FallbackHV)
    HyperVector = _FallbackHV

__all__ = ["HyperVector"]
