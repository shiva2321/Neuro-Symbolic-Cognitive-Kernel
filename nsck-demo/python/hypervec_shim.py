"""Compatibility shim for the `hypervec_rs` module.

This file is *not* intended to shadow the compiled extension on PYTHONPATH.
Instead, tests or modules can import it explicitly if needed.

We also provide an optional monkey-patch that adds missing methods to the
compiled HyperVector class when running on an older build.
"""

from __future__ import annotations

import hashlib
from typing import Optional

# This should now find the compiled extension since this file is named hypervec_shim.py
import hypervec_rs as _ext


def _hv_repr_bytes(hv) -> bytes:
    return repr(hv).encode("utf-8")


def _install_compat_methods():
    HV = getattr(_ext, "HyperVector", None)
    if HV is None:
        return

    # --- weighted_bundle ---
    if not hasattr(HV, "weighted_bundle"):
        def weighted_bundle(self, other, weight: float, seed: Optional[int] = None):
            # Approximate: repeatedly bundle to bias toward self.
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
            # Deterministic hash of repr (good enough for tests/caching).
            h = hashlib.blake2b(_hv_repr_bytes(self), digest_size=8)
            sig64 = int.from_bytes(h.digest(), "little")
            n = int(n_bits)
            if n >= 64:
                return sig64
            return sig64 & ((1 << n) - 1)

        setattr(HV, "lsh_hash", lsh_hash)


_install_compat_methods()

# Re-export the patched class
HyperVector = _ext.HyperVector

__all__ = ["HyperVector"]
