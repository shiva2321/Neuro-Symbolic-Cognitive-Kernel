"""Compatibility shim for the ``snn_rs`` Rust extension.

Follows the identical pattern as ``python/core/vsa/hypervec_shim.py``:
  1. Try to load the compiled Rust extension (snn_rs.pyd / snn_rs.so).
  2. If missing or broken, fall back to pure-Python implementations
     already present in snn_perception.py, vsa_snn_bridge.py, and hebbian.py.

Usage (anywhere in the codebase):
    from python.core.perception.snn_shim import (
        LIFLayer, SnnCore, StdpEngine,
        HebbianMatrix, ConceptMapper, RateCoder,
        USE_RUST as SNN_USE_RUST,
    )

Build the Rust extension (one-time, requires Rust + maturin):
    cd nsck/rust_snn
    maturin develop --release
"""

from __future__ import annotations

# ── Try Rust backend ────────────────────────────────────────────────────────
USE_RUST: bool = False
_snn_rs = None

try:
    import snn_rs as _snn_rs  # compiled Rust extension
    USE_RUST = True
except ImportError:
    pass

# ── Announce backend ────────────────────────────────────────────────────────
if USE_RUST:
    print(">> [SNN]  Using Rust backend (snn_rs)")
else:
    print(">> [SNN]  Rust backend missing — using Python fallback")

# ── Export classes ──────────────────────────────────────────────────────────

if USE_RUST:
    # ── Rust classes ────────────────────────────────────────────────────────
    LIFLayer      = _snn_rs.LIFLayer        # rayon-parallel LIF step
    SnnCore       = _snn_rs.SnnCore         # full perceive() inner loop
    StdpEngine    = _snn_rs.StdpEngine      # parallel STDP weight update
    HebbianMatrix = _snn_rs.HebbianMatrix   # Oja's rule, parallel
    ConceptMapper = _snn_rs.ConceptMapper   # Jaccard concept recognition
    RateCoder     = _snn_rs.RateCoder       # spike train → rates + active neurons

else:
    # ── Python fallback classes ─────────────────────────────────────────────
    import sys
    import os
    _here = os.path.dirname(__file__)
    _root = os.path.abspath(os.path.join(_here, "../../../.."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from python.core.perception.snn_perception import LIFNeuronLayer as LIFLayer  # type: ignore
    from python.core.learning.hebbian import HebbianMatrixNumPy as HebbianMatrix  # type: ignore
    from python.core.perception.snn_perception import SimpleConceptMapper as ConceptMapper  # type: ignore
    from python.core.perception.vsa_snn_bridge import RateCoder  # type: ignore
    # Full-loop fallbacks: share the pure-NumPy implementations defined in snn_perception
    from python.core.perception.snn_perception import PythonSnnCore as SnnCore  # type: ignore
    from python.core.perception.snn_perception import PythonStdpEngine as StdpEngine  # type: ignore


__all__ = [
    "USE_RUST",
    "LIFLayer",
    "SnnCore",
    "StdpEngine",
    "HebbianMatrix",
    "ConceptMapper",
    "RateCoder",
]
