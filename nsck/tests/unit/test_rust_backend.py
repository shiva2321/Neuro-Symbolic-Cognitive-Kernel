"""Tests for Rust backend verification (NSCK V11)."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))


def test_get_backend_info():
    import python.core.vsa.hypervec_shim as shim
    info = shim.get_backend_info()
    assert "vsa_backend" in info
    assert "snn_backend" in info
    assert "hypervec_rs_available" in info
    assert "snn_rs_available" in info
    assert "version" in info
    assert info["vsa_backend"] in ("Rust", "Python")
    assert info["snn_backend"] in ("Rust", "Python")


def test_hypervec_operations():
    import python.core.vsa.hypervec_shim as shim
    hv1 = shim.HyperVector(42)
    hv2 = shim.HyperVector(43)
    # XOR bind
    bound = hv1.bind(hv2) if hasattr(hv1, 'bind') else hv1.bundle(hv2)
    assert bound is not None
    # Bundle
    bundled = hv1.bundle(hv2)
    assert bundled is not None
    # Similarity
    sim = hv1.similarity(hv2)
    assert 0.0 <= sim <= 1.0


def test_similarity_self():
    import python.core.vsa.hypervec_shim as shim
    hv1 = shim.HyperVector(42)
    sim = hv1.similarity(hv1)
    assert sim == pytest.approx(1.0, abs=0.05)


def test_python_fallback_works():
    """Python fallback should always work regardless of Rust availability."""
    from python.core.vsa.hypervec_py import HyperVectorPy
    hv1 = HyperVectorPy(42)
    hv2 = HyperVectorPy(43)
    sim = hv1.similarity(hv2)
    assert 0.0 <= sim <= 1.0


def test_throughput_positive():
    """Ensure at least some throughput (sanity check)."""
    import time
    import python.core.vsa.hypervec_shim as shim
    n_ops = 1000
    hv1 = shim.HyperVector(1)
    hv2 = shim.HyperVector(2)
    start = time.time()
    for _ in range(n_ops):
        _ = hv1.similarity(hv2)
    elapsed = time.time() - start
    ops_per_sec = n_ops / elapsed if elapsed > 0 else float('inf')
    assert ops_per_sec > 100  # At least 100 ops/sec
