"""Tests for PerceptionDistiller (WP-4)."""
from __future__ import annotations
import pytest
import numpy as np


def _make_hv(seed: int):
    import python.core.vsa.hypervec_shim as hv_mod
    return hv_mod.HyperVector(seed)


def test_quality_tracking():
    """Distiller tracks observations per modality."""
    from python.core.learning.perception_distiller import PerceptionDistiller
    d = PerceptionDistiller(threshold=0.80)
    hv1 = _make_hv(1)
    hv2 = _make_hv(1)  # Same seed = identical
    
    for _ in range(5):
        d.observe("text", hv1, hv2)
    
    report = d.get_report()
    assert "text" in report
    assert report["text"]["observations"] == 5


def test_graduation_at_threshold():
    """Graduates when rolling avg >= threshold."""
    from python.core.learning.perception_distiller import PerceptionDistiller
    d = PerceptionDistiller(threshold=0.0)  # threshold=0 → always graduate after 10 obs
    hv = _make_hv(42)
    
    for _ in range(10):
        d.observe("text", hv, hv)
    
    assert d.is_graduated("text") is True


def test_no_premature_graduation():
    """Does not graduate before 10 observations."""
    from python.core.learning.perception_distiller import PerceptionDistiller
    d = PerceptionDistiller(threshold=0.0)
    hv = _make_hv(42)
    
    for _ in range(9):
        d.observe("text", hv, hv)
    
    assert d.is_graduated("text") is False


def test_hybrid_routing_before_graduation():
    """is_graduated returns False before enough observations."""
    from python.core.learning.perception_distiller import PerceptionDistiller
    d = PerceptionDistiller(threshold=0.90)
    assert d.is_graduated("image") is False


def test_distiller_report_format():
    """Report contains expected keys."""
    from python.core.learning.perception_distiller import PerceptionDistiller
    d = PerceptionDistiller()
    hv = _make_hv(1)
    d.observe("audio", hv, hv)
    
    report = d.get_report()
    assert "audio" in report
    r = report["audio"]
    assert "observations" in r
    assert "current_quality" in r
    assert "avg" in r
    assert "graduated" in r
