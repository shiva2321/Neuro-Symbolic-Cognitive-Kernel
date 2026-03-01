"""Unit tests for NSCKVisionFusion and AccuracyEstimator."""
import os
import sys
import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_NSCK_VISION_DIR = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_NSCK_VISION_DIR)
_NSCK_DIR = os.path.join(_REPO_ROOT, "nsck")
for p in [_NSCK_DIR, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _make_fusion():
    """Create a minimal NSCKVisionFusion for testing."""
    from python.core.vision.absorption_memory import AbsorptionMemory
    from python.core.vision.domain_tagger import DomainTagger
    from python.core.vision.vision_fusion import NSCKVisionFusion
    mem = AbsorptionMemory()
    tagger = DomainTagger()
    return NSCKVisionFusion(absorption_memory=mem, domain_tagger=tagger)


def test_fusion_nsck_wins():
    """When NSCK conf > ref conf * 0.95 AND min_causal_depth=0, NSCK wins."""
    import python.core.vsa.hypervec_shim as hypervec_rs
    fusion = _make_fusion()
    query_hv = hypervec_rs.HyperVector(42)

    response = fusion.fuse(
        query_hv=query_hv,
        label="nsck_label",
        nsck_confidence=0.9,
        reference_result={"label": "ref_label", "confidence": 0.7},
        override_threshold=0.95,
        min_causal_depth=0,
    )
    assert response.nsck_overrode_reference is True
    assert response.label == "nsck_label"


def test_fusion_reference_wins():
    """When ref conf >> NSCK conf, reference wins but NSCK augments."""
    import python.core.vsa.hypervec_shim as hypervec_rs
    fusion = _make_fusion()
    query_hv = hypervec_rs.HyperVector(42)

    response = fusion.fuse(
        query_hv=query_hv,
        label="nsck_label",
        nsck_confidence=0.2,
        reference_result={"label": "ref_label", "confidence": 0.9},
        override_threshold=0.95,
        min_causal_depth=1,
    )
    assert response.nsck_overrode_reference is False
    assert response.reference_label == "ref_label"


def test_fusion_agreement():
    """When both agree on same label, accuracy_rating should be reasonable."""
    import python.core.vsa.hypervec_shim as hypervec_rs
    fusion = _make_fusion()
    query_hv = hypervec_rs.HyperVector(42)

    response = fusion.fuse(
        query_hv=query_hv,
        label="cat",
        nsck_confidence=0.8,
        reference_result={"label": "cat", "confidence": 0.85},
        override_threshold=0.95,
        min_causal_depth=0,
    )
    assert response.accuracy_rating > 0.3


def test_accuracy_estimator():
    """AccuracyEstimator returns values in [0, 1]."""
    from python.core.vision.vision_fusion import AccuracyEstimator
    estimator = AccuracyEstimator()

    acc = estimator.estimate(
        nsck_confidence=0.9,
        reference_confidence=0.85,
        agreement=True,
        domain_coverage=0.8,
        episodic_hits=3,
        causal_chain_depth=2,
        analogy_count=1,
    )
    assert 0.0 <= acc <= 1.0

    acc_low = estimator.estimate(
        nsck_confidence=0.1,
        reference_confidence=0.9,
        agreement=False,
        domain_coverage=0.1,
        episodic_hits=0,
        causal_chain_depth=0,
        analogy_count=0,
    )
    assert 0.0 <= acc_low <= 1.0
    assert acc > acc_low


def test_vision_response_schema():
    """VisionResponse has all required fields with correct types."""
    import python.core.vsa.hypervec_shim as hypervec_rs
    fusion = _make_fusion()
    query_hv = hypervec_rs.HyperVector(99)
    response = fusion.fuse(query_hv=query_hv, label="dog", nsck_confidence=0.7)

    assert hasattr(response, "query_id")
    assert hasattr(response, "label")
    assert hasattr(response, "confidence")
    assert hasattr(response, "accuracy_rating")
    assert hasattr(response, "top5")
    assert hasattr(response, "causal_chain")
    assert hasattr(response, "cross_domain_analogies")
    assert hasattr(response, "source_model_provenance")
    assert hasattr(response, "reference_label")
    assert hasattr(response, "reference_confidence")
    assert hasattr(response, "nsck_overrode_reference")
    assert hasattr(response, "latency_ms")
    assert hasattr(response, "timestamp")
    assert hasattr(response, "accuracy_breakdown")

    assert isinstance(response.query_id, str)
    assert isinstance(response.label, str)
    assert 0.0 <= response.confidence <= 1.0
    assert 0.0 <= response.accuracy_rating <= 1.0
    assert isinstance(response.causal_chain, list)
    assert isinstance(response.cross_domain_analogies, list)
    assert isinstance(response.source_model_provenance, list)
    assert isinstance(response.nsck_overrode_reference, bool)
