"""Unit tests for cross-domain capabilities."""
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


def test_domain_tagger():
    """Register 3 domains, verify all accessible."""
    from python.core.vision.domain_tagger import DomainTagger
    tagger = DomainTagger()

    tagger.register_model("model_a", "medical", {"type": "ResNet"})
    tagger.register_model("model_b", "satellite", {"type": "ViT"})
    tagger.register_model("model_c", "traffic", {"type": "EfficientNet"})

    domains = tagger.all_domains()
    assert "medical" in domains
    assert "satellite" in domains
    assert "traffic" in domains

    medical_models = tagger.get_domain_models("medical")
    assert "model_a" in medical_models


def test_cross_domain_similarity():
    """Cross-domain similarity is computed correctly."""
    from python.core.vision.domain_tagger import DomainTagger
    tagger = DomainTagger()

    tagger.register_model("model_a", "domain_A", {})
    tagger.register_model("model_b", "domain_B", {})

    sim_same = tagger.get_cross_domain_similarity("domain_A", "domain_A")
    assert sim_same == 1.0

    sim_diff = tagger.get_cross_domain_similarity("domain_A", "domain_B")
    assert 0.0 <= sim_diff <= 1.0


def test_absorption_memory_domain_query():
    """Records stored under a domain are returned by query_by_domain."""
    from python.core.vision.absorption_memory import AbsorptionMemory
    import python.core.vsa.hypervec_shim as hypervec_rs

    mem = AbsorptionMemory()
    hv_a = hypervec_rs.HyperVector(1)
    hv_b = hypervec_rs.HyperVector(2)

    mem.store(hv=hv_a, label="cat", domain="animals", model_id="model1", confidence=0.9, metadata={})
    mem.store(hv=hv_b, label="car", domain="vehicles", model_id="model2", confidence=0.8, metadata={})

    animals = mem.query_by_domain("animals")
    assert len(animals) == 1
    assert animals[0].label == "cat"
    assert animals[0].domain == "animals"

    vehicles = mem.query_by_domain("vehicles")
    assert len(vehicles) == 1
    assert vehicles[0].label == "car"
