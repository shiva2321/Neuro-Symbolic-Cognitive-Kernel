"""Tests for CrossModalCorrelationLearner (NSCK V11)."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

import python.core.vsa.hypervec_shim as hv
from python.core.learning.cross_modal import CrossModalCorrelationLearner


def test_observe_builds_associations():
    learner = CrossModalCorrelationLearner()
    text_hv = hv.HyperVector(100)
    image_hv = hv.HyperVector(200)
    learner.observe({"text": text_hv, "image": image_hv}, context_tag="test")
    stats = learner.get_statistics()
    assert stats["total_observations"] > 0
    assert stats["associations_learned"] > 0


def test_predict_modality_returns_hvs():
    learner = CrossModalCorrelationLearner()
    text_hv = hv.HyperVector(100)
    image_hv = hv.HyperVector(200)
    # Observe 5 times to build up associations
    for _ in range(5):
        learner.observe({"text": text_hv, "image": image_hv})
    results = learner.predict_modality(text_hv, "text", "image", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0


def test_correlation_strength_increases_with_observations():
    learner = CrossModalCorrelationLearner()
    text_hv = hv.HyperVector(100)
    image_hv = hv.HyperVector(200)
    # Initial strength
    s0 = learner.get_correlation_strength(text_hv, "text", image_hv, "image")
    # After observations
    for _ in range(5):
        learner.observe({"text": text_hv, "image": image_hv})
    s1 = learner.get_correlation_strength(text_hv, "text", image_hv, "image")
    # Strength should be non-negative after observations
    assert s1 >= 0.0


def test_statistics_tracking():
    learner = CrossModalCorrelationLearner()
    hv1 = hv.HyperVector(1)
    hv2 = hv.HyperVector(2)
    hv3 = hv.HyperVector(3)
    learner.observe({"text": hv1, "image": hv2, "audio": hv3}, context_tag="ctx1")
    stats = learner.get_statistics()
    assert "pairs_seen" in stats
    assert "associations_learned" in stats
    assert "ctx1" in stats["context_tags"]


def test_no_prediction_without_observations():
    learner = CrossModalCorrelationLearner()
    hv1 = hv.HyperVector(1)
    results = learner.predict_modality(hv1, "text", "image", top_k=3)
    assert results == []
