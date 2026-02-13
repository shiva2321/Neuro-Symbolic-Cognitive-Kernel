"""
Test Context Engine
Verifies: contextual disambiguation, learning, domain inference.
"""
import sys
import os

from python.context_engine import ContextEngine, ContextFrame, DisambiguatedMeaning
from python.semantic_memory import SemanticMemory


def test_red_danger_vs_red_roses():
    """Core test: 'red' means danger in traffic but beauty in garden."""
    print("--- Test: Red = Danger vs Red Roses = Beauty ---")

    engine = ContextEngine()

    # Traffic context
    traffic_ctx = ContextFrame(
        domain="traffic",
        active_concepts=["car", "signal", "road"],
        environment_cues={"location": "intersection"},
    )
    result = engine.disambiguate("red", traffic_ctx)
    assert result.meaning == "danger", f"Expected 'danger', got '{result.meaning}'"
    assert result.confidence >= 0.8, f"Confidence too low: {result.confidence}"
    print(f"  Traffic: red → {result.meaning} (conf={result.confidence:.2f})")

    # Garden context
    garden_ctx = ContextFrame(
        domain="garden",
        active_concepts=["rose", "flower", "petal"],
        environment_cues={"location": "park"},
    )
    result = engine.disambiguate("red", garden_ctx)
    assert result.meaning == "beauty", f"Expected 'beauty', got '{result.meaning}'"
    print(f"  Garden: red → {result.meaning} (conf={result.confidence:.2f})")

    print("✅ Red Danger vs Red Roses Test Passed")


def test_disambiguation_with_cues():
    """Test disambiguation using environmental cues when domain doesn't match directly."""
    print("\n--- Test: Disambiguation via Cues ---")

    engine = ContextEngine()

    # Context where domain is 'outdoor' (not registered for 'hot')
    # but cues contain 'temperature' keywords
    ctx = ContextFrame(
        domain="outdoor",
        active_concepts=["sun", "temperature", "weather"],
        environment_cues={"sensor": "thermometer"},
    )
    result = engine.disambiguate("hot", ctx)
    assert result.meaning == "high_temperature", f"Expected 'high_temperature', got '{result.meaning}'"
    print(f"  Outdoor+temperature cues: hot → {result.meaning} (conf={result.confidence:.2f})")

    print("✅ Disambiguation via Cues Test Passed")


def test_default_fallback():
    """Test that unknown context falls back to default meaning."""
    print("\n--- Test: Default Fallback ---")

    engine = ContextEngine()

    ctx = ContextFrame(
        domain="unknown_domain",
        active_concepts=[],
        environment_cues={},
    )
    result = engine.disambiguate("red", ctx)
    assert result.meaning == "danger", f"Expected default 'danger', got '{result.meaning}'"
    assert result.context_domain == "default"
    print(f"  Unknown domain: red → {result.meaning} (fallback)")

    print("✅ Default Fallback Test Passed")


def test_unknown_concept():
    """Test behaviour with a concept that has no registered meanings."""
    print("\n--- Test: Unknown Concept ---")

    engine = ContextEngine()

    ctx = ContextFrame(
        domain="general",
        active_concepts=["foo"],
        environment_cues={},
    )
    result = engine.disambiguate("xyzzy", ctx)
    assert result.meaning == "xyzzy", "Unknown concept should map to itself"
    assert result.confidence < 0.5
    print(f"  Unknown: xyzzy → {result.meaning} (conf={result.confidence:.2f})")

    print("✅ Unknown Concept Test Passed")


def test_learning_from_feedback():
    """Test that the engine learns new context-meaning associations."""
    print("\n--- Test: Learning from Feedback ---")

    engine = ContextEngine()

    # Initially, 'red' in 'celebration' domain is not registered
    ctx = ContextFrame(
        domain="celebration",
        active_concepts=["party", "fireworks"],
        environment_cues={},
    )
    result_before = engine.disambiguate("red", ctx)
    # Should fall back to default
    assert result_before.context_domain != "celebration"

    # Learn that red means 'festive' in celebration context
    engine.learn_from_feedback("red", "festive", ctx)

    # Now re-disambiguate
    result_after = engine.disambiguate("red", ctx)
    assert result_after.meaning == "festive", f"Expected 'festive', got '{result_after.meaning}'"
    assert result_after.confidence >= 0.8
    print(f"  After learning: red in celebration → {result_after.meaning}")

    print("✅ Learning from Feedback Test Passed")


def test_multiple_alternatives():
    """Test that alternative meanings are provided."""
    print("\n--- Test: Alternative Meanings ---")

    engine = ContextEngine()

    ctx = ContextFrame(
        domain="traffic",
        active_concepts=["signal"],
        environment_cues={},
    )
    result = engine.disambiguate("red", ctx)
    assert len(result.alternative_meanings) > 0, "Should have alternative meanings"
    print(f"  Primary: {result.meaning}, Alternatives: {result.alternative_meanings}")

    print("✅ Alternative Meanings Test Passed")


def test_domain_inference():
    """Test automatic domain inference from cues."""
    print("\n--- Test: Domain Inference ---")

    engine = ContextEngine()

    domain = engine.infer_context_domain(
        cues=["car", "traffic", "signal"],
        environment={"location": "road"},
    )
    assert domain == "traffic", f"Expected 'traffic', got '{domain}'"
    print(f"  Inferred domain from traffic cues: {domain}")

    print("✅ Domain Inference Test Passed")


def test_batch_disambiguation():
    """Test disambiguating multiple concepts at once."""
    print("\n--- Test: Batch Disambiguation ---")

    engine = ContextEngine()

    ctx = ContextFrame(
        domain="garden",
        active_concepts=["flower", "plant"],
        environment_cues={},
    )
    results = engine.disambiguate_all(["red", "green"], ctx)
    assert len(results) == 2
    assert results[0].meaning == "beauty"
    # 'green' has no 'garden' domain — falls back to default 'safe'
    assert results[1].meaning in ("nature", "safe", "healthy")
    print(f"  Garden: red → {results[0].meaning}, green → {results[1].meaning}")

    print("✅ Batch Disambiguation Test Passed")


def test_statistics():
    """Test that statistics are reported correctly."""
    print("\n--- Test: Statistics ---")

    engine = ContextEngine()
    stats = engine.get_statistics()
    assert stats["registered_concepts"] > 0
    assert stats["total_meanings"] > 0
    print(f"  Stats: {stats}")

    print("✅ Statistics Test Passed")


if __name__ == "__main__":
    test_red_danger_vs_red_roses()
    test_disambiguation_with_cues()
    test_default_fallback()
    test_unknown_concept()
    test_learning_from_feedback()
    test_multiple_alternatives()
    test_domain_inference()
    test_batch_disambiguation()
    test_statistics()
    print("\n🎉 All Context Engine Tests Passed!")
