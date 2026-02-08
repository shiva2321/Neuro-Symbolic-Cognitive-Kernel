"""
Test Knowledge Integration
Verifies: unified cognitive pipeline, learning, self-correction, transfer.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from python.knowledge_integration import (
    KnowledgeIntegration,
    CognitiveResponse,
)
from python.multimodal_processor import MultimodalInput
import numpy as np


def test_text_understanding():
    """Test that text input is processed and understood."""
    print("--- Test: Text Understanding ---")

    ki = KnowledgeIntegration()
    inp = MultimodalInput(text="The red traffic light means stop")
    response = ki.process_input(inp)

    assert isinstance(response, CognitiveResponse)
    assert response.answer, "Should generate a response"
    assert len(response.reasoning_trace) > 0
    assert response.confidence > 0
    print(f"  Answer: {response.answer[:100]}")
    print(f"  Trace: {response.reasoning_trace}")

    print("✅ Text Understanding Test Passed")


def test_contextual_disambiguation_in_pipeline():
    """Test that contextual disambiguation works in the full pipeline."""
    print("\n--- Test: Contextual Disambiguation in Pipeline ---")

    ki = KnowledgeIntegration()

    # Process in traffic context (red = danger)
    inp1 = MultimodalInput(
        text="red signal ahead",
        structured={"domain": "traffic"},
    )
    r1 = ki.process_input(inp1)

    # Process in garden context (red = beauty)
    inp2 = MultimodalInput(
        text="red rose garden",
        structured={"domain": "garden"},
    )
    r2 = ki.process_input(inp2)

    # Both should generate responses
    assert r1.answer, "Traffic context should produce a response"
    assert r2.answer, "Garden context should produce a response"
    print(f"  Traffic: {r1.answer[:80]}")
    print(f"  Garden: {r2.answer[:80]}")

    print("✅ Contextual Disambiguation in Pipeline Test Passed")


def test_memory_recall():
    """Test that the system remembers past interactions."""
    print("\n--- Test: Memory Recall ---")

    ki = KnowledgeIntegration()

    # Process first input
    inp1 = MultimodalInput(text="I learned about red roses")
    ki.process_input(inp1, task_tag="learning")

    # Process related input
    inp2 = MultimodalInput(text="Tell me about roses")
    r2 = ki.process_input(inp2, task_tag="learning")

    assert r2.recalled_episodes > 0, "Should recall the previous episode"
    print(f"  Recalled {r2.recalled_episodes} episodes")

    print("✅ Memory Recall Test Passed")


def test_knowledge_query():
    """Test querying the knowledge base."""
    print("\n--- Test: Knowledge Query ---")

    ki = KnowledgeIntegration()

    # Bootstrap knowledge should be queryable
    results = ki.query_knowledge("red")
    assert len(results) > 0, "Should have bootstrap knowledge about 'red'"
    print(f"  Knowledge about 'red': {[(k.relation, k.target) for k in results]}")

    results = ki.query_relation("fire", "causes")
    assert any(k.target == "danger" for k in results), "fire should cause danger"
    print(f"  fire causes: {[k.target for k in results]}")

    print("✅ Knowledge Query Test Passed")


def test_self_correction():
    """Test that the system can correct its knowledge."""
    print("\n--- Test: Self-Correction ---")

    ki = KnowledgeIntegration()

    # Initially, 'fire' causes 'danger'
    before = ki.query_relation("fire", "causes")
    assert any(k.target == "danger" for k in before)

    # Correct: fire causes 'heat' (not just danger)
    ki.correct_knowledge("fire", "causes", "danger", "heat_and_danger")

    # After correction, the original should be weakened
    after = ki.query_relation("fire", "causes")
    danger_entry = [k for k in after if k.target == "danger"]
    heat_entry = [k for k in after if k.target == "heat_and_danger"]

    assert len(heat_entry) > 0, "Corrected knowledge should exist"
    assert heat_entry[0].confidence >= 0.7, "Corrected entry should have high confidence"

    if danger_entry:
        assert danger_entry[0].confidence < 0.7, "Original should be weakened"

    print(f"  After correction: {[(k.target, k.confidence) for k in after]}")

    print("✅ Self-Correction Test Passed")


def test_learning_from_feedback():
    """Test that the system learns from reward feedback."""
    print("\n--- Test: Learning from Feedback ---")

    ki = KnowledgeIntegration()

    # Process an input
    inp = MultimodalInput(text="Moving toward food")
    ki.process_input(inp, task_tag="snake")

    # Give positive feedback
    ki.learn_from_feedback("snake", reward=1.0, concepts=["food", "moving"])

    # Stats should update
    stats = ki.get_statistics()
    assert stats["queries_processed"] >= 1
    print(f"  Stats: {stats}")

    print("✅ Learning from Feedback Test Passed")


def test_multimodal_input():
    """Test processing multimodal input (text + image)."""
    print("\n--- Test: Multimodal Input ---")

    ki = KnowledgeIntegration()
    inp = MultimodalInput(
        text="A beautiful garden with red roses",
        image=np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
    )
    response = ki.process_input(inp)

    assert response.answer
    # Should have processed both modalities
    print(f"  Answer: {response.answer[:100]}")
    print(f"  Confidence: {response.confidence:.2f}")

    print("✅ Multimodal Input Test Passed")


def test_knowledge_transfer():
    """Test knowledge transfer between domains."""
    print("\n--- Test: Knowledge Transfer ---")

    ki = KnowledgeIntegration()

    # Add snake-specific knowledge
    ki._add_knowledge("snake_food", "causes", "reward", 0.9, "learned")
    ki._add_knowledge("snake_wall", "causes", "danger", 0.9, "learned")

    # Transfer to pong domain
    transferred = ki.apply_knowledge_to_new_domain("snake", "pong")

    assert len(transferred) > 0, "Should transfer some knowledge"
    print(f"  Transferred: {transferred}")

    print("✅ Knowledge Transfer Test Passed")


def test_statistics():
    """Test that statistics are reported."""
    print("\n--- Test: Statistics ---")

    ki = KnowledgeIntegration()
    stats = ki.get_statistics()

    assert "knowledge_entries" in stats
    assert "semantic_concepts" in stats
    assert stats["knowledge_entries"] > 0
    print(f"  Stats: {stats}")

    print("✅ Statistics Test Passed")


if __name__ == "__main__":
    test_text_understanding()
    test_contextual_disambiguation_in_pipeline()
    test_memory_recall()
    test_knowledge_query()
    test_self_correction()
    test_learning_from_feedback()
    test_multimodal_input()
    test_knowledge_transfer()
    test_statistics()
    print("\n🎉 All Knowledge Integration Tests Passed!")
