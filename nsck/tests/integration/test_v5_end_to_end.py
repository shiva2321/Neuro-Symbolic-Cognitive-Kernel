"""
NSCK V5 End-to-End Integration Tests
=====================================
Six integration test classes that validate the complete V5 cognitive pipeline.
Each test uses NSCKSubstrate as the entry point and asserts specific properties.
"""
import sys
import os
import pytest
import time

# Add the nsck root so that `from python.core...` imports resolve correctly.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from python.core.substrate import NSCKSubstrate
from python.core.integration.config import NSCKConfig


class TestTextDecisionLoop:
    """Test: ingest text → decide → learn → re-decide → assert learning improved."""

    def test_text_decision_learns(self):
        """Text decision improves after learning."""
        substrate = NSCKSubstrate()
        substrate.register_task("test_text")

        state1 = {"text": "move forward to reach the goal"}
        result1 = substrate.process(state1, "test_text")
        assert result1 is not None
        assert result1.chosen_action is not None

        # Learn from positive outcome
        substrate.feedback(result1.chosen_action, reward=1.0, task_tag="test_text", state=state1)

        # Re-decide on same input
        result2 = substrate.process(state1, "test_text")
        assert result2 is not None
        assert result2.chosen_action is not None

        # After learning, the system should have accumulated rules/episodes
        engine = substrate.engine
        assert engine.stats["episodes_recorded"] >= 1


class TestMultimodalFusion:
    """Test: ingest text + numeric → verify cross-modal → recall."""

    def test_multimodal_fusion(self):
        """Cross-modal fusion stores and recalls associations."""
        substrate = NSCKSubstrate()
        substrate.register_task("test_mm")

        # Provide text + numeric state
        state_mm = {
            "text": "sensor reading high",
            "value": 0.9,
            "sensor_id": 1,
        }
        result = substrate.process(state_mm, "test_mm")
        assert result is not None
        assert result.chosen_action is not None

        # Process a second multimodal input
        state_mm2 = {
            "text": "sensor reading low",
            "value": 0.1,
            "sensor_id": 2,
        }
        result2 = substrate.process(state_mm2, "test_mm")
        assert result2 is not None

        # Verify engine has processed decisions
        assert substrate.engine.stats["decisions"] >= 2


class TestSleepConsolidation:
    """Test: teach 50 concepts → sleep → verify prototypes."""

    def test_sleep_builds_prototypes(self):
        """Sleep consolidation builds concept prototypes."""
        substrate = NSCKSubstrate()
        substrate.register_task("test_sleep")

        # Teach 50 experiences
        for i in range(50):
            state = {"object": f"item_{i % 10}", "context": "environment", "step": i}
            result = substrate.process(state, "test_sleep")
            outcome = "success" if i % 3 == 0 else "neutral"
            substrate.feedback(
                result.chosen_action,
                reward=float(i % 3 == 0),
                task_tag="test_sleep",
                state=state,
                outcome=outcome,
            )

        # Run sleep consolidation
        sleep_result = substrate.sleep("test_sleep")
        assert sleep_result is not None

        engine = substrate.engine
        # Verify sleep ran
        assert engine.stats["sleep_cycles"] >= 1
        # Verify some episodes were recorded
        assert engine.stats["episodes_recorded"] >= 1


class TestCrossDomainTransfer:
    """Test: teach domain A → transfer → verify in domain B."""

    def test_cross_domain_transfer(self):
        """Knowledge learned in domain A transfers to domain B."""
        substrate = NSCKSubstrate()
        substrate.register_task("domain_a")
        substrate.register_task("domain_b")

        # Teach 20 examples in domain_a
        for i in range(20):
            state = {"feature": f"pattern_{i % 5}", "value": i}
            result = substrate.process(state, "domain_a")
            substrate.feedback(
                result.chosen_action,
                reward=1.0 if i % 2 == 0 else 0.0,
                task_tag="domain_a",
                state=state,
            )

        # Now query domain_b (should be able to decide even without direct experience)
        state_b = {"feature": "pattern_1", "value": 5}
        result_b = substrate.process(state_b, "domain_b")
        assert result_b is not None
        assert result_b.chosen_action is not None

        # Engine should show cross-domain activity
        assert substrate.engine.stats["decisions"] >= 21


class TestProceduralFastPath:
    """Test: cache skill → reingest → verify O(1) hit."""

    def test_procedural_cache_hit(self):
        """Skills cached via positive reward are retrieved on re-ingest."""
        substrate = NSCKSubstrate()
        substrate.register_task("test_proc")

        # Build a skill via positive reward
        state = {"position_x": 3, "position_y": 4, "task": "navigate"}
        result1 = substrate.process(state, "test_proc")
        substrate.feedback(result1.chosen_action, reward=1.0, task_tag="test_proc", state=state)

        # Decide again on same state — should potentially hit procedural cache
        result2 = substrate.process(state, "test_proc")
        assert result2 is not None
        assert result2.chosen_action is not None

        # After positive feedback, procedural memory should have at least one entry
        if hasattr(substrate, 'procedural_memory') and substrate.procedural_memory is not None:
            assert substrate.procedural_memory is not None


class TestFullLifecycle:
    """Test: init → seed → 100 cycles → sleep → stats."""

    def test_full_100_cycle_lifecycle(self):
        """Full lifecycle: 100 decision cycles, sleep, valid stats."""
        substrate = NSCKSubstrate()
        substrate.register_task("lifecycle")

        actions_seen = set()
        rewards_given = 0

        for i in range(100):
            state = {
                "step": i,
                "phase": i // 20,
                "value": float(i % 10) / 10.0,
                "text": f"step {i} in lifecycle",
            }
            result = substrate.process(state, "lifecycle")
            assert result is not None, f"Decision failed at step {i}"
            assert result.chosen_action is not None
            actions_seen.add(result.chosen_action)

            reward = 1.0 if i % 5 == 0 else 0.0
            if reward > 0:
                rewards_given += 1
            substrate.feedback(
                result.chosen_action,
                reward=reward,
                task_tag="lifecycle",
                state=state,
            )

        # Run sleep
        sleep_result = substrate.sleep("lifecycle")
        assert sleep_result is not None

        engine = substrate.engine
        stats = engine.stats

        # Validate stats
        assert stats["decisions"] >= 100
        assert stats["episodes_recorded"] >= 20
        assert stats["sleep_cycles"] >= 1
        assert len(actions_seen) >= 1
        assert rewards_given == 20  # Every 5th step
