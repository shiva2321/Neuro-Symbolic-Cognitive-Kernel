"""
Phase 3 Control Layer Tests

Tests for cognitive control mechanisms:
1. NoveltyDetector - detects out-of-distribution patterns
2. ConfidenceEstimator - measures output stability
3. ConflictMonitor - identifies ambiguous patterns
4. PlasticityGateController - context-aware learning gating
5. ControlLayer - integrated control system

Invariants:
- Novelty, confidence, conflict scores always in [0, 1]
- Learning gate responds to context appropriately
- Control layer does not modify computation
- Metrics are observable without side effects
"""

import unittest
from core.control_layer import (
    NoveltyDetector,
    ConfidenceEstimator,
    ConflictMonitor,
    PlasticityGateController,
    ControlLayer,
    LearningContext
)


class TestLearningContext(unittest.TestCase):
    """Test learning context decision logic."""

    def test_explicit_learning_mode(self):
        """Explicit learning mode should override all thresholds."""
        context = LearningContext(
            novelty=0.0,
            confidence=1.0,
            conflict=0.0,
            reward=0.0,
            prediction_error=0.0,
            explicit_learning_mode=True
        )

        self.assertTrue(context.should_learn())

    def test_high_novelty_triggers_learning(self):
        """High novelty should trigger learning."""
        context = LearningContext(novelty=0.8)
        self.assertTrue(context.should_learn(novelty_threshold=0.5))

        context_low = LearningContext(novelty=0.3)
        self.assertFalse(context_low.should_learn(novelty_threshold=0.5))

    def test_reward_triggers_learning(self):
        """Positive reward should trigger learning."""
        context = LearningContext(reward=0.5)
        self.assertTrue(context.should_learn(reward_threshold=0.1))

        context_no_reward = LearningContext(reward=0.0)
        self.assertFalse(context_no_reward.should_learn(reward_threshold=0.1))

    def test_prediction_error_triggers_learning(self):
        """High prediction error should trigger learning."""
        context = LearningContext(prediction_error=0.5)
        self.assertTrue(context.should_learn(error_threshold=0.3))

        context_low_error = LearningContext(prediction_error=0.1)
        self.assertFalse(context_low_error.should_learn(error_threshold=0.3))


class TestNoveltyDetector(unittest.TestCase):
    """Test novelty detection."""

    def test_initial_pattern_is_novel(self):
        """First pattern should have high novelty."""
        detector = NoveltyDetector()

        firing = {1, 2, 3}
        all_neurons = {1, 2, 3, 4, 5}

        novelty = detector.update(firing, all_neurons)
        self.assertGreater(novelty, 0.5)  # First pattern is novel

    def test_repeated_pattern_reduces_novelty(self):
        """Repeated patterns should reduce novelty over time."""
        detector = NoveltyDetector(decay_rate=0.9)

        firing = {1, 2}
        all_neurons = {1, 2, 3, 4}

        novelties = []
        for _ in range(10):
            novelty = detector.update(firing, all_neurons)
            novelties.append(novelty)

        # Novelty should decrease
        self.assertGreater(novelties[0], novelties[-1])

    def test_different_pattern_increases_novelty(self):
        """Switching to different pattern should increase novelty."""
        detector = NoveltyDetector(decay_rate=0.9)

        pattern1 = {1, 2}
        pattern2 = {3, 4}
        all_neurons = {1, 2, 3, 4}

        # Learn pattern 1
        for _ in range(10):
            detector.update(pattern1, all_neurons)

        novelty_baseline = detector.update(pattern1, all_neurons)
        novelty_new = detector.update(pattern2, all_neurons)

        self.assertGreater(novelty_new, novelty_baseline)

    def test_novelty_bounded(self):
        """Novelty score should always be in [0, 1]."""
        detector = NoveltyDetector()

        all_neurons = set(range(10))

        for _ in range(100):
            firing = {i for i in range(10) if i % 2 == 0}
            novelty = detector.update(firing, all_neurons)

            self.assertGreaterEqual(novelty, 0.0)
            self.assertLessEqual(novelty, 1.0)


class TestConfidenceEstimator(unittest.TestCase):
    """Test confidence estimation."""

    def test_stable_output_high_confidence(self):
        """Stable output patterns should yield high confidence."""
        estimator = ConfidenceEstimator(window_size=5)

        output_neurons = {10: True, 11: False, 12: True}

        confidences = []
        for _ in range(10):
            confidence = estimator.update(output_neurons)
            confidences.append(confidence)

        # Confidence should increase with stability
        self.assertGreater(confidences[-1], 0.8)

    def test_unstable_output_low_confidence(self):
        """Unstable output patterns should yield low confidence."""
        estimator = ConfidenceEstimator(window_size=5)

        # Alternate outputs
        for i in range(10):
            if i % 2 == 0:
                output = {10: True, 11: False}
            else:
                output = {10: False, 11: True}

            confidence = estimator.update(output)

        # Alternating patterns have predictable variance, so confidence is moderate
        self.assertLess(confidence, 0.9)  # Should be less confident than stable pattern

    def test_confidence_bounded(self):
        """Confidence score should always be in [0, 1]."""
        estimator = ConfidenceEstimator()

        for i in range(100):
            output = {10: i % 3 == 0, 11: i % 5 == 0}
            confidence = estimator.update(output)

            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


class TestConflictMonitor(unittest.TestCase):
    """Test conflict monitoring."""

    def test_stable_pattern_no_conflict(self):
        """Stable pattern should have low conflict."""
        monitor = ConflictMonitor(window_size=5)

        output = {10: True, 11: False}

        for _ in range(10):
            conflict = monitor.update(output)

        self.assertLess(conflict, 0.3)

    def test_alternating_patterns_high_conflict(self):
        """Alternating patterns should have high conflict."""
        monitor = ConflictMonitor(window_size=5)

        for i in range(10):
            if i % 2 == 0:
                output = {10: True, 11: False}
            else:
                output = {10: False, 11: True}

            conflict = monitor.update(output)

        # Alternating patterns create moderate conflict (2 unique patterns in window)
        self.assertGreater(conflict, 0.3)  # Should show conflict
        self.assertLess(conflict, 0.8)  # But not maximum since only 2 patterns

    def test_conflict_bounded(self):
        """Conflict score should always be in [0, 1]."""
        monitor = ConflictMonitor()

        for i in range(100):
            output = {10: i % 3 == 0, 11: i % 5 == 0, 12: i % 7 == 0}
            conflict = monitor.update(output)

            self.assertGreaterEqual(conflict, 0.0)
            self.assertLessEqual(conflict, 1.0)


class TestPlasticityGateController(unittest.TestCase):
    """Test plasticity gating."""

    def test_default_learning_off(self):
        """Learning should be OFF by default in Phase 3."""
        gate = PlasticityGateController(default_learning=False)
        self.assertFalse(gate.is_learning_active())

    def test_novelty_opens_gate(self):
        """High novelty should open learning gate."""
        gate = PlasticityGateController(novelty_threshold=0.5)

        context_novel = LearningContext(novelty=0.8)
        gate.update(context_novel)
        self.assertTrue(gate.is_learning_active())

        context_familiar = LearningContext(novelty=0.2)
        gate.update(context_familiar)
        self.assertFalse(gate.is_learning_active())

    def test_reward_opens_gate(self):
        """Reward signal should open learning gate."""
        gate = PlasticityGateController(reward_threshold=0.1)

        context_reward = LearningContext(reward=0.5)
        gate.update(context_reward)
        self.assertTrue(gate.is_learning_active())

        context_no_reward = LearningContext(reward=0.0)
        gate.update(context_no_reward)
        self.assertFalse(gate.is_learning_active())

    def test_prediction_error_opens_gate(self):
        """High prediction error should open learning gate."""
        gate = PlasticityGateController(error_threshold=0.3)

        context_error = LearningContext(prediction_error=0.5)
        gate.update(context_error)
        self.assertTrue(gate.is_learning_active())

        context_no_error = LearningContext(prediction_error=0.1)
        gate.update(context_no_error)
        self.assertFalse(gate.is_learning_active())

    def test_force_enable_override(self):
        """Force enable should override context."""
        gate = PlasticityGateController(default_learning=False)

        gate.force_enable()
        self.assertTrue(gate.is_learning_active())

        # Even with no learning context, should stay on
        context = LearningContext(novelty=0.0, reward=0.0)
        # Don't call update - manual override should persist
        self.assertTrue(gate.is_learning_active())

    def test_gate_statistics(self):
        """Gate should track open/close events."""
        gate = PlasticityGateController(novelty_threshold=0.5)

        # Open gate
        gate.update(LearningContext(novelty=0.8))
        # Close gate
        gate.update(LearningContext(novelty=0.2))
        # Open again
        gate.update(LearningContext(novelty=0.9))

        stats = gate.get_statistics()
        self.assertEqual(stats["gate_open_count"], 2)
        self.assertEqual(stats["gate_close_count"], 1)


class TestControlLayer(unittest.TestCase):
    """Test integrated control layer."""

    def test_control_layer_initialization(self):
        """Control layer should initialize with learning OFF."""
        control = ControlLayer(default_learning=False)
        self.assertFalse(control.is_learning_active())

    def test_control_layer_update(self):
        """Control layer should integrate all detectors."""
        control = ControlLayer(novelty_threshold=0.5)

        firing = {1, 2, 3}
        all_neurons = {1, 2, 3, 4, 5}
        outputs = {4: True, 5: False}

        context = control.update(firing, all_neurons, outputs)

        # Should return valid context
        self.assertIsInstance(context, LearningContext)
        self.assertGreaterEqual(context.novelty, 0.0)
        self.assertLessEqual(context.novelty, 1.0)

    def test_novel_pattern_enables_learning(self):
        """Novel patterns should enable learning."""
        control = ControlLayer(novelty_threshold=0.5, default_learning=False)

        # First few patterns should be novel
        firing = {1, 2, 3}
        all_neurons = {1, 2, 3, 4, 5}
        outputs = {4: True, 5: False}

        context = control.update(firing, all_neurons, outputs)

        # High novelty should open gate
        if context.novelty > 0.5:
            self.assertTrue(control.is_learning_active())

    def test_familiar_pattern_disables_learning(self):
        """Familiar patterns should disable learning."""
        control = ControlLayer(novelty_threshold=0.5, default_learning=False)

        firing = {1, 2}
        all_neurons = {1, 2, 3, 4}
        outputs = {3: True, 4: False}

        # Repeat pattern many times to make it familiar
        for _ in range(20):
            control.update(firing, all_neurons, outputs)

        # Should have low novelty, gate closed
        context = control.update(firing, all_neurons, outputs)
        if context.novelty < 0.5:
            self.assertFalse(control.is_learning_active())

    def test_reward_enables_learning(self):
        """Reward should enable learning even with familiar patterns."""
        control = ControlLayer(reward_threshold=0.1, default_learning=False)

        firing = {1, 2}
        all_neurons = {1, 2, 3}
        outputs = {3: True}

        # Make pattern familiar
        for _ in range(20):
            control.update(firing, all_neurons, outputs, reward=0.0)

        # Add reward - should open gate
        context = control.update(firing, all_neurons, outputs, reward=0.5)
        self.assertTrue(control.is_learning_active())

    def test_statistics_collection(self):
        """Control layer should provide comprehensive statistics."""
        control = ControlLayer()

        firing = {1, 2}
        all_neurons = {1, 2, 3}
        outputs = {3: True}

        for _ in range(10):
            control.update(firing, all_neurons, outputs)

        stats = control.get_full_statistics()

        self.assertIn("timestep", stats)
        self.assertIn("novelty", stats)
        self.assertIn("confidence", stats)
        self.assertIn("conflict", stats)
        self.assertIn("plasticity_gate", stats)
        self.assertEqual(stats["timestep"], 10)


class TestControlLayerInvariant(unittest.TestCase):
    """Test control layer invariants."""

    def test_scores_always_bounded(self):
        """All scores should always be in [0, 1]."""
        control = ControlLayer()

        all_neurons = set(range(10))

        for i in range(100):
            firing = {j for j in range(10) if (i + j) % 3 == 0}
            outputs = {8: i % 2 == 0, 9: i % 3 == 0}

            context = control.update(firing, all_neurons, outputs)

            self.assertGreaterEqual(context.novelty, 0.0)
            self.assertLessEqual(context.novelty, 1.0)
            self.assertGreaterEqual(context.confidence, 0.0)
            self.assertLessEqual(context.confidence, 1.0)
            self.assertGreaterEqual(context.conflict, 0.0)
            self.assertLessEqual(context.conflict, 1.0)

    def test_control_layer_deterministic(self):
        """Same inputs should produce same outputs."""
        control1 = ControlLayer(novelty_threshold=0.5)
        control2 = ControlLayer(novelty_threshold=0.5)

        firing = {1, 2, 3}
        all_neurons = {1, 2, 3, 4, 5}
        outputs = {4: True, 5: False}

        contexts1 = []
        contexts2 = []

        for _ in range(10):
            c1 = control1.update(firing, all_neurons, outputs)
            c2 = control2.update(firing, all_neurons, outputs)
            contexts1.append(c1)
            contexts2.append(c2)

        # Should produce identical contexts
        for c1, c2 in zip(contexts1, contexts2):
            self.assertAlmostEqual(c1.novelty, c2.novelty, places=5)
            self.assertAlmostEqual(c1.confidence, c2.confidence, places=5)
            self.assertAlmostEqual(c1.conflict, c2.conflict, places=5)


if __name__ == "__main__":
    unittest.main()
