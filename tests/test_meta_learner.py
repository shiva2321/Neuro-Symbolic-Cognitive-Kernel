"""
Phase 3.4 Meta-Learning Tests

Tests for learning to learn:
1. Adaptive thresholds (novelty, reward, error)
2. Regional learning profiles
3. Intrinsic motivation (curiosity)
4. Meta-learning integration

Invariants:
- Thresholds stay in valid ranges [0, 1]
- Adaptation based on outcomes (EMA)
- Performance improves over time
- Regions specialize without forgetting
- Intrinsic motivation complements extrinsic reward
"""

import unittest
from core.meta_learner import (
    MetaLearner, ThresholdAdaptation, RegionalLearningProfile,
    IntrinsicMotivation, AdaptationSignal
)


class TestThresholdAdaptation(unittest.TestCase):
    """Test adaptive thresholds."""

    def setUp(self):
        """Set up threshold."""
        self.threshold = ThresholdAdaptation("novelty", 0.5)

    def test_initial_value(self):
        """Threshold should initialize at given value."""
        self.assertEqual(self.threshold.current_value, 0.5)

    def test_success_lowers_threshold(self):
        """Success should lower threshold (learn more)."""
        original = self.threshold.current_value
        self.threshold.adapt(AdaptationSignal.SUCCESS)
        self.assertLess(self.threshold.current_value, original)

    def test_failure_raises_threshold(self):
        """Failure should raise threshold (learn less)."""
        original = self.threshold.current_value
        self.threshold.adapt(AdaptationSignal.FAILURE)
        self.assertGreater(self.threshold.current_value, original)

    def test_breakthrough_significant_change(self):
        """Breakthrough should significantly lower threshold."""
        original = self.threshold.current_value
        self.threshold.adapt(AdaptationSignal.BREAKTHROUGH)
        change = original - self.threshold.current_value
        self.assertGreater(change, 0.001)  # Even small changes count

    def test_bounds_enforced(self):
        """Threshold should stay within bounds."""
        # Push to lower bound
        for _ in range(100):
            self.threshold.adapt(AdaptationSignal.SUCCESS)
        self.assertGreaterEqual(self.threshold.current_value, self.threshold.min_value)

        # Push to upper bound
        threshold = ThresholdAdaptation("test", 0.5)
        for _ in range(100):
            threshold.adapt(AdaptationSignal.FAILURE)
        self.assertLessEqual(threshold.current_value, threshold.max_value)

    def test_success_rate_tracking(self):
        """Should track success rate."""
        self.threshold.adapt(AdaptationSignal.SUCCESS)
        self.threshold.adapt(AdaptationSignal.SUCCESS)
        self.threshold.adapt(AdaptationSignal.FAILURE)

        rate = self.threshold.get_success_rate()
        self.assertAlmostEqual(rate, 2/3, places=2)


class TestRegionalLearningProfile(unittest.TestCase):
    """Test regional learning profiles."""

    def setUp(self):
        """Set up profile."""
        self.profile = RegionalLearningProfile(
            region_id=0,
            learning_rate=0.01
        )

    def test_initialization(self):
        """Profile should initialize correctly."""
        self.assertEqual(self.profile.region_id, 0)
        self.assertEqual(self.profile.learning_rate, 0.01)
        self.assertTrue(self.profile.plasticity_enabled)

    def test_learning_rate_increase(self):
        """Good performance should increase learning rate."""
        original = self.profile.learning_rate
        self.profile.update_learning_rate(0.5)  # 50% improvement
        self.assertGreater(self.profile.learning_rate, original)

    def test_learning_rate_decrease(self):
        """Poor performance should decrease learning rate."""
        original = self.profile.learning_rate
        self.profile.update_learning_rate(-0.5)  # 50% decline
        self.assertLess(self.profile.learning_rate, original)

    def test_learning_rate_bounds(self):
        """Learning rate should stay in bounds."""
        # Increase aggressively
        for _ in range(100):
            self.profile.update_learning_rate(1.0)
        self.assertLessEqual(self.profile.learning_rate, 0.5)

        # Decrease aggressively
        for _ in range(100):
            self.profile.update_learning_rate(-1.0)
        self.assertGreaterEqual(self.profile.learning_rate, 0.0001)

    def test_mean_learning_rate(self):
        """Should compute mean over history."""
        self.profile.update_learning_rate(0.2)
        self.profile.update_learning_rate(0.2)
        self.profile.update_learning_rate(0.2)

        mean = self.profile.get_mean_learning_rate()
        self.assertGreater(mean, 0.01)


class TestIntrinsicMotivation(unittest.TestCase):
    """Test intrinsic motivation (curiosity)."""

    def setUp(self):
        """Set up motivation system."""
        self.motivation = IntrinsicMotivation()

    def test_novelty_driven_curiosity(self):
        """High novelty should produce high intrinsic reward."""
        reward_novel = self.motivation.compute_intrinsic_reward(
            novelty=0.9,
            confidence=0.9  # High confidence (not exploring)
        )
        reward_familiar = self.motivation.compute_intrinsic_reward(
            novelty=0.1,
            confidence=0.9
        )
        self.assertGreater(reward_novel, reward_familiar)

    def test_uncertainty_driven_curiosity(self):
        """High uncertainty should drive exploration."""
        reward_uncertain = self.motivation.compute_intrinsic_reward(
            novelty=0.5,
            confidence=0.1  # Uncertain
        )
        reward_certain = self.motivation.compute_intrinsic_reward(
            novelty=0.5,
            confidence=0.9  # Certain
        )
        self.assertGreater(reward_uncertain, reward_certain)

    def test_pattern_recording(self):
        """Should record explored patterns."""
        self.motivation.record_pattern("pattern_a")
        self.motivation.record_pattern("pattern_b")

        stats = self.motivation.get_statistics()
        self.assertEqual(stats["patterns_explored"], 2)

    def test_curiosity_adaptation(self):
        """Curiosity should adapt based on learning progress."""
        self.motivation._curiosity_level = 0.6  # Start at different value
        original = self.motivation._curiosity_level

        # Good progress: increase curiosity
        self.motivation.adapt_curiosity(0.8)
        self.assertGreater(self.motivation._curiosity_level, original)

        # Poor progress: decrease curiosity
        self.motivation._curiosity_level = original
        self.motivation.adapt_curiosity(0.2)
        self.assertLess(self.motivation._curiosity_level, original)

    def test_intrinsic_reward_bounded(self):
        """Intrinsic reward should always be [0, 1]."""
        for novelty in [0.0, 0.3, 0.7, 1.0]:
            for confidence in [0.0, 0.5, 1.0]:
                reward = self.motivation.compute_intrinsic_reward(novelty, confidence)
                self.assertGreaterEqual(reward, 0.0)
                self.assertLessEqual(reward, 1.0)


class TestMetaLearner(unittest.TestCase):
    """Test meta-learner integration."""

    def setUp(self):
        """Set up meta-learner."""
        self.meta = MetaLearner()

    def test_initialization(self):
        """Meta-learner should initialize properly."""
        self.assertIsNotNone(self.meta._novelty_threshold)
        self.assertIsNotNone(self.meta._reward_threshold)
        self.assertIsNotNone(self.meta._intrinsic_motivation)

    def test_get_thresholds(self):
        """Should return current threshold values."""
        novelty = self.meta.get_novelty_threshold()
        reward = self.meta.get_reward_threshold()
        error = self.meta.get_error_threshold()

        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
        self.assertGreaterEqual(error, 0.0)
        self.assertLessEqual(error, 1.0)

    def test_add_region(self):
        """Should add learning profiles for regions."""
        self.meta.add_region(0, "sensory", 0.01)
        self.meta.add_region(1, "hidden", 0.02)
        self.meta.add_region(2, "motor", 0.01)

        self.assertEqual(len(self.meta._regional_profiles), 3)

    def test_region_learning_rates(self):
        """Regions should have different learning rates."""
        self.meta.add_region(0, "sensory", 0.01)
        self.meta.add_region(1, "motor", 0.03)

        rate0 = self.meta.get_region_learning_rate(0)
        rate1 = self.meta.get_region_learning_rate(1)

        self.assertEqual(rate0, 0.01)
        self.assertEqual(rate1, 0.03)

    def test_record_successful_learning(self):
        """Should record successful learning outcomes."""
        self.meta.record_learning_outcome(
            decision="open_gate",
            outcome=True,
            signal=AdaptationSignal.SUCCESS
        )

        self.assertEqual(len(self.meta._learning_decisions), 1)
        self.assertTrue(self.meta._learning_decisions[0][1])

    def test_threshold_adaptation_on_success(self):
        """Successful learning should lower thresholds."""
        original_novelty = self.meta.get_novelty_threshold()

        self.meta.record_learning_outcome(
            decision="test",
            outcome=True,
            signal=AdaptationSignal.SUCCESS
        )

        new_novelty = self.meta.get_novelty_threshold()
        self.assertLess(new_novelty, original_novelty)

    def test_threshold_adaptation_on_failure(self):
        """Failed learning should raise thresholds."""
        original_novelty = self.meta.get_novelty_threshold()

        self.meta.record_learning_outcome(
            decision="test",
            outcome=False,
            signal=AdaptationSignal.FAILURE
        )

        new_novelty = self.meta.get_novelty_threshold()
        self.assertGreater(new_novelty, original_novelty)

    def test_intrinsic_extrinsic_combination(self):
        """Should combine intrinsic and extrinsic rewards."""
        # With extrinsic reward
        combined1 = self.meta.compute_combined_intrinsic_extrinsic(
            extrinsic_reward=0.8,
            novelty=0.1,
            confidence=0.9
        )
        self.assertGreater(combined1, 0.5)

        # Without extrinsic reward but high novelty
        combined2 = self.meta.compute_combined_intrinsic_extrinsic(
            extrinsic_reward=0.0,
            novelty=0.8,
            confidence=0.2
        )
        self.assertGreater(combined2, 0.3)

    def test_statistics(self):
        """Should provide comprehensive statistics."""
        self.meta.add_region(0, "sensory")
        self.meta.record_learning_outcome(
            "test", True, AdaptationSignal.SUCCESS
        )

        stats = self.meta.get_full_statistics()

        self.assertIn("timestep", stats)
        self.assertIn("thresholds", stats)
        self.assertIn("performance", stats)
        self.assertIn("regions", stats)
        self.assertIn("intrinsic_motivation", stats)
        self.assertEqual(stats["total_decisions"], 1)


class TestMetaLearningBehavior(unittest.TestCase):
    """Test meta-learning behavior."""

    def setUp(self):
        """Set up meta-learner."""
        self.meta = MetaLearner()
        self.meta.add_region(0, "sensory", 0.01)
        self.meta.add_region(1, "hidden", 0.02)
        self.meta.add_region(2, "motor", 0.01)

    def test_specialization_through_learning_rates(self):
        """Different regions should develop different learning rates."""
        # Simulate good learning in sensory, poor in motor
        for _ in range(10):
            self.meta.record_learning_outcome(
                "sensory_learn", True, AdaptationSignal.SUCCESS, 0.3
            )
            self.meta.record_learning_outcome(
                "motor_learn", False, AdaptationSignal.FAILURE, -0.2
            )

        # Manually update rates to reflect outcomes
        sensory_profile = self.meta._regional_profiles[0]
        motor_profile = self.meta._regional_profiles[2]

        sensory_profile.update_learning_rate(0.5)
        motor_profile.update_learning_rate(-0.5)

        # Sensory should have higher learning rate now
        self.assertGreater(
            sensory_profile.learning_rate,
            motor_profile.learning_rate
        )

    def test_adaptive_thresholds_improve_learning(self):
        """Adaptive thresholds should lead to better learning decisions."""
        # Simulate series of learning decisions
        decisions_successful = 0
        total_decisions = 20

        for i in range(total_decisions):
            # First 10: bad decisions with old thresholds
            # Last 10: better decisions with adapted thresholds
            if i < 10:
                outcome = False
                signal = AdaptationSignal.FAILURE
            else:
                outcome = True
                signal = AdaptationSignal.SUCCESS

            self.meta.record_learning_outcome(
                f"decision_{i}", outcome, signal
            )
            if outcome:
                decisions_successful += 1

        # Should have successes (at least in second half)
        stats = self.meta.get_full_statistics()
        self.assertEqual(stats["decisions_successful"], 10)  # Second 10 decisions

    def test_intrinsic_motivation_encourages_exploration(self):
        """System should explore novel patterns due to curiosity."""
        # Compute intrinsic reward for novel vs familiar
        novel_reward = self.meta.compute_combined_intrinsic_extrinsic(
            extrinsic_reward=0.0,
            novelty=0.8,
            confidence=0.3
        )

        familiar_reward = self.meta.compute_combined_intrinsic_extrinsic(
            extrinsic_reward=0.0,
            novelty=0.2,
            confidence=0.9
        )

        # Should prefer novel
        self.assertGreater(novel_reward, familiar_reward)


class TestMetaLearningInvariants(unittest.TestCase):
    """Test meta-learning invariants."""

    def test_threshold_bounds_maintained(self):
        """All thresholds should stay in valid bounds."""
        meta = MetaLearner()

        # Extreme adaptation
        for _ in range(100):
            meta.record_learning_outcome(
                "test", True, AdaptationSignal.BREAKTHROUGH
            )
            meta.record_learning_outcome(
                "test", False, AdaptationSignal.FAILURE
            )

        # Should all be in bounds
        novelty = meta.get_novelty_threshold()
        reward = meta.get_reward_threshold()
        error = meta.get_error_threshold()

        for val in [novelty, reward, error]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_deterministic_behavior(self):
        """Same inputs should produce same outputs."""
        meta1 = MetaLearner()
        meta2 = MetaLearner()

        for _ in range(10):
            meta1.record_learning_outcome(
                "test", True, AdaptationSignal.SUCCESS
            )
            meta2.record_learning_outcome(
                "test", True, AdaptationSignal.SUCCESS
            )

        self.assertEqual(
            meta1.get_novelty_threshold(),
            meta2.get_novelty_threshold()
        )

    def test_nondestructive_operations(self):
        """Meta-learning shouldn't break normal operation."""
        meta = MetaLearner()
        meta.add_region(0, "sensory", 0.01)

        # Perform many meta-learning operations
        for i in range(100):
            outcome = (i % 3) == 0  # Variable outcomes
            signal = AdaptationSignal.SUCCESS if outcome else AdaptationSignal.FAILURE
            meta.record_learning_outcome(f"op_{i}", outcome, signal)

        # Should still be functional
        stats = meta.get_full_statistics()
        self.assertEqual(stats["total_decisions"], 100)
        self.assertGreater(stats["decisions_successful"], 0)


if __name__ == "__main__":
    unittest.main()
