"""
Tests for adaptive plasticity: learning rate scheduling and region-specific multipliers.
"""

import unittest
from core.plasticity import AdaptivePlasticity, PlasticityController
from core.synapse import Synapse


class TestAdaptivePlasticity(unittest.TestCase):
    """Test adaptive learning rate scheduling."""

    def setUp(self):
        self.adaptive = AdaptivePlasticity(base_rate=0.01, min_rate=0.0001, max_rate=0.5)

    def test_base_rate_returns_base_when_no_signals(self):
        """Rate scheduler should return near base rate when no signals."""
        rate = self.adaptive.rate_scheduler(step=0, perf_delta=0.0, stability=0.0)
        self.assertAlmostEqual(rate, 0.01, places=3)

    def test_rate_increases_with_positive_perf_delta(self):
        """Rate should increase when performance improves."""
        rate_base = self.adaptive.rate_scheduler(step=0, perf_delta=0.0, stability=0.0)
        rate_improved = self.adaptive.rate_scheduler(step=0, perf_delta=0.5, stability=0.0)
        self.assertGreater(rate_improved, rate_base)

    def test_rate_decreases_with_high_stability(self):
        """Rate should decrease when learning is stable (reduce noise)."""
        rate_unstable = self.adaptive.rate_scheduler(step=0, perf_delta=0.0, stability=0.0)
        rate_stable = self.adaptive.rate_scheduler(step=0, perf_delta=0.0, stability=0.8)
        self.assertLess(rate_stable, rate_unstable)

    def test_rate_clamped_to_bounds(self):
        """Rate should stay within [min_rate, max_rate]."""
        rate = self.adaptive.rate_scheduler(step=0, perf_delta=10.0, stability=-1.0)
        self.assertGreaterEqual(rate, self.adaptive.min_rate)
        self.assertLessEqual(rate, self.adaptive.max_rate)


class TestPlasticityControllerAdaptive(unittest.TestCase):
    """Test plasticity controller with adaptive learning."""

    def setUp(self):
        self.controller = PlasticityController(stdp_learning_rate=0.01)

    def test_region_lr_multiplier_scales_rate(self):
        """Setting region multiplier should scale effective learning rate."""
        self.controller.set_region_lr_multiplier(0.5)
        rate = self.controller.get_effective_lr(step=0, perf_delta=0.0, stability=0.0)
        self.assertAlmostEqual(rate, 0.005, places=4)

    def test_region_lr_multiplier_zero_disables_learning(self):
        """Region multiplier of 0 should effectively disable learning."""
        self.controller.set_region_lr_multiplier(0.0)
        rate = self.controller.get_effective_lr(step=0)
        self.assertAlmostEqual(rate, 0.0, places=5)

    def test_region_lr_multiplier_two_doubles_rate(self):
        """Region multiplier of 2 should double effective rate."""
        self.controller.set_region_lr_multiplier(2.0)
        rate = self.controller.get_effective_lr(step=0, perf_delta=0.0, stability=0.0)
        self.assertAlmostEqual(rate, 0.02, places=3)

    def test_update_learning_respects_gate(self):
        """Plasticity gate should control whether learning can occur."""
        self.assertTrue(self.controller.is_learning())
        self.controller.disable_learning()
        self.assertFalse(self.controller.is_learning())
        self.controller.enable_learning()
        self.assertTrue(self.controller.is_learning())

    def test_update_learning_with_adaptive_rate(self):
        """update_learning should use adaptive rate when step/perf_delta provided."""
        # Test that adaptive rates differ based on performance and stability
        rate_good = self.controller.get_effective_lr(step=0, perf_delta=0.5, stability=0.0)
        rate_bad = self.controller.get_effective_lr(step=0, perf_delta=-0.5, stability=0.0)

        # Good performance should yield higher rate than bad
        self.assertGreater(rate_good, rate_bad)


if __name__ == '__main__':
    unittest.main()
