"""
Tests for signal computation: confidence, novelty, error, stability.
"""

import unittest
from core.metrics import SignalComputer


class TestSignalComputer(unittest.TestCase):
    """Test signal computation for meta-learning."""

    def setUp(self):
        self.computer = SignalComputer(window_size=10)

    def test_confidence_starts_neutral(self):
        """Initial confidence should be ~0.5 (neutral)."""
        conf = self.computer.compute_confidence()
        self.assertAlmostEqual(conf, 0.5)

    def test_confidence_increases_with_stable_weights(self):
        """Confidence should increase when weights are consistent."""
        # All weights the same = low variance
        for _ in range(5):
            self.computer.update([0.5, 0.5, 0.5], 0.25)
        conf_stable = self.computer.compute_confidence()

        # Reset and add variable weights
        self.computer.clear()
        for _ in range(5):
            self.computer.update([0.1, 0.9, 0.5], 0.25)
        conf_variable = self.computer.compute_confidence()

        self.assertGreater(conf_stable, conf_variable)

    def test_novelty_detects_new_patterns(self):
        """Novelty should be high for unseen patterns."""
        novelty_new = self.computer.compute_novelty("pattern_A")
        self.assertAlmostEqual(novelty_new, 1.0)

        # After recording, novelty of same pattern should be low
        self.computer.update([], 0.0, 0.0, "pattern_A")
        novelty_seen = self.computer.compute_novelty("pattern_A")
        self.assertAlmostEqual(novelty_seen, 0.0)

    def test_novelty_zero_for_empty_pattern(self):
        """Novelty should be 0 for empty pattern ID."""
        novelty = self.computer.compute_novelty("")
        self.assertAlmostEqual(novelty, 0.0)

    def test_error_accumulates(self):
        """Error should be average of samples."""
        self.computer.update([], 0.0, 0.5)
        self.computer.update([], 0.0, 0.3)
        error = self.computer.compute_error()
        expected = (0.5 + 0.3) / 2
        self.assertAlmostEqual(error, expected)

    def test_stability_increases_with_consistent_firing(self):
        """Stability should increase when firing rate is consistent."""
        # Consistent firing rate
        for _ in range(5):
            self.computer.update([], 0.25)
        stab_consistent = self.computer.compute_stability()

        # Reset and use variable firing rates
        self.computer.clear()
        for i in range(5):
            self.computer.update([], 0.1 + (i * 0.1))
        stab_variable = self.computer.compute_stability()

        self.assertGreater(stab_consistent, stab_variable)

    def test_get_signals_schema_valid(self):
        """get_signals should return all required keys."""
        signals = self.computer.get_signals()
        required_keys = {'confidence', 'novelty', 'error', 'stability', 'performance_delta'}
        self.assertEqual(set(signals.keys()), required_keys)

        # All values should be floats in [0, 1] (except perf_delta which is [-1, 1])
        for key, val in signals.items():
            self.assertIsInstance(val, float)
            if key != 'performance_delta':
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_window_size_limits_history(self):
        """Signal computer should respect window size."""
        computer = SignalComputer(window_size=5)
        for i in range(10):
            computer.update([float(i)], 0.0)

        # Only last 5 should be in history
        self.assertEqual(len(computer._weight_history), 5)

    def test_clear_resets_all_state(self):
        """Clearing should reset confidence/novelty/error/stability to neutral."""
        # Add some data
        for _ in range(5):
            self.computer.update([0.5], 0.25, 0.5, "pattern_X")

        # Clear
        self.computer.clear()

        # Should be back to neutral
        signals = self.computer.get_signals()
        self.assertAlmostEqual(signals['confidence'], 0.5)
        self.assertAlmostEqual(signals['error'], 0.0)
        self.assertAlmostEqual(signals['novelty'], 0.0)
        self.assertAlmostEqual(signals['stability'], 0.5)


if __name__ == '__main__':
    unittest.main()
