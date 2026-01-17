"""
Tests for Phase 5 Intrinsic Motivation

Tests the intrinsic reward computation based on novelty and prediction error changes.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.metrics import SignalComputer


class TestIntrinsicMotivation(unittest.TestCase):
    """Tests for Phase 5 intrinsic motivation computation."""

    def setUp(self):
        """Set up SignalComputer instance for each test."""
        self.computer = SignalComputer(window_size=100)

    def test_intrinsic_increases_with_novelty(self):
        """Test that intrinsic reward increases with novelty."""
        # Low novelty
        low_intrinsic = self.computer.compute_intrinsic_reward(
            novelty=0.0,
            pred_error_delta=0.0,
            beta_intrinsic=0.5
        )

        # High novelty
        high_intrinsic = self.computer.compute_intrinsic_reward(
            novelty=1.0,
            pred_error_delta=0.0,
            beta_intrinsic=0.5
        )

        self.assertGreater(high_intrinsic, low_intrinsic)
        self.assertGreaterEqual(low_intrinsic, 0.0)
        self.assertLessEqual(high_intrinsic, 1.0)

    def test_intrinsic_increases_with_pred_error_change(self):
        """Test that intrinsic reward increases with prediction error change."""
        # No error change
        no_change = self.computer.compute_intrinsic_reward(
            novelty=0.0,
            pred_error_delta=0.0,
            beta_intrinsic=0.5
        )

        # Large error change (positive)
        large_change_pos = self.computer.compute_intrinsic_reward(
            novelty=0.0,
            pred_error_delta=0.8,
            beta_intrinsic=0.5
        )

        # Large error change (negative - abs value used)
        large_change_neg = self.computer.compute_intrinsic_reward(
            novelty=0.0,
            pred_error_delta=-0.8,
            beta_intrinsic=0.5
        )

        self.assertGreater(large_change_pos, no_change)
        self.assertGreater(large_change_neg, no_change)
        self.assertAlmostEqual(large_change_pos, large_change_neg, places=5)

    def test_intrinsic_capped_at_one(self):
        """Test that intrinsic reward is capped at 1.0."""
        # Maximum values should cap at 1.0
        max_intrinsic = self.computer.compute_intrinsic_reward(
            novelty=1.0,
            pred_error_delta=1.0,
            beta_intrinsic=10.0  # Excessive beta
        )

        self.assertLessEqual(max_intrinsic, 1.0)
        self.assertEqual(max_intrinsic, 1.0)

    def test_beta_intrinsic_weight(self):
        """Test that beta_intrinsic controls reward magnitude."""
        # Low beta
        low_beta = self.computer.compute_intrinsic_reward(
            novelty=0.5,
            pred_error_delta=0.5,
            beta_intrinsic=0.1
        )

        # High beta
        high_beta = self.computer.compute_intrinsic_reward(
            novelty=0.5,
            pred_error_delta=0.5,
            beta_intrinsic=0.9
        )

        self.assertGreater(high_beta, low_beta)
        # With beta=0, should get zero
        zero_beta = self.computer.compute_intrinsic_reward(
            novelty=0.5,
            pred_error_delta=0.5,
            beta_intrinsic=0.0
        )
        self.assertEqual(zero_beta, 0.0)

    def test_combined_novelty_and_error(self):
        """Test combined novelty and prediction error contribution."""
        # Only novelty
        only_novelty = self.computer.compute_intrinsic_reward(
            novelty=1.0,
            pred_error_delta=0.0,
            beta_intrinsic=1.0
        )

        # Only error
        only_error = self.computer.compute_intrinsic_reward(
            novelty=0.0,
            pred_error_delta=1.0,
            beta_intrinsic=1.0
        )

        # Both
        both = self.computer.compute_intrinsic_reward(
            novelty=1.0,
            pred_error_delta=1.0,
            beta_intrinsic=1.0
        )

        # Both should be greater (but capped at 1.0)
        self.assertGreaterEqual(both, only_novelty)
        self.assertGreaterEqual(both, only_error)
        # With 50/50 weighting, each contributes 0.5 when at max
        self.assertAlmostEqual(only_novelty, 0.5, places=5)
        self.assertAlmostEqual(only_error, 0.5, places=5)

    def test_typical_values(self):
        """Test intrinsic reward computation with typical values."""
        # Moderate novelty and error change
        typical = self.computer.compute_intrinsic_reward(
            novelty=0.6,
            pred_error_delta=0.3,
            beta_intrinsic=0.1
        )

        # Should be small but positive
        self.assertGreater(typical, 0.0)
        self.assertLess(typical, 0.1)  # Beta scales it down

        # Expected: (0.6*0.5 + 0.3*0.5) * 0.1 = 0.45 * 0.1 = 0.045
        expected = (0.6 * 0.5 + 0.3 * 0.5) * 0.1
        self.assertAlmostEqual(typical, expected, places=5)


if __name__ == '__main__':
    unittest.main()
