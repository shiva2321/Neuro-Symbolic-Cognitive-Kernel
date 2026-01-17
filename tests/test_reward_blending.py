"""
Tests for Phase 5 Reward Blending in ControlLayer

Tests the blending of extrinsic and intrinsic rewards.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.control_layer import ControlLayer


class TestRewardBlending(unittest.TestCase):
    """Tests for Phase 5 reward blending."""

    def setUp(self):
        """Set up ControlLayer instance for each test."""
        self.control = ControlLayer(beta_blend=0.5)

    def test_beta_zero_all_extrinsic(self):
        """Test that β=0 returns only extrinsic reward."""
        extrinsic = 0.8
        intrinsic = 0.3

        blended = self.control.blend_rewards(extrinsic, intrinsic, beta=0.0)

        self.assertAlmostEqual(blended, extrinsic, places=5)

    def test_beta_one_all_intrinsic(self):
        """Test that β=1 returns only intrinsic reward."""
        extrinsic = 0.8
        intrinsic = 0.3

        blended = self.control.blend_rewards(extrinsic, intrinsic, beta=1.0)

        self.assertAlmostEqual(blended, intrinsic, places=5)

    def test_beta_half_equal_blend(self):
        """Test that β=0.5 gives equal blend of both rewards."""
        extrinsic = 0.6
        intrinsic = 0.4

        blended = self.control.blend_rewards(extrinsic, intrinsic, beta=0.5)

        expected = (extrinsic * 0.5) + (intrinsic * 0.5)
        self.assertAlmostEqual(blended, expected, places=5)

    def test_default_beta_from_init(self):
        """Test that default beta from __init__ is used when not provided."""
        control = ControlLayer(beta_blend=0.3)

        extrinsic = 0.7
        intrinsic = 0.5

        blended = control.blend_rewards(extrinsic, intrinsic)

        # Should use beta=0.3
        expected = extrinsic * 0.7 + intrinsic * 0.3
        self.assertAlmostEqual(blended, expected, places=5)

    def test_beta_clamping(self):
        """Test that beta values outside [0, 1] are clamped."""
        extrinsic = 0.6
        intrinsic = 0.4

        # Beta > 1 should clamp to 1
        blended_high = self.control.blend_rewards(extrinsic, intrinsic, beta=2.0)
        self.assertAlmostEqual(blended_high, intrinsic, places=5)

        # Beta < 0 should clamp to 0
        blended_low = self.control.blend_rewards(extrinsic, intrinsic, beta=-0.5)
        self.assertAlmostEqual(blended_low, extrinsic, places=5)

    def test_various_beta_values(self):
        """Test blending with various beta values."""
        extrinsic = 0.8
        intrinsic = 0.2

        # β = 0.2: mostly extrinsic (0.8*0.8 + 0.2*0.2 = 0.64 + 0.04 = 0.68)
        blend_02 = self.control.blend_rewards(extrinsic, intrinsic, beta=0.2)
        expected_02 = 0.8 * 0.8 + 0.2 * 0.2
        self.assertAlmostEqual(blend_02, expected_02, places=5)

        # β = 0.7: mostly intrinsic (0.8*0.3 + 0.2*0.7 = 0.24 + 0.14 = 0.38)
        blend_07 = self.control.blend_rewards(extrinsic, intrinsic, beta=0.7)
        expected_07 = 0.8 * 0.3 + 0.2 * 0.7
        self.assertAlmostEqual(blend_07, expected_07, places=5)

        # When extrinsic > intrinsic, higher beta decreases total reward
        # (emphasizes lower intrinsic value more)
        self.assertGreater(blend_02, blend_07)

    def test_zero_rewards(self):
        """Test blending when both rewards are zero."""
        blended = self.control.blend_rewards(0.0, 0.0, beta=0.5)
        self.assertEqual(blended, 0.0)

    def test_max_rewards(self):
        """Test blending when both rewards are at maximum."""
        blended = self.control.blend_rewards(1.0, 1.0, beta=0.5)
        self.assertEqual(blended, 1.0)


if __name__ == '__main__':
    unittest.main()
