"""
Tests for Phase 5 Self-Model

Tests outcome prediction, residual tracking, and learning improvement.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.self_model import SelfModel


class TestSelfModeling(unittest.TestCase):
    """Tests for Phase 5 self-modeling."""

    def setUp(self):
        """Set up SelfModel instance for each test."""
        self.model = SelfModel(input_dim=4, num_actions=4)

    def test_initial_prediction_high_uncertainty(self):
        """Test that initial predictions have high uncertainty."""
        state = [0.5, 0.3, 0.2, 0.1]
        action = 0

        predicted, uncertainty = self.model.predict_outcome(state, action)

        # Should have high uncertainty for novel state-action
        self.assertGreater(uncertainty, 0.5)
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(predicted), len(state))

    def test_update_reduces_residual(self):
        """Test that repeated updates reduce prediction error."""
        state = [0.5, 0.3, 0.2, 0.1]
        action = 0
        next_state = [0.6, 0.3, 0.2, 0.1]

        # First update
        residual_1 = self.model.update(state, action, next_state)

        # Repeat same transition
        residual_2 = self.model.update(state, action, next_state)
        residual_3 = self.model.update(state, action, next_state)

        # Later residuals should be smaller (learning the pattern)
        self.assertLess(residual_3, residual_1)

    def test_residual_computation(self):
        """Test residual error computation."""
        prediction = [0.5, 0.5, 0.5, 0.5]
        actual = [0.5, 0.5, 0.5, 0.5]

        # Perfect prediction
        residual_perfect = self.model.compute_residual_error(prediction, actual)
        self.assertEqual(residual_perfect, 0.0)

        # Imperfect prediction
        actual_diff = [0.6, 0.4, 0.6, 0.4]
        residual_imperfect = self.model.compute_residual_error(prediction, actual_diff)
        self.assertGreater(residual_imperfect, 0.0)

    def test_uncertainty_decreases_with_experience(self):
        """Test that uncertainty decreases as model learns."""
        state = [0.3, 0.3, 0.3, 0.3]
        action = 1
        next_state = [0.4, 0.3, 0.3, 0.3]

        # Initial prediction (high uncertainty)
        _, uncertainty_1 = self.model.predict_outcome(state, action)

        # Learn from experience
        for _ in range(5):
            self.model.update(state, action, next_state)

        # Later prediction (lower uncertainty)
        _, uncertainty_2 = self.model.predict_outcome(state, action)

        self.assertLess(uncertainty_2, uncertainty_1)

    def test_mean_residual_tracking(self):
        """Test that mean residual is tracked correctly."""
        state = [0.2, 0.2, 0.2, 0.2]

        # Make several predictions with known errors
        self.model.update(state, 0, [0.3, 0.2, 0.2, 0.2])  # Error in first dim
        self.model.update(state, 1, [0.2, 0.3, 0.2, 0.2])  # Error in second dim

        mean_residual = self.model.get_mean_residual()

        # Should have non-zero mean residual
        self.assertGreater(mean_residual, 0.0)
        self.assertLess(mean_residual, 1.0)

    def test_residual_variance_tracking(self):
        """Test residual variance computation."""
        state = [0.5, 0.5, 0.5, 0.5]

        # Consistent errors
        for i in range(5):
            self.model.update(state, 0, [0.6, 0.5, 0.5, 0.5])

        variance_consistent = self.model.get_residual_variance()

        # Clear and test variable errors
        self.model.clear()
        self.model.update(state, 0, [0.6, 0.5, 0.5, 0.5])
        self.model.update(state, 0, [0.9, 0.5, 0.5, 0.5])
        self.model.update(state, 0, [0.3, 0.5, 0.5, 0.5])

        variance_variable = self.model.get_residual_variance()

        # Variable errors should have higher variance
        self.assertGreater(variance_variable, variance_consistent)

    def test_different_actions_tracked_separately(self):
        """Test that different actions lead to different predictions."""
        state = [0.5, 0.5, 0.5, 0.5]

        # Learn action 0 -> increase first dim
        for _ in range(3):
            self.model.update(state, 0, [0.7, 0.5, 0.5, 0.5])

        # Learn action 1 -> increase second dim
        for _ in range(3):
            self.model.update(state, 1, [0.5, 0.7, 0.5, 0.5])

        # Predictions should be different
        pred_0, _ = self.model.predict_outcome(state, 0)
        pred_1, _ = self.model.predict_outcome(state, 1)

        # Action 0 should predict higher first dim
        self.assertGreater(pred_0[0], pred_1[0])
        # Action 1 should predict higher second dim
        self.assertGreater(pred_1[1], pred_0[1])


if __name__ == '__main__':
    unittest.main()
