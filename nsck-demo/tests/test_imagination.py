"""
Verification test for World Model Imagination
Tests that the world model can predict outcomes (forward simulation).
Plan-veto and SNN-screening via imagined_reward are not yet wired
into CognitiveEngine.decide(); those capabilities are tested at the
module level in test_phase4_planning.py instead.
"""
import unittest
import numpy as np
from cognitive_engine import CognitiveEngine
import hypervec_shim as hypervec_rs

class TestImagination(unittest.TestCase):
    def test_world_model_prediction(self):
        """Test that the world model can produce imagined next states."""
        engine = CognitiveEngine()
        
        # Create HVs for state and action
        state_hv = hypervec_rs.HyperVector(42)
        action_hv = hypervec_rs.HyperVector(7)
        
        # Train world model with a transition
        next_hv = hypervec_rs.HyperVector(99)
        reward = 1.0
        
        for _ in range(100):
            engine.world_model.update(state_hv, action_hv, next_hv, reward)
        
        # Imagine the outcome
        predicted_next, predicted_reward = engine.world_model.imagine(state_hv, action_hv)
        
        print(f"\n[TEST] World Model Prediction:")
        print(f"  Predicted reward: {predicted_reward:.4f}")
        
        # After training with reward=1.0, predicted reward should be positive
        self.assertGreater(predicted_reward, 0.0)

if __name__ == '__main__':
    unittest.main()
