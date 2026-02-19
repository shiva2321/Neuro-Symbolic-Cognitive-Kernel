"""
Verification test for Phase 5.1: World Model (Dynamics Predictor)
"""
import unittest
import torch
import numpy as np
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.neural.world_model import WorldModel
import python.core.vsa.hypervec_shim as hypervec_rs

class TestWorldModel(unittest.TestCase):
    def test_dynamics_learning(self):
        """Test that the world model can learn a simple (S, A) -> S' transition."""
        engine = CognitiveEngine()
        
        # Define a simple transition pattern
        # UP action in a specific state leads to a "Success" state
        state_1 = {"head": (5, 5), "food": (5, 4)} # Food is above
        action = "ACTION_UP"
        next_state = {"head": (5, 4), "food": (5, 4)} # Head on food
        reward = 1.0
        
        # Train multiple times on this pattern
        print("\n[TEST] Training World Model on pattern: UP leads to Success...")
        for i in range(200):
            # Must call decide to set current_state.situation_hv
            engine.decide(state_1, "snake")
            
            engine.learn(
                state=state_1,
                action=action,
                reward=reward,
                task_tag="snake",
                outcome="success",
                next_state=next_state
            )
        
        # Verify prediction
        # Get HV for state_1 and action
        verifier = engine.verifiers["snake"]
        s1_preds = verifier.get_active_predicates(state_1, "snake")
        s1_hv = engine.episodic_memory.create_situation_hv(state_1, "snake", s1_preds)
        
        from python.core.perception.symbol_grounding import GLOBAL_PRIMITIVES_MAP
        a_hv = hypervec_rs.HyperVector(GLOBAL_PRIMITIVES_MAP[action])
        
        pred_ns_hv_bits, pred_reward = engine.world_model.imagine(s1_hv, a_hv)
        
        print(f"[TEST] Predicted Reward: {pred_reward:.4f} (Expected close to 1.0)")
        self.assertGreater(pred_reward, 0.5)
        
        # Check predicted state similarity to actual next state
        ns_preds = verifier.get_active_predicates(next_state, "snake")
        ns_hv = engine.episodic_memory.create_situation_hv(next_state, "snake", ns_preds)
        
        # Since pred_ns_hv_bits is a float array, we should ideally convert it back to HV 
        # or compare at bit level.
        # Let's compare similarity using bit-vector correlation.
        ns_bits = engine.world_model.hv_to_numpy(ns_hv)
        
        # Similarity = (pred_ns_hv_bits > 0.5) == ns_bits
        # But neural nets might output skewed values. Simple dot product or correlation:
        correlation = np.dot(pred_ns_hv_bits, ns_bits) / (np.linalg.norm(pred_ns_hv_bits) * np.linalg.norm(ns_bits) + 1e-9)
        print(f"[TEST] State Prediction Correlation: {correlation:.4f}")
        self.assertGreater(correlation, 0.5)

    def test_imagination_interface(self):
        """Test the high-level imagination interface in WorldModel."""
        wm = WorldModel(hv_dim=10240)
        hv1 = hypervec_rs.HyperVector(100)
        hv2 = hypervec_rs.HyperVector(200)
        
        pred_hv, pred_r = wm.imagine(hv1, hv2)
        
        self.assertEqual(pred_hv.shape, (10240,))
        self.assertIsInstance(pred_r, float)

if __name__ == '__main__':
    unittest.main()
