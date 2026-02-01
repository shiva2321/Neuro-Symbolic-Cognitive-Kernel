"""
Verification test for Phase 5.2: Mental Simulation
"""
import unittest
import numpy as np
from cognitive_engine import CognitiveEngine
import hypervec_shim as hypervec_rs

class TestImagination(unittest.TestCase):
    def test_plan_veto(self):
        """Test that a dangerous plan is vetoed by imagination."""
        engine = CognitiveEngine()
        
        # 1. Setup a "Death" transition in the world model
        # Action UP in state_A leads to reward -1.0
        state_A = {"head": (5, 5), "food": (0, 0)} # Far from food
        action = "ACTION_UP"
        next_state = {"head": (5, 4), "food": (0, 0)} # Moved up
        reward = -1.0 # WALL!
        
        print("\n[TEST] Training World Model that UP is dangerous...")
        # Get HVs for training
        verifier = engine.verifiers["snake"]
        s_preds = verifier.get_active_predicates(state_A, context="snake")
        s_hv = engine.episodic_memory.create_situation_hv(state_A, "snake", s_preds)
        
        ns_preds = verifier.get_active_predicates(next_state, context="snake")
        ns_hv = engine.episodic_memory.create_situation_hv(next_state, "snake", ns_preds)
        
        from symbol_grounding import GLOBAL_PRIMITIVES_MAP
        seed = GLOBAL_PRIMITIVES_MAP.get(action, 0)
        action_hv = hypervec_rs.HyperVector(seed)
        
        for i in range(500):
            engine.world_model.update(s_hv, action_hv, ns_hv, reward)
            
        # 2. Inject a plan that includes ACTION_UP
        engine.active_plan = ["ACTION_UP", "ACTION_UP"]
        
        # 3. Decide and observe veto
        print("[TEST] Deciding with dangerous plan...")
        # We need to make sure we don't explore instead of checking the plan
        engine.curiosity.min_exploration_prob = 0.0 
        
        decision = engine.decide(state_A, "snake")
        
        # The plan should be cleared (vetoed)
        self.assertEqual(engine.active_plan, [])
        # The action should NOT be ACTION_UP (it should fall back to rules or stay)
        self.assertNotEqual(decision.chosen_action, "ACTION_UP")
        print(f"[TEST] Veto Successful. Action chosen: {decision.chosen_action}, Imagined Reward: {decision.imagined_reward:.4f}")
        self.assertLess(decision.imagined_reward, -0.3)

    def test_snn_screening(self):
        """Test that dangerous SNN advice is screened by imagination."""
        engine = CognitiveEngine()
        
        # 1. Train world model that RIGHT is dangerous
        state_B = {"head": (1, 1), "food": (9, 9)}
        action = "ACTION_RIGHT"
        reward = -1.0
        
        # Get HVs for training
        verifier = engine.verifiers["snake"]
        s_preds = verifier.get_active_predicates(state_B, context="snake")
        s_hv = engine.episodic_memory.create_situation_hv(state_B, "snake", s_preds)
        
        from symbol_grounding import GLOBAL_PRIMITIVES_MAP
        seed = GLOBAL_PRIMITIVES_MAP.get(action, 0)
        action_hv = hypervec_rs.HyperVector(seed)
        
        for i in range(500):
            # In world model training, RIGHT leads to negative reward
            engine.world_model.update(s_hv, action_hv, s_hv, reward)
            
        # 2. Mock SNN suggesting ACTION_RIGHT
        metacog = {
            "action": "ACTION_RIGHT",
            "confidence": 0.9,
            "layer_used": "neural"
        }
        
        # Disable plan and rules to isolate SNN screening
        engine.active_plan = []
        # Mock spatial planning and curiosity to prevent them from overriding SNN advice
        engine._try_spatial_planning = lambda s, t: None
        
        from curiosity import ExplorationDecision
        engine.curiosity.should_explore = lambda *args, **kwargs: ExplorationDecision(False, 0, 0, "test")
        
        print("[TEST] Screening SNN's dangerous suggestion...")
        decision = engine.decide(state_B, "snake", metacognition_result=metacog)
        
        # Action should be vetoed
        self.assertNotEqual(decision.chosen_action, "ACTION_RIGHT")
        self.assertLess(decision.imagined_reward, -0.3)
        print(f"[TEST] SNN Veto Successful. Action chosen: {decision.chosen_action}, Imagined Reward: {decision.imagined_reward:.4f}")

if __name__ == '__main__':
    unittest.main()
