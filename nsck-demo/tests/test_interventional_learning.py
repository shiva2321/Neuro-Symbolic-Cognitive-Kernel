import unittest
from cognitive_engine import CognitiveEngine
from config import NSCKConfig
import torch

class TestInterventionalLearning(unittest.TestCase):
    def setUp(self):
        self.config = NSCKConfig()
        self.engine = CognitiveEngine(self.config)
        self.task = "snake"
        
    def test_hypothesis_biased_exploration(self):
        """Verify that the engine biases exploration towards testable hypotheses."""
        # 1. Seed a 'weak link' hypothesis
        # We need to simulate some observations of ACTION_UP leading to SUCCESS, 
        # but not enough to reach full confidence (min_evidence=10, min_confidence=0.6)
        
        # stats[context][(cause, effect)] = { 'c_e': 0, 'c_ne': 0, 'nc_e': 0, 'nc_ne': 0 }
        # Let's say ACTION_UP leads to SUCCESS 2 times out of 3.
        # Delta-P = P(S|UP) - P(S|~UP) = 0.66 - 0.0 = 0.66
        # But evidence (3) < 10, so it's a hypothesis.
        
        context = "snake"
        stats_key = ("ACTION_UP", "SUCCESS")
        self.engine.causal_discovery.stats[context][stats_key] = {
            'c_e': 2,
            'c_ne': 1,
            'nc_e': 0,
            'nc_ne': 10
        }
        
        # Verify it shows up as a hypothesis
        hypotheses = self.engine.causal_discovery.get_hypotheses(context)
        self.assertIn(("ACTION_UP", "SUCCESS"), hypotheses)
        
        # 2. Force the engine into exploration mode
        # We can do this by setting confidence very low or mocking should_explore.
        # But let's just use the current state and a large number of trials.
        
        state = {
            "head": (5, 5),
            "food": (8, 8), # Food is far away, default might be RIGHT
            "body": []
        }
        
        # SNN result with low confidence
        metacog = {"confidence": 0.0, "action": "ACTION_STAY"}
        
        up_count = 0
        trials = 200
        
        # Bypass planning to force exploration
        self.engine._try_spatial_planning = lambda s, t: None
        
        print(f"\n[TEST] Running {trials} trials of interventional exploration (Planning Bypassed)...")
        for _ in range(trials):
            # We need to reset the plan if any, to force exploration
            self.engine.active_plan = []
            
            decision = self.engine.decide(state, context, metacog)
            if decision.chosen_action == "ACTION_UP":
                up_count += 1
                
        print(f"[TEST] ACTION_UP chosen {up_count}/{trials} times.")
        
        # Significant bias: Random exploration with 4 actions would be ~25%.
        # With interventional boost, it should be much higher.
        self.assertGreater(up_count, trials * 0.45, f"Expected ACTION_UP to be biased (>45%), but got {up_count}")

if __name__ == "__main__":
    unittest.main()
