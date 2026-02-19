import unittest
from python.core.reasoning.cognitive_engine import CognitiveEngine
from python.core.reasoning.causal_reasoning import create_snake_causal_graph
from python.core.integration.config import NSCKConfig

class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        self.config = NSCKConfig()
        self.engine = CognitiveEngine(self.config)
        # Register snake domain with its pre-built causal graph
        self.engine.register_task("snake", causal_graph=create_snake_causal_graph())
        
    def test_counterfactual_logic(self):
        """Verify CausalReasoner can simulate alternative outcomes."""
        task = "snake"
        state = {"head": (5, 5), "food": (5, 4)} # Food is UP
        
        reasoner = self.engine.causal_reasoners[task]
        
        cf = reasoner.simulate_counterfactual(state, "ACTION_UP", "ACTION_DOWN", task)
        
        print("\n[TEST] Counterfactual (UP vs DOWN):")
        print(f"  Added: {cf['diff_added']}")
        print(f"  Removed: {cf['diff_removed']}")
        
        # In snake, UP and DOWN lead to different directions
        self.assertNotEqual(cf['action_taken'], cf['hypothetical_action'])
        self.assertIn("HEAD_MOVES_DOWN", cf['predicted_outcome'])

    def test_contrastive_explanation(self):
        """Verify ExplanationGenerator produces contrastive text."""
        task = "snake"
        cf = {
            "action_taken": "ACTION_UP",
            "hypothetical_action": "ACTION_DOWN",
            "predicted_outcome": ["ACTION_DOWN", "DEATH"],
            "diff_added": ["DEATH"],
            "diff_removed": ["SUCCESS"]
        }
        
        explanation = self.engine.explainer.explain_contrastive(
            "ACTION_UP", "ACTION_DOWN", cf, task
        )
        
        print("\n[TEST] Contrastive Explanation:")
        print(f"  Summary: {explanation.summary}")
        print(f"  Details: {explanation.details}")
        
        self.assertIn("instead of move down", explanation.summary)
        self.assertIn("avoid death", explanation.summary)
        self.assertIn("Taken: move up", explanation.details)

if __name__ == "__main__":
    unittest.main()
