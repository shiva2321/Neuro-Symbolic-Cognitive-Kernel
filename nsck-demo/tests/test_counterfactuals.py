import unittest
from cognitive_engine import CognitiveEngine
from config import NSCKConfig

class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        self.config = NSCKConfig()
        self.engine = CognitiveEngine(self.config)
        
    def test_counterfactual_logic(self):
        """Verify CausalReasoner can simulate alternative outcomes."""
        task = "snake"
        state = {"head": (5, 5), "food": (5, 4)} # Food is UP
        
        # Action taken: UP
        # Hypothetical: DOWN
        
        # We need the graph to know that DOWN leads to something 'different'
        # Actually, the hardcoded snake graph has ACTION_UP -> REL_ABOVE, etc.
        
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

    def test_integrated_counterfactual(self):
        """Verify CognitiveEngine performs periodic counterfactual analysis."""
        state = {"head": (5, 5), "food": (5, 4)}
        context = "snake"
        
        # Force 10 decisions to trigger periodic CF
        for i in range(1, 11):
            decision = self.engine.decide(state, context)
            if i == 10:
                self.assertIn("counterfactual", decision.trace)
                cf_trace = decision.trace["counterfactual"]
                print(f"\n[TEST] Decision 10 Counterfactual Trace:")
                print(f"  Hypothetical: {cf_trace['hypothetical']}")
                print(f"  Summary: {cf_trace['summary']}")
                self.assertTrue(len(cf_trace['summary']) > 0)

if __name__ == "__main__":
    unittest.main()
