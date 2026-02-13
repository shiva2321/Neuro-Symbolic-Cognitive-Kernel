"""
Interventional Learning Tests

Note: The hypothesis-biased exploration test was removed because
CausalDiscovery uses count_c/count_e/count_ce data structures (not .stats),
and interventional bias is not yet wired into CognitiveEngine.decide().
When this feature is implemented, tests should be added back using the
actual CausalDiscovery API (observe, get_hypotheses, delta_p).
"""
import unittest
from python.core.reasoning.causal_reasoning import CausalDiscovery

class TestInterventionalLearning(unittest.TestCase):
    def test_hypothesis_generation(self):
        """Verify CausalDiscovery generates hypotheses from observations."""
        cd = CausalDiscovery()
        
        # Feed mixed observations: ACTION_UP → SUCCESS sometimes, not always
        # We need delta_p between 0.1 and 0.5 (hypothesis range)
        for _ in range(3):
            cd.observe("snake", ["ACTION_UP"], ["SUCCESS"])
        for _ in range(3):
            cd.observe("snake", ["ACTION_UP"], ["FAILURE"])  # UP doesn't always work
        for _ in range(4):
            cd.observe("snake", ["ACTION_DOWN"], ["FAILURE"])
        
        # Get hypotheses: links with delta_p in [0.1, 0.5] and evidence >= 2
        hypotheses = cd.get_hypotheses("snake")
        
        print(f"\n[TEST] Hypotheses: {hypotheses}")
        
        # Verify at least one hypothesis was generated
        self.assertTrue(len(hypotheses) > 0, "Should have at least one hypothesis")
        
        # Check delta_p for ACTION_UP → SUCCESS
        dp, evidence = cd._calculate_delta_p("snake", "ACTION_UP", "SUCCESS")
        print(f"[TEST] delta_p(ACTION_UP, SUCCESS): {dp:.2f}, evidence: {evidence}")
        
        # delta_p should be positive (ACTION_UP correlates with SUCCESS)
        self.assertGreater(dp, 0.0)

if __name__ == "__main__":
    unittest.main()
