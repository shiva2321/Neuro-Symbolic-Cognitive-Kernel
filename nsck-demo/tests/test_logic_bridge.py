"""
Test Logic Bridge Integration (Phase 5)
Verifies: Rule Resolution, Precedence, Safety Veto, and Probabilities
"""
import unittest
from python.brain_fusion import FusedBrain, Rule, QueryResult
from python.metacognition import MetacognitiveEngine, inference_to_probs, SafetyGate

class TestLogicBridge(unittest.TestCase):
    
    def setUp(self):
        self.brain = FusedBrain()
        self.meta = MetacognitiveEngine(self.brain)
        
        # Setup Knowledge Base
        # Global Rule: REL_ABOVE -> ACTION_UP (Priority 0)
        self.brain.global_rules.append(Rule(
            condition=frozenset(["REL_ABOVE"]),
            consequence="ACTION_UP",
            strength=1.0,
            priority=0,
            task_tag="global"
        ))
        
        # Add Global Concept
        # (Needed so valid extract_action happens)
        # We don't need actual vectors for Logic Channel test, just mapping presence
        
    def test_rule_resolution_basic(self):
        """Test simple rule resolution."""
        results = self.brain.resolve_rules({"REL_ABOVE"}, "snake")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].action, "ACTION_UP")
        self.assertAlmostEqual(results[0].score, 1.0) # 1.0 strength * 1.0 prio
        
    def test_precedence_task_over_global(self):
        """Test Task rule overrides Global rule for same condition."""
        # Task Rule: REL_ABOVE -> ACTION_DOWN (Priority 0)
        # Should beat Global due to layer_boost
        self.brain.task_rules["snake"].append(Rule(
            condition=frozenset(["REL_ABOVE"]),
            consequence="ACTION_DOWN",
            strength=1.0,
            priority=0,
            task_tag="snake"
        ))
        
        results = self.brain.resolve_rules({"REL_ABOVE"}, "snake")
        
        print("\n--- Precedence Test Results ---")
        for r in results:
            print(f"Action: {r.action}, Score: {r.score}, Layer: {r.layer}")
            
        self.assertEqual(results[0].action, "ACTION_DOWN")
        # Task score: 1.0 * 1.2 (layer_boost) = 1.2
        # Global score: 1.0
        self.assertGreater(results[0].score, results[1].score)
        
    def test_priority_overrides_task(self):
        """Test High Priority Global overrides Normal Task."""
        # Task: DOWN (1.2 score)
        self.brain.task_rules["snake"].append(Rule(
            condition=frozenset(["REL_ABOVE"]),
            consequence="ACTION_DOWN",
            strength=1.0,
            priority=0,
            task_tag="snake"
        ))
        
        # Global Safety Override: UP (Priority 5)
        # Score: 1.0 * (1.0 + 0.5) = 1.5
        self.brain.global_rules.append(Rule(
            condition=frozenset(["REL_ABOVE"]),
            consequence="ACTION_UP",
            strength=1.0,
            priority=5,
            task_tag="global"
        ))
        
        results = self.brain.resolve_rules({"REL_ABOVE"}, "snake")
        self.assertEqual(results[0].action, "ACTION_UP")
        self.assertGreater(results[0].score, 1.2)

    def test_inference_to_probs(self):
        """Test conversion to probability array."""
        # Mock InferenceResult
        from python.metacognition import InferenceResult
        
        mock_result = InferenceResult(
            action="ACTION_LEFT",
            confidence=0.8,
            uncertainty_reason="test",
            alternatives=[],
            escalation=None,
            trace={
                "top_k": [("ACTION_LEFT", 0.8), ("ACTION_RIGHT", 0.2)]
            }
        )
        
        all_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        probs = inference_to_probs(mock_result, all_actions)
        
        print(f"\nProbs: {probs}")
        
        # LEFT should have highest prob
        # Logic: Score[LEFT] = 0.8 + 1.0 (boost) = 1.8
        # Score[RIGHT] = 0.2
        # LEFT > RIGHT
        left_idx = all_actions.index("LEFT")
        right_idx = all_actions.index("RIGHT")
        
        self.assertGreater(probs[left_idx], probs[right_idx])
        self.assertAlmostEqual(sum(probs), 1.0)
        
if __name__ == "__main__":
    unittest.main()
