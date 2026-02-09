import unittest
from unittest.mock import MagicMock
from cognitive_engine import create_cognitive_engine

class TestMetacognitiveVeto(unittest.TestCase):
    def test_confidence_passthrough(self):
        """Verify that metacognition confidence flows into the decision."""
        engine = create_cognitive_engine()
        
        # SNN returns 0.4 confidence
        meta_result = {"action": "ACTION_UP", "confidence": 0.4}
        
        state_dec = engine.decide({}, "snake", meta_result)
        
        # The decide() method sets confidence = metacognition_result["confidence"]
        self.assertAlmostEqual(state_dec.confidence, 0.4, places=2)
        
    def test_low_confidence_falls_back(self):
        """Verify that very low confidence does not prevent the system from acting."""
        engine = create_cognitive_engine()
        
        # Very low confidence — system should still produce a valid action
        meta_result = {"action": "ACTION_SNN", "confidence": 0.1}
        
        dec = engine.decide({}, "snake", meta_result)
        
        # System should produce some action (may or may not follow SNN advice)
        self.assertIsNotNone(dec.chosen_action)
        self.assertTrue(len(dec.chosen_action) > 0)
        
    def test_high_confidence_action_selection(self):
        """Verify that the competition mechanism selects an action even with high confidence."""
        engine = create_cognitive_engine()
        
        # High confidence SNN
        meta_result = {"action": "ACTION_UP", "confidence": 0.9}
        
        dec = engine.decide({}, "snake", meta_result)
        
        # Should get a valid action from the competition
        self.assertIsNotNone(dec.chosen_action)
        self.assertIn(dec.trace["winner"], ["SNN", "PLANNER", "ACTIVE_INFERENCE", "RULES", "EXPLORATION", "DEFAULT"])

if __name__ == '__main__':
    unittest.main()
