import unittest
from unittest.mock import MagicMock
from cognitive_engine import create_cognitive_engine

class TestMetacognitiveVeto(unittest.TestCase):
    def test_confidence_fusion(self):
        """Verify weighted fusion of SNN and Self-Model confidence."""
        engine = create_cognitive_engine()
        
        # Mock SelfModel to return 0.8 (High)
        engine.self_model.predict_success = MagicMock(return_value=0.8)
        engine._try_spatial_planning = MagicMock(return_value=None) # Ensure no plan interrupts
        
        # SNN returns 0.4 (Low)
        meta_result = {"action": "ACTION_UP", "confidence": 0.4}
        
        # Expected: 0.4*0.4 (0.16) + 0.6*0.8 (0.48) = 0.64
        state_dec = engine.decide({}, "snake", meta_result)
        
        self.assertAlmostEqual(state_dec.confidence, 0.64, places=2)
        
    def test_veto_triggers_fallback(self):
        """Verify that low confidence vetoes SNN action."""
        engine = create_cognitive_engine()
        engine._try_spatial_planning = MagicMock(return_value=None) # Disable planner
        engine.curiosity.should_explore = MagicMock(return_value=type('obj', (object,), {'should_explore': False, 'reason': 'test'})())
        
        # 1. High Confidence Case (No Veto)
        engine.self_model.predict_success = MagicMock(return_value=0.9)
        meta_result = {"action": "ACTION_SNN", "confidence": 0.9} # Fusion = 0.9
        
        dec = engine.decide({}, "snake", meta_result)
        self.assertEqual(dec.chosen_action, "ACTION_SNN", "Should listen to SNN when confident")
        self.assertNotIn("warning", dec.trace)
        
        # 2. Low Confidence Case (Veto!)
        engine.self_model.predict_success = MagicMock(return_value=0.1)
        meta_result = {"action": "ACTION_SNN", "confidence": 0.1} # Fusion = 0.1 (<0.2 threshold)
        
        dec_veto = engine.decide({}, "snake", meta_result)
        
        self.assertNotEqual(dec_veto.chosen_action, "ACTION_SNN", "Should VETO SNN action")
        self.assertIn("veto_confusion", dec_veto.trace)
        self.assertTrue(dec_veto.trace["veto_confusion"])
        self.assertEqual(dec_veto.trace["warning"], "LOW_CONFIDENCE_VETO")
        
        # It should have fallen back to default (ACTION_STAY) or rules
        self.assertIn(dec_veto.trace["mode"], ["learned_rule", "default"])

if __name__ == '__main__':
    unittest.main()
