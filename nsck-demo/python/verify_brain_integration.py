
import unittest
import sys
import os

# Add path to python folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cognitive_engine import CognitiveEngine
from agency import TILE_FOOD

class TestBrainIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize Engine (Headless)
        self.engine = CognitiveEngine()
        
    def test_active_inference_proposal(self):
        """
        Verify that ActiveAgent proposes actions when hungry.
        """
        print("\n--- Testing Brain Active Inference Integration ---")
        
        # 1. Set State: Hungry
        self.engine.homeostasis.drives["hunger"] = 0.9
        print(f"Set Hunger to 0.9. Expecting High Salience on Active Inference.")
        
        # 2. Mock State
        # Snake at (5,5), Food at (5,4) (UP)
        state_data = {
            "head": (5, 5),
            "food": (5, 4),
            "body": [(5, 5), (5, 6), (5, 7)],
            "visual_input": "mock_tensor" 
        }
        
        # 3. Decide
        # We need to mock 'situation_hv' or just ensure decide can handle defaults if not fully initialized
        # The engine.decide method requires 'visual_input' to generate HV usually.
        # However, for this integration test, we want to see if the ActiveAgent part runs.
        # We might hit errors if other parts (SNN) aren't mocked.
        # Let's try running it and see. The 'state' dict is passed to decide.
        
        # We need to bypass SNN forward pass if possible or provide mock
        # Ideally CognitiveEngine would mock SNN if not loaded.
        
        # Actually, looking at code, 'decide' takes 'state' and 'task_tag'.
        # It calls 'self.snn.forward(state['visual_input'])'
        # We need to mock self.snn
        
        class MockSNN:
            def forward(self, x): return None # Dummy
            def get_concept_hv(self, x): return None
            
        self.engine.snn = MockSNN()
        
        # It also calls 'self.semantic_memory.get_situation_hv'
        # We need to ensure that doesn't crash.
        
        action = self.engine.decide(state_data, "snake")
        print(f"Global Winner Action: {action}")
        
        # 4. Inspection
        # We want to check if ACTIVE_INFERENCE was in the coalitions.
        # CognitiveEngine doesn't publicize coalitions easily, but we can look at logs if we had them hooked.
        # OR we can inspect the 'active_agent' internal state to see if it updated.
        
        # Check if ActiveAgent belief updated
        # We passed food at (5,4)
        food_prob = self.engine.active_agent.belief_map[5, 4, TILE_FOOD]
        print(f"ActiveAgent Belief for Food @ (5,4): {food_prob}")
        self.assertAlmostEqual(food_prob, 1.0, msg="ActiveAgent did not update belief!")
        
        # Check preferences (Hungry means Food should be high value)
        food_pref = self.engine.active_agent.preferences[TILE_FOOD]
        print(f"Food Preference Value: {food_pref}")
        self.assertEqual(food_pref, 20.0, msg="Hunger did not trigger Food Preference!")

if __name__ == "__main__":
    unittest.main()
