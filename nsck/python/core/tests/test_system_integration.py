
import sys
import os
import unittest
import numpy as np

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from python.core.reasoning.cognitive_engine import create_cognitive_engine
from python.core.perception.snn_perception import SNNPerceptionModule
from python.core.language.language_module import LanguageModule
from python.core.reasoning.global_workspace import Coalition

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        """Bootstrap the full engine."""
        print("\n[Setup] Bootstrapping Cognitive Engine...")
        self.engine = create_cognitive_engine()
        
    def test_perception_pipeline(self):
        """Verify SNN can perceive noise and broadcast to Global Workspace."""
        print("\n[Test 1] SNN Perception -> Global Workspace")
        
        if not self.engine.perception:
            self.skipTest("SNN Module not initialized (missing dependencies?)")
            
        # 1. Generate Input
        sensory_input = np.random.rand(64).astype(np.float32)
        
        # 2. Perceive & Decide
        state = self.engine.perceive_and_decide(sensory_input, task_tag="test_env")
        
        # 3. Verify Trace
        print(f"  Decision Trace: {state.trace}")
        
        # Check if SNN coalition was proposed (even if it didn't win)
        # Note: We need access to the coalitions that *competed*.
        # The engine trace stores 'proposals' as a list of source names.
        
        proposals = state.trace.get("proposals", 0) 
        # In current Engine implementation, trace["proposals"] is an integer count if using 'simple' tracing,
        # OR a list if detailed. 
        # Looking at previous output: 'proposals': 3
        # So it is an integer count of how many coalitions competed.
        
        # Verify that we had at least 1 proposal (the SNN one + maybe rules/exploration)
        self.assertGreater(proposals, 0, "Global Workspace should have received proposals")
        
        # Also check 'winner'
        self.assertIn("winner", state.trace)
        print(f"  Winner: {state.trace['winner']}")
        
    def test_language_integration(self):
        """Verify Language Module defaults to VSA and parses correctly."""
        print("\n[Test 2] Language -> VSA Parser")
        
        lm = self.engine.language
        self.assertTrue(lm.use_vsa, "LanguageModule should default to VSA")
        
        text = "The dog runs" # 'run' is in the demo vocab, 'eats' might not be properly mapped yet
        result = lm.understand(text)
        
        structured = result["structured_output"]
        print(f"  Parsed: {structured}")
        
        # Note: VSA might return "run" or "runs" depending on stemming.
        # And usually VSA demo maps "dog" -> "dog", "run" -> "run".
        self.assertIn(structured.get("intent"), ["run", "runs"])
        self.assertIn("dog", structured.get("entities", []))
        self.assertEqual(structured.get("relation"), "vsa_parsed")
        
    def test_end_to_end_dialogue(self):
        """Verify the 'Chat' loop works with VSA."""
        print("\n[Test 3] User Input -> Engine Response")
        
        response = self.engine.process_dialogue("The dog runs")
        print(f"  Agent Response: {response}")
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()
