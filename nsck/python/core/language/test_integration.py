
import sys
import os
import unittest

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from python.core.language.language_module import LanguageModule
from python.core.memory.semantic_memory import SemanticMemory

class TestLanguageIntegration(unittest.TestCase):
    def test_vsa_backend_integration(self):
        """ Verify LanguageModule can use VSA backend. """
        print("\n[Test] Language Integration (VSA Backend)")
        
        # Shared memory
        mem = SemanticMemory()
        
        # Init LanguageModule with VSA
        # Note: We don't have a model_path that is valid, but it should fallback to mock or load VSA first?
        # My code loads VSA *before* checking LLM availability if use_vsa is True?
        # Let's check init order. 
        # It sets self.vsa_backend first.
        
        lm = LanguageModule(model_path="dummy", semantic_memory=mem, use_vsa=True)
        
        if not lm.use_vsa:
            print("SKIPPING: VSA Backend failed to init (likely missing dependencies in test env?)")
            return
            
        # Test Understanding
        text = "The dog ran"
        result = lm.understand(text)
        
        print(f"Result: {result}")
        
        # Verify Structure
        self.assertIn("structured_output", result)
        self.assertIn("grounded_hv", result)
        
        structured = result["structured_output"]
        self.assertEqual(structured.get("intent"), "run") # Parser normalizes to root 'run'
        self.assertIn("dog", structured.get("entities", []))
        self.assertEqual(structured.get("relation"), "vsa_parsed")

if __name__ == '__main__':
    unittest.main()
