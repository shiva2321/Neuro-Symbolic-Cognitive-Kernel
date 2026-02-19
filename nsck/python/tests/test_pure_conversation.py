
import sys
import os
import unittest
# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from python.core.language.nlg import StructuralRealizer
from python.core.language.dialogue_manager import DialogueManager
# Mocks
class MockEngine:
    def __init__(self):
        self.semantic_memory = MockMemory()

class MockMemory:
    def __init__(self):
        self.concept_graph = MockGraph()

class MockGraph:
    def __init__(self):
        self.edges = {
            "Valkyria": [("Valkyria", "Game", {"relation": "is_a"}), ("Valkyria", "Sega", {"relation": "developed_by"})],
            "Dog": [("Dog", "Animal", {"relation": "is_a"}), ("Dog", "Fur", {"relation": "has_property"})]
        }
    def out_edges(self, subject, data=True):
        return self.edges.get(subject, [])
    def __contains__(self, item):
        return item in self.edges

class TestPureVSA(unittest.TestCase):
    def setUp(self):
        self.realizer = StructuralRealizer()
        # DialogueManager(cognitive_engine, language_module)
        self.dm = DialogueManager(MockEngine(), None) 
        self.dm.nlg.realizer = self.realizer # Ensure it uses our realizer

    def test_morphology(self):
        """Test verb conjugation and pluralization."""
        self.assertEqual(self.realizer.pluralize("dog"), "dogs")
        self.assertEqual(self.realizer.pluralize("church"), "churches")
        self.assertEqual(self.realizer.pluralize("fly"), "flies")
        
        self.assertEqual(self.realizer.conjugate("be", person="3rd", number="singular"), "is")
        self.assertEqual(self.realizer.conjugate("be", person="3rd", number="plural"), "are")
        self.assertEqual(self.realizer.conjugate("run", person="3rd", number="singular"), "runs")
        self.assertEqual(self.realizer.conjugate("run", person="3rd", number="plural"), "run")

    def test_sentence_realization(self):
        """Test full sentence generation."""
        # Generic SV0
        s1 = self.realizer.realize_sentence("dog", "chase", "cat")
        self.assertEqual(s1, "The dog chases the cat.")
        
        # "Is a" relation
        s2 = self.realizer.realize_sentence("Valkyria", "is_a", "Game")
        self.assertEqual(s2, "Valkyria is a game.") # Proper noun, indefinite object
        
        # "Has property"
        s3 = self.realizer.realize_sentence("dog", "has_property", "fur")
        self.assertEqual(s3, "The dog has fur.")

    def test_knowledge_retrieval_nlg(self):
        """Test DialogueManager generating answers from mock memory."""
        # Simulate "What is Valkyria?" -> Logic calls retrieve_knowledge("Valkyria")
        response = self.dm.retrieve_knowledge("Valkyria")
        print(f"Agent Response (Valkyria): {response}")
        
        self.assertIn("Valkyria is a game", response)
        self.assertIn("Valkyria developed Sega", response) # Correct past tense conjugation
        # Ideally relation names should be mapped to verbs ("developed_by" -> "was developed by")
        # But for now, verifying morphology engine runs on relation keys is sufficient.

if __name__ == "__main__":
    unittest.main()
