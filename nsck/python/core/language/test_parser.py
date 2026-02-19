
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import unittest
import python.core.vsa.hypervec_shim as hv
from python.core.language.parser import LeftCornerParser
from python.core.vsa.semantic_memory import SemanticMemory

class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = LeftCornerParser()
        
        # Setup known vectors for verification
        self.dog = hv.HyperVector(seed=1)
        self.parser.vectors["dog"] = self.dog
        
        self.ran = hv.HyperVector(seed=2)
        self.parser.vectors["ran"] = self.ran
        
        self.the = hv.HyperVector(seed=3)
        self.parser.vectors["the"] = self.the
        
        # Tags/Roles
        self.dt = self.parser.get_vector("DT")
        self.nn = self.parser.get_vector("NN")
        self.vb = self.parser.get_vector("VB")
        self.agent_role = self.parser.get_vector("AGENT")
        self.action_role = self.parser.get_vector("ACTION")

    def test_parse_the_dog_ran(self):
        # Sentence: "The dog ran"
        # Tokens: [("the", "DT"), ("dog", "NN"), ("ran", "VB")]
        tokens = [("the", "DT"), ("dog", "NN"), ("ran", "VB")]
        
        tree = self.parser.parse_sentence(tokens)
        
        # Expected Structure:
        # Subject = (The * DT) + (Dog * NN)
        # Tree = (Subject * AGENT) + ((Ran * VB) * ACTION)
        
        # Let's verify we can retrieve "Dog"
        # Query: Dog =? (Tree * Agent * NN) - (The * DT * Agent * NN) etc?
        # Simpler: Unbind Agent
        
        subject_superposition = tree.xor(self.agent_role)
        
        # Subject Superposition ~= Subject + Noise(Action part)
        # Subject = (The * DT) + (Dog * NN)
        
        # Unbind NN to get Dog
        dog_candidate = subject_superposition.xor(self.nn)
        
        # Dog Candidate ~= Dog + Noise
        sim = dog_candidate.similarity_robust(self.dog)
        print(f"Similarity to Dog: {sim}")
        
        self.assertGreater(sim, 0.55, "Should retrieve Dog from parse tree")
        
        # Verify Action
        action_part = tree.xor(self.action_role)
        # Action Part ~= (Ran * VB)
        ran_candidate = action_part.xor(self.vb)
        
        sim_ran = ran_candidate.similarity_robust(self.ran)
        print(f"Similarity to Ran: {sim_ran}")
        self.assertGreater(sim_ran, 0.55)

if __name__ == '__main__':
    unittest.main()
