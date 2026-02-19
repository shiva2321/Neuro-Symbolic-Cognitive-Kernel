import sys
import os
# Add 'nsck' folder to path so 'import python.core...' works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import unittest
import numpy as np
import python.core.vsa.hypervec_shim as hv
from python.core.vsa.semantic_memory import SemanticMemory
from python.core.vsa.resonator import ResonatorNetwork

class TestResonator(unittest.TestCase):
    def setUp(self):
        # 1. Setup Memories
        self.agents = SemanticMemory()
        self.actions = SemanticMemory()
        
        # 2. Populate with concepts
        self.agent_names = ["dog", "cat", "robot", "human"]
        self.action_names = ["run", "eat", "sleep", "code"]
        
        for name in self.agent_names:
            v = hv.HyperVector(seed=hash(name) % 10000)
            self.agents.add_concept(name, v)
            
        for name in self.action_names:
            v = hv.HyperVector(seed=hash(name) % 10000)
            self.actions.add_concept(name, v)
            
        # 3. Setup Resonator
        self.resonator = ResonatorNetwork(
            codebooks={"agent": self.agents, "action": self.actions},
            verbose=True
        )

    def test_factorization(self):
        # Create Composite: S = Dog * Run
        dog = self.agents.get_concept("dog")
        run = self.actions.get_concept("run")
        
        # Binding (XOR)
        S = dog.xor(run)
        
        # Factorize
        print("\n[Test] Factorizing 'Dog * Run'...")
        factors = self.resonator.factorize(S)
        
        print(f"Result: {factors}")
        
        self.assertEqual(factors["agent"][0], "dog")
        self.assertEqual(factors["action"][0], "run")

if __name__ == '__main__':
    unittest.main()
