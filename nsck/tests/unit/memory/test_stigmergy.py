import unittest
from python.core.memory.semantic_memory import SemanticMemory

class TestStigmergy(unittest.TestCase):
    def setUp(self):
        self.memory = SemanticMemory(use_rust=False)
        self.memory.add_concept("A", {})
        self.memory.add_concept("B", {})
        self.memory.add_concept("C", {})
        self.memory.add_relation("A", "leads_to", "B")
        self.memory.add_relation("B", "leads_to", "C")

    def test_mark_path_increments_stigmergy(self):
        g = self.memory.concept_graph
        # Mark edge A->B
        if g.has_edge("A", "B"):
            edge_data = g["A"]["B"]
            old_stigmergy = edge_data.get("stigmergy", 0.0)
            edge_data["stigmergy"] = old_stigmergy + 1.0
            self.assertGreater(edge_data["stigmergy"], old_stigmergy)

    def test_evaporate_stigmergy(self):
        g = self.memory.concept_graph
        # Set stigmergy, then decay
        if g.has_edge("A", "B"):
            g["A"]["B"]["stigmergy"] = 1.0
            g["A"]["B"]["stigmergy"] *= 0.9
            self.assertLess(g["A"]["B"]["stigmergy"], 1.0)

    def test_spread_activation_basic(self):
        activations = self.memory.spread_activation(["A"], steps=2)
        self.assertIsInstance(activations, dict)
        self.assertIn("A", activations)

    def test_spread_activation_propagates(self):
        activations = self.memory.spread_activation(["A"], steps=3)
        self.assertGreater(len(activations), 0)

if __name__ == "__main__":
    unittest.main()
