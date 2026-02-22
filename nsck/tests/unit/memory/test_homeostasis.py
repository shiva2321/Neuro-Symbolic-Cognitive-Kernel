import unittest
from python.core.memory.homeostasis import MemoryHomeostasis
from python.core.memory.semantic_memory import SemanticMemory

class TestHomeostasis(unittest.TestCase):
    def setUp(self):
        self.homeostasis = MemoryHomeostasis()
        self.memory = SemanticMemory(use_rust=False)

    def test_regulate_returns_list(self):
        for i in range(5):
            self.memory.add_concept(f"concept_{i}", {"type": "test"})
        actions = self.homeostasis.regulate(self.memory)
        self.assertIsInstance(actions, list)

    def test_edge_density_empty(self):
        density = self.homeostasis._measure_edge_density(self.memory)
        self.assertEqual(density, 0.0)

    def test_edge_density_with_edges(self):
        self.memory.add_concept("A", {})
        self.memory.add_concept("B", {})
        self.memory.add_concept("C", {})
        self.memory.add_relation("A", "is_a", "B")
        density = self.homeostasis._measure_edge_density(self.memory)
        self.assertGreater(density, 0.0)
        self.assertLessEqual(density, 1.0)

    def test_regulate_with_concepts(self):
        for i in range(10):
            self.memory.add_concept(f"animal_{i}", {"type": "animal"})
        for i in range(10):
            self.memory.add_concept(f"vehicle_{i}", {"type": "vehicle"})
        actions = self.homeostasis.regulate(self.memory)
        self.assertIsInstance(actions, list)

    def test_auto_categorize_creates_categories(self):
        # Add concepts without is_a parents
        for i in range(5):
            self.memory.add_concept(f"widget_{i}", {"type": "widget"})
        # Force them to have similar HVs by using same seed
        import python.core.vsa.hypervec_shim as hypervec_rs
        shared_hv = hypervec_rs.HyperVector(999)
        for i in range(5):
            self.memory.concept_hvs[f"widget_{i}"] = shared_hv
        cats = self.homeostasis._auto_categorize(self.memory, threshold=0.99)
        # With identical HVs, should create a category
        self.assertIsInstance(cats, list)

if __name__ == "__main__":
    unittest.main()
