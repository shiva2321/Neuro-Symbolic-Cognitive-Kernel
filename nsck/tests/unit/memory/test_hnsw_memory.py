import unittest
from python.core.memory.semantic_memory import SemanticMemory
import python.core.vsa.hypervec_shim as hypervec_rs

class TestHNSWMemory(unittest.TestCase):
    def test_semantic_memory_fallback_no_crash(self):
        """SemanticMemory should work without hnswlib."""
        mem = SemanticMemory(use_rust=False)
        mem.add_concept("cat", {"type": "animal"})
        mem.add_concept("dog", {"type": "animal"})
        query = hypervec_rs.HyperVector(hash("cat") % (2**32))
        results = mem.query(query, k=2)
        self.assertIsInstance(results, list)

    def test_query_returns_concepts(self):
        mem = SemanticMemory(use_rust=False)
        for i in range(5):
            mem.add_concept(f"item_{i}", {"index": i})
        query = hypervec_rs.HyperVector(hash("item_0") % (2**32))
        results = mem.query(query, k=3)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)

    def test_hnswlib_optional_import(self):
        try:
            import hnswlib
            has_hnswlib = True
        except ImportError:
            has_hnswlib = False
        # Either way, SemanticMemory should work
        mem = SemanticMemory(use_rust=False)
        mem.add_concept("test", {})
        self.assertIn("test", mem.concept_hvs)

if __name__ == "__main__":
    unittest.main()
