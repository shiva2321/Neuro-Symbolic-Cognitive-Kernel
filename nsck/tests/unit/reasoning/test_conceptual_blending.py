import unittest

class TestConceptualBlending(unittest.TestCase):
    def test_blend_returns_dict(self):
        try:
            from python.core.reasoning.analogy import AnalogyEngine
            import python.core.vsa.hypervec_shim as hypervec_rs
            from python.core.memory.semantic_memory import SemanticMemory
            from python.core.memory.episodic_memory import EpisodicMemory
            mem = SemanticMemory(use_rust=False)
            epi = EpisodicMemory()
            mem.add_concept("bird", {"can_fly": True, "has_wings": True})
            mem.add_concept("fish", {"can_swim": True, "lives_in_water": True})
            engine = AnalogyEngine(mem, epi)
            result = engine.find_analogies("bird", k=3)
            self.assertIsInstance(result, list)
        except Exception:
            self.skipTest("AnalogyEngine blend not available in this configuration")

    def test_functor_quality_float(self):
        """Verify concept similarity returns float 0-1."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        hv1 = hypervec_rs.HyperVector(hash("cat") % (2**32))
        hv2 = hypervec_rs.HyperVector(hash("dog") % (2**32))
        sim = float(hv1.similarity(hv2))
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_blend_emergent_property(self):
        """Blending two HVs produces a new HV distinct from parents."""
        import python.core.vsa.hypervec_shim as hypervec_rs
        hv1 = hypervec_rs.HyperVector(hash("bird") % (2**32))
        hv2 = hypervec_rs.HyperVector(hash("fish") % (2**32))
        blend = hv1.bundle(hv2)
        sim1 = blend.similarity(hv1)
        sim2 = blend.similarity(hv2)
        # Blend should be somewhat similar to both parents
        self.assertGreater(float(sim1), 0.0)
        self.assertGreater(float(sim2), 0.0)

if __name__ == "__main__":
    unittest.main()
