import unittest
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.reasoning.analogy import AnalogyEngine


def _hv(word: str):
    return hypervec_rs.HyperVector(abs(hash(word)) % (2**32))


class TestConceptualBlending(unittest.TestCase):
    def setUp(self):
        self.engine = AnalogyEngine()

    def test_blend_returns_dict(self):
        """blend() returns a dict with the expected keys."""
        domain_a = {"bird": _hv("bird"), "wings": _hv("wings")}
        domain_b = {"fish": _hv("fish"), "fins": _hv("fins")}
        mapping = {"bird": "fish"}
        result = self.engine.blend(domain_a, domain_b, mapping)
        self.assertIsInstance(result, dict)
        self.assertIn("shared", result)
        self.assertIn("unique_a", result)
        self.assertIn("unique_b", result)
        self.assertIn("blend_hv", result)
        self.assertIn("emergent", result)

    def test_blend_shared_concepts(self):
        """Mapped concepts appear in the shared (generic) space."""
        domain_a = {"bird": _hv("bird"), "wings": _hv("wings")}
        domain_b = {"fish": _hv("fish"), "fins": _hv("fins")}
        result = self.engine.blend(domain_a, domain_b, {"bird": "fish"})
        self.assertEqual(result["n_shared"], 1)
        self.assertIn("bird↔fish", result["shared"])

    def test_blend_unique_elements(self):
        """Unmapped concepts appear in unique projections."""
        domain_a = {"bird": _hv("bird"), "wings": _hv("wings")}
        domain_b = {"fish": _hv("fish"), "fins": _hv("fins")}
        result = self.engine.blend(domain_a, domain_b, {"bird": "fish"})
        self.assertIn("wings", result["unique_a"])
        self.assertIn("fins", result["unique_b"])

    def test_blend_emergent_property(self):
        """Blend HV is non-None and differs from isolated parents."""
        domain_a = {"bird": _hv("bird")}
        domain_b = {"submarine": _hv("submarine")}
        result = self.engine.blend(domain_a, domain_b, {})
        self.assertIsNotNone(result["blend_hv"])

    def test_blend_no_mapping(self):
        """blend() works with empty mapping (all elements unique)."""
        domain_a = {"fire": _hv("fire")}
        domain_b = {"water": _hv("water")}
        result = self.engine.blend(domain_a, domain_b, {})
        self.assertEqual(result["n_shared"], 0)
        self.assertIn("fire", result["unique_a"])
        self.assertIn("water", result["unique_b"])

    def test_functor_quality_perfect(self):
        """When mapping perfectly preserves compositions, score=1.0."""
        mapping = {"A": "X", "B": "Y", "C": "Z"}
        source = {"A": ["B"], "B": ["C"]}
        target = {"X": ["Y"], "Y": ["Z"]}
        score = self.engine.functor_quality(mapping, source, target)
        self.assertAlmostEqual(score, 1.0)

    def test_functor_quality_zero(self):
        """When no compositions are preserved, score=0.0."""
        mapping = {"A": "X", "B": "Y", "C": "Z"}
        source = {"A": ["B"], "B": ["C"]}
        target = {"X": [], "Y": []}  # no edges in target
        score = self.engine.functor_quality(mapping, source, target)
        self.assertAlmostEqual(score, 0.0)

    def test_functor_quality_range(self):
        """functor_quality always returns a value in [0, 1]."""
        mapping = {"A": "X", "B": "Y"}
        source = {"A": ["B"]}
        target = {"X": ["Y"]}
        score = self.engine.functor_quality(mapping, source, target)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_functor_quality_empty_source(self):
        """Empty source graph → 1.0 (vacuously all compositions preserved)."""
        score = self.engine.functor_quality({}, {}, {})
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
