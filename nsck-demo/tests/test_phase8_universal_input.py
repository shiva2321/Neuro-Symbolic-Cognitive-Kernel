"""
Phase 8 Verification: Universal Input Layer
============================================
Tests that scalars, categories, dicts, and lists are correctly grounded
into 10 240-bit hypervectors with the expected similarity properties.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import unittest
import numpy as np
import hypervec_shim as hv
from universal_input import UniversalInput


class TestScalarGrounding(unittest.TestCase):
    """Verify the Scalar Similarity Test."""

    def setUp(self):
        self.ui = UniversalInput()

    def test_nearby_values_high_similarity(self):
        """ground(0.84) should be more similar to ground(0.85) than ground(0.10)."""
        h84 = self.ui.ground_scalar(0.84, 0.0, 1.0)
        h85 = self.ui.ground_scalar(0.85, 0.0, 1.0)
        h10 = self.ui.ground_scalar(0.10, 0.0, 1.0)

        sim_close = h84.similarity(h85)
        sim_far = h84.similarity(h10)

        self.assertGreater(
            sim_close, sim_far,
            f"Nearby scalars should be MORE similar: "
            f"sim(0.84,0.85)={sim_close:.4f} vs sim(0.84,0.10)={sim_far:.4f}"
        )

    def test_identical_values_high_similarity(self):
        """Grounding the same value twice should produce similar vectors."""
        h1 = self.ui.ground_scalar(0.5, 0.0, 1.0)
        h2 = self.ui.ground_scalar(0.5, 0.0, 1.0)
        self.assertGreater(h1.similarity(h2), 0.8)

    def test_extreme_values_low_similarity(self):
        """0.0 and 1.0 should have relatively low similarity."""
        h0 = self.ui.ground_scalar(0.0, 0.0, 1.0)
        h1 = self.ui.ground_scalar(1.0, 0.0, 1.0)
        sim = h0.similarity(h1)
        self.assertLess(sim, 0.7,
                        f"Extreme values should have low similarity; got {sim:.4f}")


class TestCategoricalGrounding(unittest.TestCase):
    """Verify categorical codebook properties."""

    def setUp(self):
        self.ui = UniversalInput()

    def test_deterministic(self):
        """Same label in same domain → identical HV."""
        h1 = self.ui.ground_category("error", "status")
        h2 = self.ui.ground_category("error", "status")
        self.assertGreater(h1.similarity(h2), 0.99)

    def test_different_labels_quasi_orthogonal(self):
        """Different labels should be quasi-orthogonal (~0.5 similarity)."""
        h_err = self.ui.ground_category("error", "status")
        h_ok = self.ui.ground_category("ok", "status")
        sim = h_err.similarity(h_ok)
        self.assertAlmostEqual(sim, 0.5, delta=0.05,
                               msg=f"Expected ~0.5, got {sim:.4f}")

    def test_domain_namespacing(self):
        """Same label in different domains should produce different HVs."""
        h1 = self.ui.ground_category("active", "user")
        h2 = self.ui.ground_category("active", "sensor")
        sim = h1.similarity(h2)
        self.assertAlmostEqual(sim, 0.5, delta=0.05,
                               msg=f"Domain namespacing failed; sim={sim:.4f}")

    def test_lru_eviction(self):
        """Codebook should respect max size via LRU eviction."""
        ui = UniversalInput(max_codebook=10)
        for i in range(20):
            ui.ground_category(f"label_{i}", "test")
        self.assertLessEqual(len(ui._codebook), 10)


class TestDictGrounding(unittest.TestCase):
    """Verify recursive role-filler binding for dictionaries."""

    def setUp(self):
        self.ui = UniversalInput()

    def test_role_filler_recovery(self):
        """Unbinding with the role key should recover some similarity to the filler."""
        data = {"status": "error"}
        composite = self.ui.ground_dict(data, "net")

        role_hv = self.ui._get_role_hv("status", "net")
        filler_hv = self.ui.ground_category("error", "net.status")

        # Unbind: composite ⊕ role → should be similar to filler
        recovered = composite.xor(role_hv)
        sim = recovered.similarity(filler_hv)
        self.assertGreater(sim, 0.65,
                           f"Role-filler recovery failed; sim={sim:.4f}")

    def test_different_dicts_differ(self):
        """Different dicts should produce different HVs."""
        d1 = {"status": "error", "ip": "10.0.0.1"}
        d2 = {"status": "ok", "ip": "10.0.0.2"}
        h1 = self.ui.ground_dict(d1, "net")
        h2 = self.ui.ground_dict(d2, "net")
        sim = h1.similarity(h2)
        self.assertLess(sim, 0.7, f"Different dicts too similar; sim={sim:.4f}")


class TestSequenceGrounding(unittest.TestCase):
    """Verify permutation-based sequence encoding."""

    def setUp(self):
        self.ui = UniversalInput()

    def test_order_matters(self):
        """[a, b, c] should differ from [b, a, c]."""
        seq1 = self.ui.ground(["alpha", "beta", "gamma"], domain="test")
        seq2 = self.ui.ground(["beta", "alpha", "gamma"], domain="test")
        sim = seq1.similarity(seq2)
        self.assertLess(sim, 0.7, f"Order should matter; sim={sim:.4f}")

    def test_same_sequence_deterministic(self):
        """Same sequence grounded twice should be identical."""
        s1 = self.ui.ground(["x", "y"], domain="test")
        s2 = self.ui.ground(["x", "y"], domain="test")
        self.assertGreater(s1.similarity(s2), 0.8)


class TestAutoTypeDetection(unittest.TestCase):
    """Verify the auto-dispatch in ground()."""

    def setUp(self):
        self.ui = UniversalInput()

    def test_float_scalar(self):
        """Floats should be grounded as scalars."""
        result = self.ui.ground(0.5, domain="test")
        self.assertIsNotNone(result)

    def test_int_scalar(self):
        """Integers should be grounded as scalars."""
        result = self.ui.ground(42, domain="test", min_val=0, max_val=100)
        self.assertIsNotNone(result)

    def test_string_category(self):
        """Strings should be grounded as categories."""
        result = self.ui.ground("hello", domain="test")
        self.assertIsNotNone(result)

    def test_dict_composite(self):
        """Dicts should be grounded as role-filler composites."""
        result = self.ui.ground({"key": "val"}, domain="test")
        self.assertIsNotNone(result)

    def test_list_sequence(self):
        """Lists should be grounded as sequences."""
        result = self.ui.ground([1, 2, 3], domain="test")
        self.assertIsNotNone(result)

    def test_stats_counting(self):
        """Stats should count each grounding type."""
        ui = UniversalInput()
        ui.ground(0.5)
        ui.ground("label")
        ui.ground({"k": "v"})
        ui.ground([1, 2])
        stats = ui.get_stats()
        self.assertGreaterEqual(stats["scalars_grounded"], 1)
        self.assertGreaterEqual(stats["categories_grounded"], 1)
        self.assertGreaterEqual(stats["dicts_grounded"], 1)
        self.assertGreaterEqual(stats["lists_grounded"], 1)


if __name__ == "__main__":
    unittest.main()
