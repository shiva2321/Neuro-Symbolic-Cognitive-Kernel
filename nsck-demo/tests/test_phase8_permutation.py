"""
Phase 8 Verification: Permutation Operator
===========================================
Tests the circular bitwise shift (permute/permute_inverse) added to the
HyperVector class in both Rust and Python backends.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import unittest
import hypervec_shim as hv


class TestPermutation(unittest.TestCase):
    """Verify permute() and permute_inverse() on 10 240-bit hypervectors."""

    # --- Core properties ---

    def test_inverse_property(self):
        """permute(n).permute(-n) should reconstruct the original vector."""
        v = hv.HyperVector(42)
        for shift in [1, 7, 64, 65, 100, 5120, 10239]:
            restored = v.permute(shift).permute_inverse(shift)
            sim = v.similarity(restored)
            self.assertGreater(
                sim, 0.99,
                f"Inverse failed for shift={shift}: similarity={sim:.4f}"
            )

    def test_inverse_negative_shifts(self):
        """Negative shifts should also satisfy the inverse property."""
        v = hv.HyperVector(123)
        for shift in [-1, -64, -100, -5000]:
            restored = v.permute(shift).permute(-shift)
            sim = v.similarity(restored)
            self.assertGreater(sim, 0.99, f"Inverse failed for shift={shift}")

    def test_full_rotation_identity(self):
        """Shifting by DIMENSION (10240) should return to the original."""
        v = hv.HyperVector(7)
        rotated = v.permute(10240)
        self.assertGreater(v.similarity(rotated), 0.99)

    def test_zero_shift_identity(self):
        """Shift of 0 should be identity."""
        v = hv.HyperVector(99)
        same = v.permute(0)
        self.assertGreater(v.similarity(same), 0.99)

    # --- Quasi-orthogonality ---

    def test_large_shift_quasi_orthogonal(self):
        """A large shift should produce a quasi-orthogonal vector (~0.5 similarity)."""
        v = hv.HyperVector(55)
        shifted = v.permute(100)
        sim = v.similarity(shifted)
        self.assertAlmostEqual(sim, 0.5, delta=0.03,
                               msg=f"Expected ~0.5, got {sim:.4f}")

    def test_different_shifts_different_results(self):
        """Different shift amounts should produce different vectors."""
        v = hv.HyperVector(77)
        s1 = v.permute(1)
        s2 = v.permute(2)
        sim = s1.similarity(s2)
        self.assertAlmostEqual(sim, 0.5, delta=0.03,
                               msg=f"Shifts 1 vs 2 should differ; sim={sim:.4f}")

    # --- Sequence encoding ---

    def test_sequence_encoding_order_sensitivity(self):
        """[A, B, C] encodes differently from [B, A, C]."""
        a = hv.HyperVector(10)
        b = hv.HyperVector(20)
        c = hv.HyperVector(30)

        # Encode A→B→C:  A ⊕ ρ¹(B) ⊕ ρ²(C)
        seq_abc = a.xor(b.permute(1)).xor(c.permute(2))
        # Encode B→A→C:  B ⊕ ρ¹(A) ⊕ ρ²(C)
        seq_bac = b.xor(a.permute(1)).xor(c.permute(2))

        sim = seq_abc.similarity(seq_bac)
        self.assertLess(sim, 0.6,
                        f"Sequences should differ; sim={sim:.4f}")

    def test_sequence_query_recovery(self):
        """Can recover element at position 1 by unbinding the sequence."""
        a = hv.HyperVector(10)
        b = hv.HyperVector(20)
        c = hv.HyperVector(30)

        seq = a.xor(b.permute(1)).xor(c.permute(2))

        # Query position 1: permute_inverse(seq, 1)  → should be similar to B
        # Actually, to recover B from seq:  ρ⁻¹(seq ⊕ A ⊕ ρ²(C))
        # But simpler: check that ρ⁻¹(seq ⊕ A ⊕ ρ²(C)) ≈ ρ⁻¹(ρ¹(B)) = B
        residual = seq.xor(a).xor(c.permute(2))  # should ≈ ρ¹(B)
        recovered_b = residual.permute_inverse(1)
        sim = recovered_b.similarity(b)
        self.assertGreater(sim, 0.65, f"Recovery failed; sim={sim:.4f}")

    # --- Word-boundary carry correctness ---

    def test_shift_exactly_64(self):
        """Shift of exactly 64 bits (one u64 word) should be exact."""
        v = hv.HyperVector(88)
        shifted = v.permute(64)
        restored = shifted.permute_inverse(64)
        self.assertGreater(v.similarity(restored), 0.99)

    def test_shift_65_cross_boundary(self):
        """Shift of 65 tests the cross-word bit carry logic."""
        v = hv.HyperVector(44)
        shifted = v.permute(65)
        restored = shifted.permute_inverse(65)
        self.assertGreater(v.similarity(restored), 0.99)


if __name__ == "__main__":
    unittest.main()
