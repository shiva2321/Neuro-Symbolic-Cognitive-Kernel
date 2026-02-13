"""
Tests for VSA Cleanup Memory system
Verifies denoising and associative memory functionality
"""
import pytest
import numpy as np
from hypervec_py import HyperVector, CleanupMemory, bundle_with_cleanup, unbind_with_cleanup, DIMENSION


class TestCleanupMemory:
    """Test suite for CleanupMemory associative memory."""
    
    def test_initialization(self):
        """CleanupMemory should initialize with correct defaults."""
        cleanup = CleanupMemory()
        assert len(cleanup.memory) == 0
        assert cleanup.max_size == 10000
        assert cleanup._total_cleanups == 0
        
    def test_register_single_vector(self):
        """Should register a clean atomic vector."""
        cleanup = CleanupMemory()
        hv = HyperVector(seed=42)
        cleanup.register("test_vector", hv)
        
        assert "test_vector" in cleanup.memory
        assert len(cleanup.memory) == 1
        assert np.array_equal(cleanup.memory["test_vector"], hv.bits)
    
    def test_register_duplicate_label(self):
        """Duplicate registration should not overwrite without force flag."""
        cleanup = CleanupMemory()
        hv1 = HyperVector(seed=1)
        hv2 = HyperVector(seed=2)
        
        cleanup.register("label", hv1)
        cleanup.register("label", hv2)  # Should not overwrite
        
        assert np.array_equal(cleanup.memory["label"], hv1.bits)
        
        cleanup.register("label", hv2, force=True)  # Should overwrite
        assert np.array_equal(cleanup.memory["label"], hv2.bits)
    
    def test_cleanup_exact_match(self):
        """Cleanup should return exact vector when similarity is 1.0."""
        cleanup = CleanupMemory()
        original = HyperVector(seed=100)
        cleanup.register("original", original)
        
        # Query with exact same vector
        clean_hv, label = cleanup.cleanup(original, threshold=0.9)
        
        assert label == "original"
        assert clean_hv is not None
        assert clean_hv.similarity(original) == 1.0
        assert cleanup._successful_cleanups == 1
    
    def test_cleanup_noisy_vector(self):
        """Cleanup should snap noisy vector to nearest clean vector."""
        cleanup = CleanupMemory()
        original = HyperVector(seed=200)
        cleanup.register("clean", original)
        
        # Create noisy version by flipping ~1% of bits
        noisy_bits = original.bits.copy()
        flip_indices = np.random.choice(DIMENSION, size=int(DIMENSION * 0.01), replace=False)
        noisy_bits[flip_indices] = 1 - noisy_bits[flip_indices]
        noisy_hv = HyperVector.from_bits(noisy_bits)
        
        # Should still match with reasonable threshold
        clean_hv, label = cleanup.cleanup(noisy_hv, threshold=0.95)
        
        assert label == "clean"
        assert clean_hv is not None
        assert clean_hv.similarity(original) == 1.0  # Returns pristine version
        assert noisy_hv.similarity(original) > 0.95  # Noisy was close
    
    def test_cleanup_no_match_below_threshold(self):
        """Cleanup should return None if similarity below threshold."""
        cleanup = CleanupMemory()
        hv1 = HyperVector(seed=1)
        hv2 = HyperVector(seed=2)  # Random, ~0.5 similarity
        
        cleanup.register("stored", hv1)
        
        # Query with very different vector and high threshold
        clean_hv, label = cleanup.cleanup(hv2, threshold=0.99)
        
        assert label is None
        assert clean_hv is None
    
    def test_cleanup_multiple_candidates(self):
        """Cleanup should return best match among multiple candidates."""
        cleanup = CleanupMemory()
        
        # Register 3 vectors
        hv_a = HyperVector(seed=10)
        hv_b = HyperVector(seed=20)
        hv_c = HyperVector(seed=30)
        
        cleanup.register("A", hv_a)
        cleanup.register("B", hv_b)
        cleanup.register("C", hv_c)
        
        # Create noisy version of hv_b
        noisy_bits = hv_b.bits.copy()
        flip_indices = np.random.choice(DIMENSION, size=100, replace=False)
        noisy_bits[flip_indices] = 1 - noisy_bits[flip_indices]
        noisy_b = HyperVector.from_bits(noisy_bits)
        
        # Should match to B (closest)
        clean_hv, label = cleanup.cleanup(noisy_b, threshold=0.4)
        
        assert label == "B"
        assert clean_hv.similarity(hv_b) == 1.0
    
    def test_cleanup_or_keep(self):
        """cleanup_or_keep should return original if no match."""
        cleanup = CleanupMemory()
        hv_stored = HyperVector(seed=1)
        hv_query = HyperVector(seed=999)
        
        cleanup.register("stored", hv_stored)
        
        # No match expected
        result = cleanup.cleanup_or_keep(hv_query, threshold=0.99)
        
        # Should return original noisy vector
        assert result.similarity(hv_query) == 1.0
    
    def test_batch_register(self):
        """Should register multiple vectors at once."""
        cleanup = CleanupMemory()
        vectors = {
            "vec1": HyperVector(seed=1),
            "vec2": HyperVector(seed=2),
            "vec3": HyperVector(seed=3)
        }
        
        cleanup.batch_register(vectors)
        
        assert len(cleanup.memory) == 3
        assert "vec1" in cleanup.memory
        assert "vec2" in cleanup.memory
        assert "vec3" in cleanup.memory
    
    def test_lru_eviction(self):
        """Should evict least recently used when max_size reached."""
        cleanup = CleanupMemory(max_size=3)
        
        # Fill to capacity
        cleanup.register("A", HyperVector(seed=1))
        cleanup.register("B", HyperVector(seed=2))
        cleanup.register("C", HyperVector(seed=3))
        
        # Access A and B to increase their counts
        cleanup.cleanup(HyperVector.from_bits(cleanup.memory["A"]), threshold=0.9)
        cleanup.cleanup(HyperVector.from_bits(cleanup.memory["B"]), threshold=0.9)
        
        # Add D - should evict C (least accessed)
        cleanup.register("D", HyperVector(seed=4))
        
        assert "A" in cleanup.memory
        assert "B" in cleanup.memory
        assert "C" not in cleanup.memory
        assert "D" in cleanup.memory
    
    def test_get_stats(self):
        """Should return accurate statistics."""
        cleanup = CleanupMemory(max_size=100)
        hv = HyperVector(seed=42)
        cleanup.register("test", hv)
        
        cleanup.cleanup(hv, threshold=0.9)  # Successful
        cleanup.cleanup(HyperVector(seed=999), threshold=0.99)  # Failed
        
        stats = cleanup.get_stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == 100
        assert stats['capacity_used'] == 0.01
        assert stats['total_cleanups'] == 2
        assert stats['successful_cleanups'] == 1
        assert stats['success_rate'] == 0.5
        assert stats['most_accessed'] == 'test'
    
    def test_clear(self):
        """Should clear all memory and reset counters."""
        cleanup = CleanupMemory()
        cleanup.register("test", HyperVector(seed=1))
        cleanup.cleanup(HyperVector(seed=1), threshold=0.5)
        
        cleanup.clear()
        
        assert len(cleanup.memory) == 0
        assert cleanup._total_cleanups == 0
        assert cleanup._successful_cleanups == 0


class TestUtilityFunctions:
    """Test utility functions that use cleanup memory."""
    
    def test_bundle_with_cleanup(self):
        """Should bundle vectors and optionally clean result."""
        cleanup = CleanupMemory()
        
        # Register expected result
        hv_a = HyperVector(seed=10)
        hv_b = HyperVector(seed=20)
        expected_bundle = hv_a.bundle(hv_b)
        cleanup.register("expected", expected_bundle)
        
        # Bundle with cleanup
        vectors = [hv_a, hv_b]
        result = bundle_with_cleanup(vectors, cleanup_mem=cleanup, threshold=0.5)
        
        # Should match expected
        assert result.similarity(expected_bundle) >= 0.99
    
    def test_bundle_without_cleanup(self):
        """Should bundle vectors without cleanup if not provided."""
        hv_a = HyperVector(seed=1)
        hv_b = HyperVector(seed=2)
        
        result = bundle_with_cleanup([hv_a, hv_b], cleanup_mem=None)
        expected = hv_a.bundle(hv_b)
        
        # Bundle uses random tiebreak, so results may differ
        # But should be similar (both are valid bundles of a+b)
        assert result.similarity(expected) > 0.6  # Reasonable similarity
    
    def test_unbind_with_cleanup(self):
        """Should unbind and cleanup result."""
        cleanup = CleanupMemory()
        
        # Bind two vectors
        hv_a = HyperVector(seed=100)
        hv_b = HyperVector(seed=200)
        bound = hv_a.xor(hv_b)
        
        # Register expected result (hv_a)
        cleanup.register("original", hv_a)
        
        # Unbind with cleanup
        result = unbind_with_cleanup(bound, hv_b, cleanup_mem=cleanup, threshold=0.5)
        
        # Should recover hv_a
        assert result.similarity(hv_a) == 1.0
    
    def test_bind_unbind_with_noise(self):
        """Complex operation: bind → add noise → unbind → cleanup."""
        cleanup = CleanupMemory()
        
        # Original vectors
        message = HyperVector(seed=1111)
        key = HyperVector(seed=2222)
        
        cleanup.register("message", message)
        cleanup.register("key", key)
        
        # Encrypt
        encrypted = message.xor(key)
        
        # Add noise (flip 50 bits)
        noisy_bits = encrypted.bits.copy()
        flip_indices = np.random.choice(DIMENSION, size=50, replace=False)
        noisy_bits[flip_indices] = 1 - noisy_bits[flip_indices]
        noisy_encrypted = HyperVector.from_bits(noisy_bits)
        
        # Decrypt with cleanup
        decrypted = unbind_with_cleanup(noisy_encrypted, key, cleanup_mem=cleanup, threshold=0.5)
        
        # Should recover original message (cleanup snaps back)
        assert decrypted.similarity(message) >= 0.99


class TestIntegrationWithBrainStore:
    """Test integration between CleanupMemory and persistence layer."""
    
    def test_load_from_store_mock(self):
        """Should load concepts from BrainStore."""
        # Mock BrainStore
        class MockStore:
            class MockConcept:
                def __init__(self, name, vector):
                    self.name = name
                    self.vector = vector
            
            def load_concepts(self):
                hv1 = HyperVector(seed=1)
                hv2 = HyperVector(seed=2)
                return [
                    self.MockConcept("concept1", hv1.bits),
                    self.MockConcept("concept2", hv2.bits)
                ]
        
        cleanup = CleanupMemory()
        store = MockStore()
        cleanup.load_from_store(store)
        
        assert len(cleanup.memory) == 2
        assert "concept1" in cleanup.memory
        assert "concept2" in cleanup.memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
