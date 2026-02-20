import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any

DIMENSION = 10240

class HyperVectorPy:
    def __init__(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
            
        # Represent as boolean array for simplicity in Python
        # or packed bits (uint8) for memory. Boolean is easier for XOR.
        # 0 or 1
        self.bits = self.rng.integers(0, 2, size=DIMENSION, dtype=np.int8)

    @classmethod
    def from_bits(cls, bits):
        obj = cls.__new__(cls)
        obj.bits = bits
        return obj

    @staticmethod
    def zero():
        obj = HyperVectorPy.__new__(HyperVectorPy)
        obj.bits = np.zeros(DIMENSION, dtype=np.int8)
        return obj

    def xor(self, other):
        # XOR is addition in binary fields
        new_bits = np.bitwise_xor(self.bits, other.bits)
        return HyperVectorPy.from_bits(new_bits)

    def bundle(self, other):
        # Majority rule: same bits are kept; differing bits use a
        # deterministic tie-breaking mask derived from both inputs so
        # the same pair always produces the same output vector.
        a = self.bits
        b = other.bits
        diff = np.bitwise_xor(a, b)
        same = np.bitwise_and(a, b)

        # Seed from XOR-weight of each vector's first 64 bits — fast,
        # collision-resistant enough for VSA usage, and reproducible.
        seed = int(a[:64].sum()) ^ (int(b[:64].sum()) << 14)
        rand_mask = np.random.default_rng(seed & 0x7FFFFFFF).integers(
            0, 2, size=DIMENSION, dtype=np.int8)

        bundle_bits = np.bitwise_or(same, np.bitwise_and(diff, rand_mask))
        return HyperVectorPy.from_bits(bundle_bits)

    def similarity(self, other):
        """
        Hamming-based similarity (legacy, kept for compatibility).
        For noise-robust comparison, use cosine_similarity() instead.
        """
        # Hamming distance
        diff = np.bitwise_xor(self.bits, other.bits)
        hamming_dist = np.sum(diff)
        
        # Similarity = 1 - dist/dim
        return 1.0 - (hamming_dist / DIMENSION)
    
    def cosine_similarity(self, other):
        """
        Cosine similarity on bipolar representation (more robust to noise).
        
        Converts binary {0,1} to bipolar {-1,+1} then computes cosine.
        This is more robust to bit flips than raw Hamming distance.
        
        Returns:
            float in [-1, 1]: 1 = identical, 0 = orthogonal, -1 = opposite
        """
        # Convert to bipolar: 0 -> -1, 1 -> +1
        a_bipolar = (self.bits.astype(np.float32) - 0.5) * 2
        b_bipolar = (other.bits.astype(np.float32) - 0.5) * 2
        
        # Cosine similarity
        dot_product = np.dot(a_bipolar, b_bipolar)
        norm_a = np.linalg.norm(a_bipolar)
        norm_b = np.linalg.norm(b_bipolar)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def similarity_robust(self, other, method='cosine'):
        """
        Noise-robust similarity with configurable method.
        
        Args:
            other: HyperVector to compare against
            method: 'hamming' (legacy) or 'cosine' (recommended)
        
        Returns:
            float in [0, 1]: similarity score
        """
        if method == 'cosine':
            # Cosine returns [-1, 1], normalize to [0, 1]
            cos_sim = self.cosine_similarity(other)
            return (cos_sim + 1.0) / 2.0
        else:
            # Use legacy Hamming
            return self.similarity(other)

    def permute(self, shift):
        """Circular bitwise permutation (rotation) of the hypervector."""
        shift_norm = shift % DIMENSION
        if shift_norm == 0:
            return HyperVectorPy.from_bits(self.bits.copy())
        new_bits = np.roll(self.bits, -shift_norm)
        return HyperVectorPy.from_bits(new_bits)

    def permute_inverse(self, shift):
        """Inverse permutation: equivalent to permute(-shift)."""
        return self.permute(-shift)

    def weighted_bundle(self, other, weight: float, seed=None):
        """
        Produce a vector biased toward *self* (weight→1) or *other* (weight→0).

        Uses majority-vote over ``k`` copies: a high *weight* includes more
        copies of *self*, a low *weight* more copies of *other*.

        Args:
            other: The other HyperVector to blend with.
            weight: Blend weight in [0, 1].  1.0 = all self, 0.0 = all other.
            seed:   Unused; kept for API parity with the Rust shim.

        Returns:
            A new HyperVectorPy blended according to *weight*.
        """
        w = max(0.0, min(1.0, float(weight)))
        k = 7
        self_n = max(1, int(round(w * k)))
        other_n = k - self_n

        out = self
        for _ in range(self_n - 1):
            out = out.bundle(self)
        for _ in range(other_n):
            out = out.bundle(other)
        return out

    def lsh_hash(self, seed: int, n_bits: int) -> int:
        """
        Locality-Sensitive Hash of this hypervector.

        Selects ``n_bits`` random bit positions (seeded deterministically) and
        packs the values at those positions into an integer.  Similar vectors
        share many selected bits, so they tend to land in the same bucket.

        Args:
            seed:   Integer seed that selects the projection (one per LSH table).
            n_bits: Number of bits to project onto (bucket resolution).

        Returns:
            A non-negative integer bucket index.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(DIMENSION, size=int(n_bits), replace=False)
        selected = self.bits[indices]
        result = 0
        for i, b in enumerate(selected):
            if b:
                result |= (1 << i)
        return result

    def __repr__(self):
        return f"<HyperVector dim={DIMENSION} (Python)>"

# Re-export the Python class under the canonical name so that
# ``from hypervec_py import HyperVector`` works everywhere.
HyperVector = HyperVectorPy

class CleanupMemory:
    """
    Associative memory for VSA cleanup/denoising.
    
    As NSCK scales to thousands of concepts, Hamming distance noise accumulates
    during binding/unbinding operations. CleanupMemory maintains a registry of
    known "clean" atomic vectors and provides a cleanup operation to snap noisy
    vectors back to their nearest neighbor.
    
    This prevents:
    - False positives in similarity matching
    - Drift in repeatedly bound/unbound vectors
    - Crosstalk between conceptually distinct vectors
    
    Usage:
        cleanup = CleanupMemory()
        cleanup.register("apple", apple_hv)
        cleanup.register("orange", orange_hv)
        
        # After noisy operations...
        noisy_hv = some_complex_binding_operation()
        clean_hv, label = cleanup.cleanup(noisy_hv, threshold=0.4)
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize cleanup memory.
        
        Args:
            max_size: Maximum number of atomic vectors to store (prevents unbounded growth)
        """
        self.memory: Dict[str, np.ndarray] = {}  # label -> bits
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}  # For LRU eviction
        self._total_cleanups = 0
        self._successful_cleanups = 0
    
    def register(self, label: str, hv: HyperVectorPy, force: bool = False):
        """
        Register a clean atomic vector in the cleanup memory.
        
        Args:
            label: Unique identifier for this vector (e.g., "MOVE_UP", "apple", "fear")
            hv: The hypervector to register as a clean reference
            force: If True, overwrite existing entry with same label
        """
        if label in self.memory and not force:
            # Already registered, just increment access count
            self.access_count[label] = self.access_count.get(label, 0) + 1
            return
        
        # Check capacity
        if len(self.memory) >= self.max_size and label not in self.memory:
            self._evict_lru()
        
        self.memory[label] = hv.bits.copy()
        self.access_count[label] = 1
    
    def _evict_lru(self):
        """Evict least recently used entry to make space."""
        if not self.access_count:
            return
        
        lru_label = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        del self.memory[lru_label]
        del self.access_count[lru_label]
    
    def cleanup(self, noisy_hv: HyperVectorPy, threshold: float = 0.4) -> Tuple[Optional[HyperVectorPy], Optional[str]]:
        """
        Clean up a noisy hypervector by snapping to nearest known vector.
        
        Args:
            noisy_hv: The potentially noisy hypervector to clean
            threshold: Minimum similarity to consider a match (0.0-1.0)
                      Higher = stricter (only very similar vectors match)
                      Lower = more permissive (more cleanup, risk of false matches)
                      Recommended: 0.4-0.6 for typical VSA operations
        
        Returns:
            Tuple of (cleaned_hv, label) if match found above threshold
            Tuple of (None, None) if no match found (vector is too noisy or unknown)
        """
        self._total_cleanups += 1
        
        if not self.memory:
            return None, None
        
        best_label = None
        best_similarity = threshold  # Must exceed this
        
        # Find nearest neighbor in cleanup memory
        for label, clean_bits in self.memory.items():
            # Calculate similarity directly on bits
            diff = np.bitwise_xor(noisy_hv.bits, clean_bits)
            hamming_dist = np.sum(diff)
            similarity = 1.0 - (hamming_dist / DIMENSION)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label
        
        if best_label is not None:
            # Return the clean version from memory
            self.access_count[best_label] = self.access_count.get(best_label, 0) + 1
            self._successful_cleanups += 1
            clean_hv = HyperVectorPy.from_bits(self.memory[best_label].copy())
            return clean_hv, best_label
        
        return None, None
    
    def cleanup_or_keep(self, noisy_hv: HyperVectorPy, threshold: float = 0.4) -> HyperVectorPy:
        """
        Cleanup variant that returns the original vector if no match found.
        
        Convenient for pipelines where you always want a vector output.
        
        Args:
            noisy_hv: The vector to clean
            threshold: Minimum similarity for cleanup
        
        Returns:
            Cleaned vector if match found, otherwise original noisy vector
        """
        clean_hv, _ = self.cleanup(noisy_hv, threshold)
        return clean_hv if clean_hv is not None else noisy_hv
    
    def batch_register(self, vectors: Dict[str, HyperVectorPy]):
        """
        Register multiple vectors at once.
        
        Args:
            vectors: Dictionary mapping labels to hypervectors
        """
        for label, hv in vectors.items():
            self.register(label, hv)
    
    def load_from_store(self, store):
        """
        Load all atomic concepts from a BrainStore/Persistence object.
        
        Args:
            store: BrainStore instance with load_concepts() method
        """
        concepts = store.load_concepts()
        for concept in concepts:
            # Assume concept has 'name' and 'vector' attributes
            if hasattr(concept, 'vector') and concept.vector is not None:
                hv = HyperVectorPy.from_bits(concept.vector)
                self.register(concept.name, hv)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about cleanup memory usage.
        
        Returns:
            Dictionary with telemetry data
        """
        success_rate = 0.0
        if self._total_cleanups > 0:
            success_rate = self._successful_cleanups / self._total_cleanups
        
        return {
            'size': len(self.memory),
            'max_size': self.max_size,
            'capacity_used': len(self.memory) / self.max_size,
            'total_cleanups': self._total_cleanups,
            'successful_cleanups': self._successful_cleanups,
            'success_rate': success_rate,
            'most_accessed': max(self.access_count.items(), key=lambda x: x[1])[0] if self.access_count else None
        }
    
    def clear(self):
        """Clear all registered vectors (useful for testing or reset)."""
        self.memory.clear()
        self.access_count.clear()
        self._total_cleanups = 0
        self._successful_cleanups = 0
    
    def __repr__(self):
        return f"<CleanupMemory size={len(self.memory)}/{self.max_size} cleanups={self._total_cleanups}>"


# Utility functions for common VSA operations with cleanup
def bundle_with_cleanup(vectors: List[HyperVectorPy], cleanup_mem: Optional[CleanupMemory] = None, threshold: float = 0.5) -> HyperVectorPy:
    """
    Bundle multiple vectors with optional cleanup at the end.
    
    Args:
        vectors: List of hypervectors to bundle
        cleanup_mem: Optional cleanup memory for denoising result
        threshold: Cleanup threshold if cleanup_mem provided
    
    Returns:
        Bundled (and optionally cleaned) hypervector
    """
    if not vectors:
        return HyperVector.zero()
    
    result = vectors[0]
    for v in vectors[1:]:
        result = result.bundle(v)
    
    if cleanup_mem is not None:
        result = cleanup_mem.cleanup_or_keep(result, threshold)
    
    return result


def unbind_with_cleanup(bound_hv: HyperVectorPy, key_hv: HyperVectorPy, cleanup_mem: Optional[CleanupMemory] = None, threshold: float = 0.5) -> HyperVectorPy:
    """
    Unbind a vector and cleanup the result.
    
    Unbinding is the same as binding in VSA (XOR is self-inverse):
        (A ⊗ B) ⊗ B = A
    
    Args:
        bound_hv: The bound/encrypted hypervector
        key_hv: The key to unbind with
        cleanup_mem: Optional cleanup memory for denoising result
        threshold: Cleanup threshold if cleanup_mem provided
    
    Returns:
        Unbound (and optionally cleaned) hypervector
    """
    result = bound_hv.xor(key_hv)
    
    if cleanup_mem is not None:
        result = cleanup_mem.cleanup_or_keep(result, threshold)
    
    return result