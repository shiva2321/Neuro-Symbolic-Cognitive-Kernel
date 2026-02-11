import numpy as np
import random

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
        # Majority rule for 2 vectors with random tie break
        # (A & B) | (A & Rand) | (B & !Rand)
        # Using numpy vectorized operations
        rand_mask = np.random.randint(0, 2, size=DIMENSION, dtype=np.int8)
        
        # Logic: If bits are same, keep. If differ, use mask.
        # diff = A ^ B
        # res = (A & B) | (diff & rand_mask)
        # Check:
        # A=1, B=1 -> 1&1 | 0 = 1 (Correct)
        # A=0, B=0 -> 0&0 | 0 = 0 (Correct)
        # A=1, B=0 -> 0 | (1 & mask) = mask (Correct 50/50)
        
        a = self.bits
        b = other.bits
        diff = np.bitwise_xor(a, b)
        same = np.bitwise_and(a, b)
        
        bundle_bits = np.bitwise_or(same, np.bitwise_and(diff, rand_mask))
        return HyperVectorPy.from_bits(bundle_bits)

    def similarity(self, other):
        # Hamming distance
        diff = np.bitwise_xor(self.bits, other.bits)
        hamming_dist = np.sum(diff)
        
        # Similarity = 1 - dist/dim
        return 1.0 - (hamming_dist / DIMENSION)

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
    
        
    def __repr__(self):
        return f"<HyperVector dim={DIMENSION} (Python)>"

# Re-export the Python class under the canonical name so that
# ``from hypervec_py import HyperVector`` works everywhere.
HyperVector = HyperVectorPy
