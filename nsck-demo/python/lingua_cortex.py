"""
NSCK Language Cortex (LinguaCortex)
===================================
Implementation of Efficient NLP using Semantic Folding and Vector Symbolic Architectures (VSA).

Theory:
- We do not use dense embeddings (Word2Vec).
- We use Sparse Distributed Representations (SDRs) on a 2D Topographical Map.
- Words are represented as binary fingerprints (2% sparsity).

Components:
- SemanticMap: The 2D grid (128x128) that stores context.
- TextRetina: Converts text to SDRs.
"""

import numpy as np
import hashlib
from typing import List, Dict, Tuple, Set, Optional
import pickle
import os

# --- CONSTANTS ---
GRID_SIZE = 128        # 128x128 = 16,384 bits
SPARSITY = 0.02        # Target sparsity (2%)
ON_BITS = int((GRID_SIZE * GRID_SIZE) * SPARSITY)

class SemanticFingerprint:
    """
    A Sparse Distributed Representation (SDR) of a concept.
    Wraps a binary numpy array.
    """
    def __init__(self, bits: np.ndarray):
        self.bits = bits # shape: (GRID_SIZE, GRID_SIZE), dtype=bool
        
    @property
    def density(self) -> float:
        return np.mean(self.bits)
        
    def union(self, other: 'SemanticFingerprint') -> 'SemanticFingerprint':
        """Boolean OR: Bundling"""
        return SemanticFingerprint(np.logical_or(self.bits, other.bits))
        
    def intersection(self, other: 'SemanticFingerprint') -> 'SemanticFingerprint':
        """Boolean AND: Disambiguation"""
        return SemanticFingerprint(np.logical_and(self.bits, other.bits))
        
    def overlap(self, other: 'SemanticFingerprint') -> float:
        """
        Overlap Score: |A AND B| / |A|
        (How much of THIS concept is contained in OTHER?)
        """
        intersection = np.logical_and(self.bits, other.bits)
        self_count = np.sum(self.bits)
        if self_count == 0: return 0.0
        return np.sum(intersection) / self_count
        
    def jaccard(self, other: 'SemanticFingerprint') -> float:
        """
        Jaccard Similarity: |A AND B| / |A OR B|
        """
        intersection = np.logical_and(self.bits, other.bits)
        union = np.logical_or(self.bits, other.bits)
        union_count = np.sum(union)
        if union_count == 0: return 0.0
        return np.sum(intersection) / union_count
        
    def __repr__(self):
        return f"<Fingerprint density={self.density:.3f}>"


class SemanticMap:
    """
    The Topographical Semantic Space.
    Instead of a full SOM (slow), we use a Hashed Random Indexing approach 
    mapped to 2D coordinates for efficiency in Phase 1.
    """
    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.total_bits = size * size
        # Vocabulary: word -> SemanticFingerprint
        self.vocab: Dict[str, SemanticFingerprint] = {}
        
        # In a real SOM, 'contexts' would be learned clusters.
        # Here, we simulate 'contexts' as random distinct zones in the grid
        # to bootstrap the system without a massive training phase.
        self.contexts: Dict[str, Tuple[int, int]] = {}

    def _hash_word_to_seed(self, word: str) -> int:
        return int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % (10**8)

    def learn_text_snippet(self, text: str):
        """
        Unsupervised Learning:
        1. Tokenize text (stop word removal skipped for prototype simplicity).
        2. Create a 'Context Fingerprint' for the snippet.
        3. Add this context to all words in the snippet.
        """
        words = set(text.lower().split())
        words = {w for w in words if w.isalnum() and len(w) > 2}
        
        if not words: return
        
        # Generate a random 'center' for this snippet context on the 2D map
        # This simulates the "Folding" where this snippet occupies a specific coordinate.
        snippet_hash = sum([self._hash_word_to_seed(w) for w in words])
        cx = snippet_hash % self.size
        cy = (snippet_hash // self.size) % self.size
        
        # Create a localized SDR blobs around (cx, cy)
        # Using a Gaussian-like scattering of bits
        snippet_bits = np.zeros((self.size, self.size), dtype=bool)
        
        # Scatter bits around the center
        np.random.seed(snippet_hash % (2**32))
        for _ in range(ON_BITS):
            # Normal distribution around center
            dx = int(np.random.normal(0, 5))
            dy = int(np.random.normal(0, 5))
            x = (cx + dx) % self.size
            y = (cy + dy) % self.size
            snippet_bits[y, x] = True
            
        snippet_fp = SemanticFingerprint(snippet_bits)
        
        # Hebbian Learning:
        # For each word in the snippet, add the snippet's bits to the word's representation.
        # "Word W appeared in Context C, so W acquires properties of C."
        for w in words:
            if w not in self.vocab:
                self.vocab[w] = SemanticFingerprint(np.zeros((self.size, self.size), dtype=bool))
            
            # OR-ing the bits (Hebbian accumulation)
            # To prevent saturation (all 1s), we might need a threshold or decay.
            # For Phase 1, purely additive is fine if vocab is small.
            self.vocab[w] = self.vocab[w].union(snippet_fp)

    def get_fingerprint(self, word: str) -> Optional[SemanticFingerprint]:
        return self.vocab.get(word.lower())

    def similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find concept neighbors."""
        target = self.get_fingerprint(word)
        if not target: return []
        
        scores = []
        for w, fp in self.vocab.items():
            if w == word: continue
            score = target.overlap(fp)
            scores.append((w, score))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.vocab, f)
            
    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.vocab = pickle.load(f)

# --- Singleton ---
_cortex = None

def get_lingua_cortex() -> SemanticMap:
    global _cortex
    if _cortex is None:
        _cortex = SemanticMap()
    return _cortex
