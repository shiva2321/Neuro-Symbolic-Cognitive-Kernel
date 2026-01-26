"""
NCGN v7 Semantic Embeddings Layer

Manages semantic embeddings for concepts using SentenceTransformers.

Provides:
- Entry point detection: Find closest concept to user input
- Semantic weight initialization: Auto-compute edge weights
- Fuzzy matching: "Cat" and "Cats" map to similar concepts

This layer enables semantic grounding - connecting language to the graph.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import GraphTopology
    from .state import CognitiveState

from .config import Config, DEFAULT_CONFIG


class SemanticLayer:
    """
    Manages semantic embeddings for concepts.
    
    Uses SentenceTransformers to encode text into dense vectors,
    enabling:
    - Semantic similarity search
    - Automatic edge weight computation
    - Fuzzy concept matching
    """
    
    def __init__(
        self, 
        model_name: str = None,
        config: Config = DEFAULT_CONFIG
    ):
        """
        Initialize semantic layer.
        
        Args:
            model_name: SentenceTransformer model name
                       (default: config.embedding_model)
            config: Configuration object
        """
        self.config = config
        model_name = model_name or config.embedding_model
        
        # Lazy load model
        self._model = None
        self._model_name = model_name
        self.embedding_dim = config.embedding_dim
        
        # Cache for frequently used embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 10000
    
    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for SemanticLayer. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        # Check cache for all texts
        uncached = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text not in self._cache:
                uncached.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts
        if uncached:
            new_embeddings = self.model.encode(
                uncached, 
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype(np.float32)
            
            # Add to cache
            for text, embedding in zip(uncached, new_embeddings):
                if len(self._cache) < self._cache_max_size:
                    self._cache[text] = embedding
        
        # Build result array
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        uncached_idx = 0
        for i, text in enumerate(texts):
            if text in self._cache:
                result[i] = self._cache[text]
            else:
                result[i] = new_embeddings[uncached_idx]
                uncached_idx += 1
        
        return result
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to vector."""
        return self.encode([text])[0]
    
    def compute_similarity(
        self, 
        vec_a: np.ndarray, 
        vec_b: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Returns:
            Similarity in range [-1, 1]
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    
    def compute_semantic_weight(self, text_a: str, text_b: str) -> float:
        """
        Compute edge weight from semantic similarity.
        
        Args:
            text_a: First concept label
            text_b: Second concept label
            
        Returns:
            Weight in range [0, 1]
        """
        vecs = self.encode([text_a, text_b])
        cos_sim = self.compute_similarity(vecs[0], vecs[1])
        
        # Map [-1, 1] → [0, 1]
        # This ensures negative similarity becomes weak (but not inhibitory) weight
        return float((cos_sim + 1) / 2)
    
    def find_nearest(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find nearest concepts to query.
        
        Args:
            query: The search query
            candidates: List of candidate labels
            top_k: Number of results to return
            
        Returns:
            List of (label, similarity) pairs, sorted by similarity descending
        """
        if not candidates:
            return []
        
        query_vec = self.encode_single(query)
        candidate_vecs = self.encode(candidates)
        
        # Normalize for cosine similarity
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-8:
            return [(c, 0.0) for c in candidates[:top_k]]
        
        query_normalized = query_vec / query_norm
        
        cand_norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        cand_norms[cand_norms < 1e-8] = 1.0
        cand_normalized = candidate_vecs / cand_norms
        
        # Compute all similarities at once
        similarities = np.dot(cand_normalized, query_normalized)
        
        # Get top k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        
        return [(candidates[i], float(similarities[i])) for i in top_indices]
    
    def find_entry_points(
        self,
        query: str,
        topology: 'GraphTopology',
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find the best entry points into the graph for a query.
        
        Args:
            query: User input text
            topology: The graph topology
            top_k: Maximum number of entry points
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (label, similarity) pairs
        """
        all_labels = topology.registry.all_labels()
        
        if not all_labels:
            return []
        
        results = self.find_nearest(query, all_labels, top_k=top_k)
        
        # Filter by minimum similarity
        return [(label, sim) for label, sim in results if sim >= min_similarity]
    
    def batch_compute_weights(
        self, 
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Compute weights for multiple pairs efficiently.
        
        Args:
            pairs: List of (source, target) label pairs
            
        Returns:
            List of weights in same order as pairs
        """
        if not pairs:
            return []
        
        # Collect unique labels
        all_labels = list(set(
            label for pair in pairs for label in pair
        ))
        
        # Encode all at once
        embeddings = self.encode(all_labels)
        label_to_vec = {
            label: embeddings[i] for i, label in enumerate(all_labels)
        }
        
        # Compute weights
        weights = []
        for src, tgt in pairs:
            sim = self.compute_similarity(label_to_vec[src], label_to_vec[tgt])
            weights.append((sim + 1) / 2)
        
        return weights
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict:
        """Get layer statistics."""
        return {
            "model_name": self._model_name,
            "embedding_dim": self.embedding_dim,
            "cache_size": len(self._cache),
            "model_loaded": self._model is not None,
        }
    
    def __repr__(self) -> str:
        return f"SemanticLayer(model={self._model_name}, cache={len(self._cache)})"
