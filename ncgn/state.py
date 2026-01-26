"""
NCGN v7 Cognitive State Module

Structure-of-Arrays (SoA) representation of cognitive state.
All node properties are stored in contiguous numpy arrays for
cache-optimal vectorized operations.

CRITICAL: This is the core of Data-Oriented Design.
- NO object attributes for dynamic state
- All state in contiguous arrays indexed by integer
- CSR matrix cached and rebuilt only when topology changes
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import GraphTopology

from .config import Config, DEFAULT_CONFIG


class CognitiveState:
    """
    Structure-of-Arrays representation of cognitive state.
    
    Memory Layout (for N nodes):
    - activations:         float32[N] - Current membrane potential (0.0-1.0)
    - thresholds:          float32[N] - Firing threshold per node
    - refractory_counters: int32[N]   - Cooldown ticks remaining
    - resting_potentials:  float32[N] - Baseline energy level
    - novelty_scores:      float32[N] - Decays over time
    
    The adjacency matrix is cached as CSR and rebuilt only when
    the topology changes (dirty flag).
    
    Performance Notes:
    - float32 provides 2x memory bandwidth vs float64
    - Pre-allocated capacity avoids O(N) resize operations
    - Contiguous arrays enable SIMD vectorization
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.capacity = config.initial_capacity
        self.embedding_dim = config.embedding_dim
        
        # Track highest used index
        self.current_max_idx = -1
        
        # Core state arrays (pre-allocated, contiguous)
        self.activations = np.zeros(self.capacity, dtype=np.float32)
        self.thresholds = np.full(
            self.capacity, 
            config.default_threshold, 
            dtype=np.float32
        )
        self.refractory_counters = np.zeros(self.capacity, dtype=np.int32)
        self.resting_potentials = np.zeros(self.capacity, dtype=np.float32)
        self.novelty_scores = np.ones(self.capacity, dtype=np.float32)
        
        # Semantic embeddings (lazy-loaded when first used)
        self._embeddings: Optional[np.ndarray] = None
        
        # Cached sparse adjacency matrix
        self._adj_csr: Optional[csr_matrix] = None
        
        # Statistics
        self._tick_count = 0
    
    @property
    def embeddings(self) -> np.ndarray:
        """Lazy-load embeddings array on first access."""
        if self._embeddings is None:
            self._embeddings = np.zeros(
                (self.capacity, self.embedding_dim), 
                dtype=np.float32
            )
        return self._embeddings
    
    def ensure_capacity(self, required_idx: int) -> bool:
        """
        Ensure arrays can hold the required index.
        Doubles capacity if needed (amortized O(1) growth).
        
        Args:
            required_idx: The index that must be valid
            
        Returns:
            True if resize occurred, False otherwise
        """
        if required_idx < self.capacity:
            # Update max index tracking
            if required_idx > self.current_max_idx:
                self.current_max_idx = required_idx
            return False
        
        # Calculate new capacity (double until sufficient)
        new_capacity = self.capacity
        while new_capacity <= required_idx:
            new_capacity *= 2
        
        # Resize all arrays (preserving data)
        old_capacity = self.capacity
        
        # Activations - new slots start at 0
        new_activations = np.zeros(new_capacity, dtype=np.float32)
        new_activations[:old_capacity] = self.activations
        self.activations = new_activations
        
        # Thresholds - new slots get default
        new_thresholds = np.full(
            new_capacity, 
            self.config.default_threshold, 
            dtype=np.float32
        )
        new_thresholds[:old_capacity] = self.thresholds
        self.thresholds = new_thresholds
        
        # Refractory counters - new slots start at 0
        new_refractory = np.zeros(new_capacity, dtype=np.int32)
        new_refractory[:old_capacity] = self.refractory_counters
        self.refractory_counters = new_refractory
        
        # Resting potentials - new slots start at 0
        new_resting = np.zeros(new_capacity, dtype=np.float32)
        new_resting[:old_capacity] = self.resting_potentials
        self.resting_potentials = new_resting
        
        # Novelty scores - new slots start at 1.0
        new_novelty = np.ones(new_capacity, dtype=np.float32)
        new_novelty[:old_capacity] = self.novelty_scores
        self.novelty_scores = new_novelty
        
        # Embeddings (if loaded)
        if self._embeddings is not None:
            new_embeddings = np.zeros(
                (new_capacity, self.embedding_dim), 
                dtype=np.float32
            )
            new_embeddings[:old_capacity] = self._embeddings
            self._embeddings = new_embeddings
        
        self.capacity = new_capacity
        
        # Update max index tracking
        if required_idx > self.current_max_idx:
            self.current_max_idx = required_idx
        
        return True
    
    def zero_index(self, idx: int) -> None:
        """
        Zero all state at a specific index.
        
        MUST be called immediately after node deletion!
        Failure to call this will cause "ghost activation" bugs
        where deleted concepts influence new concepts.
        
        Args:
            idx: The index to clear
        """
        if idx >= self.capacity:
            return
        
        self.activations[idx] = 0.0
        self.thresholds[idx] = self.config.default_threshold
        self.refractory_counters[idx] = 0
        self.resting_potentials[idx] = 0.0
        self.novelty_scores[idx] = 1.0
        
        if self._embeddings is not None:
            self._embeddings[idx] = 0.0
    
    def set_activation(self, idx: int, energy: float) -> None:
        """Set activation at index, ensuring capacity."""
        self.ensure_capacity(idx)
        self.activations[idx] = np.clip(energy, 0.0, 1.0)
    
    def add_activation(self, idx: int, energy: float) -> None:
        """Add energy to activation at index, ensuring capacity."""
        self.ensure_capacity(idx)
        self.activations[idx] = np.clip(
            self.activations[idx] + energy, 
            0.0, 
            1.0
        )
    
    def set_embedding(self, idx: int, vector: np.ndarray) -> None:
        """Set embedding vector at index."""
        self.ensure_capacity(idx)
        if vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {vector.shape[0]}"
            )
        self.embeddings[idx] = vector.astype(np.float32)
    
    def synchronize_matrix(self, topology: 'GraphTopology') -> None:
        """
        Rebuild the CSR adjacency matrix from topology.
        
        Only rebuilds if topology.is_dirty == True.
        This is the expensive operation that should be minimized.
        
        Args:
            topology: The GraphTopology to sync with
        """
        if not topology.is_dirty:
            return
        
        sources, targets, weights = topology.get_adjacency_data()
        
        # Matrix size must accommodate all indices
        n = max(
            topology.registry.max_index() + 1 if topology.num_nodes > 0 else 1,
            self.current_max_idx + 1
        )
        
        if len(sources) == 0:
            self._adj_csr = csr_matrix((n, n), dtype=np.float32)
        else:
            # Build COO first (fast construction), then convert to CSR
            coo = coo_matrix(
                (weights, (sources, targets)), 
                shape=(n, n), 
                dtype=np.float32
            )
            self._adj_csr = coo.tocsr()
        
        topology.mark_clean()
    
    @property
    def adjacency(self) -> Optional[csr_matrix]:
        """Get the cached CSR adjacency matrix."""
        return self._adj_csr
    
    def get_active_indices(self, threshold: float = None) -> np.ndarray:
        """
        Get indices of all active nodes.
        
        Args:
            threshold: Minimum activation to be "active" 
                      (default: config.active_energy_threshold)
        
        Returns:
            Array of active node indices
        """
        if threshold is None:
            threshold = self.config.active_energy_threshold
        
        return np.where(self.activations > threshold)[0]
    
    def get_active_count(self, threshold: float = None) -> int:
        """Count of active nodes."""
        return len(self.get_active_indices(threshold))
    
    def get_total_energy(self) -> float:
        """Sum of all activations."""
        return float(np.sum(self.activations))
    
    def get_firing_indices(self) -> np.ndarray:
        """Get indices of nodes ready to fire (above threshold, not refractory)."""
        can_fire = (
            (self.activations > self.thresholds) & 
            (self.refractory_counters == 0)
        )
        return np.where(can_fire)[0]
    
    def clear_all_activations(self) -> None:
        """Reset all activations to zero."""
        self.activations.fill(0.0)
    
    def clear_all(self) -> None:
        """Reset all state to defaults."""
        self.activations.fill(0.0)
        self.thresholds.fill(self.config.default_threshold)
        self.refractory_counters.fill(0)
        self.resting_potentials.fill(0.0)
        self.novelty_scores.fill(1.0)
        if self._embeddings is not None:
            self._embeddings.fill(0.0)
        self._adj_csr = None
        self._tick_count = 0
        self.current_max_idx = -1
    
    def get_stats(self) -> Dict:
        """Get state statistics."""
        active_count = self.get_active_count()
        return {
            "capacity": self.capacity,
            "current_max_idx": self.current_max_idx,
            "active_count": active_count,
            "total_energy": self.get_total_energy(),
            "tick_count": self._tick_count,
            "matrix_shape": self._adj_csr.shape if self._adj_csr is not None else None,
            "matrix_nnz": self._adj_csr.nnz if self._adj_csr is not None else 0,
            "embeddings_loaded": self._embeddings is not None,
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"CognitiveState(capacity={stats['capacity']}, "
            f"active={stats['active_count']}, "
            f"energy={stats['total_energy']:.2f})"
        )
