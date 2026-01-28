"""
NCGN v7 Propagation Engine

Vectorized System 1 dynamics using sparse matrix operations.
One "tick" of cognition is a matrix-vector multiplication
followed by nonlinear activation and refractory masking.

Core Equation:
    A_{t+1} = σ((A_t × (1-δ) + W·A_t × α + I_ext) / (1 + β·ΣA)) ⊙ (1-R)

Where:
    - A_t: Activation vector at time t
    - δ: Decay rate
    - W: Sparse adjacency matrix (CSR)
    - α: Flow rate / conductivity
    - I_ext: External input vector
    - β: Normalization constant
    - σ: Sigmoid activation
    - R: Refractory mask
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import GraphTopology
    from .state import CognitiveState

from .config import Config, DEFAULT_CONFIG


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-np.clip(x, -500, 500))),
        np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500)))
    )


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    x_scaled = x / max(temperature, 0.01)
    x_shifted = x_scaled - np.max(x_scaled)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / (np.sum(exp_x) + 1e-10)


class PropagationEngine:
    """
    Vectorized System 1 dynamics using sparse matrix operations.
    
    This is the "Physics Engine for Concepts" - treating thought
    as signal propagation through a weighted graph using linear algebra.
    
    Features:
    - Sparse CSR matrix multiplication for O(E) propagation
    - Vectorized refractory period handling
    - Divisive normalization for homeostasis
    - Seizure damping for stability
    - External input buffering
    """
    
    def __init__(
        self, 
        state: 'CognitiveState', 
        topology: 'GraphTopology',
        config: Config = DEFAULT_CONFIG
    ):
        self.state = state
        self.topology = topology
        self.config = config
        
        # Tick counter
        self.tick_count = 0
        
        # External input buffer (injected before propagation)
        self._external_buffer = np.zeros(state.capacity, dtype=np.float32)
        
        # Track which indices have pending input
        self._pending_indices: Set[int] = set()
        
        # Last firing set (for learning eligibility)
        self._last_firing_set: Set[int] = set()
        
        # Surprise tracking
        self._expected_state: Optional[np.ndarray] = None
        self._last_surprise: float = 0.0
    
    def inject(self, idx: int, energy: float) -> None:
        """
        Queue external energy to be applied next tick.
        
        This is the ONLY way external energy enters the system.
        Energy is buffered and applied atomically at the start of propagate().
        
        Args:
            idx: Node index to inject into
            energy: Amount of energy to add
        """
        self.state.ensure_capacity(idx)
        
        # Ensure buffer is large enough
        if idx >= len(self._external_buffer):
            new_buffer = np.zeros(self.state.capacity, dtype=np.float32)
            new_buffer[:len(self._external_buffer)] = self._external_buffer
            self._external_buffer = new_buffer
        
        self._external_buffer[idx] += energy
        self._pending_indices.add(idx)
    
    def inject_by_label(self, label: str, energy: float) -> bool:
        """
        Inject energy by concept label.
        
        Args:
            label: Concept label
            energy: Amount of energy
            
        Returns:
            True if successful, False if label not found
        """
        idx = self.topology.registry.get_index(label)
        if idx is None:
            return False
        self.inject(idx, energy)
        return True
    
    def inject_multiple(self, injections: Dict[str, float]) -> int:
        """
        Inject energy into multiple concepts.
        
        Args:
            injections: Dict mapping labels to energy values
            
        Returns:
            Number of successful injections
        """
        count = 0
        for label, energy in injections.items():
            if self.inject_by_label(label, energy):
                count += 1
        return count
    
    def _apply_external_input(self) -> None:
        """Apply buffered external input to activations."""
        if not self._pending_indices:
            return
        
        A = self.state.activations
        n = min(len(A), len(self._external_buffer))
        
        A[:n] = np.clip(A[:n] + self._external_buffer[:n], 0.0, 1.0)
        
        # Clear buffer
        self._external_buffer[:n] = 0.0
        self._pending_indices.clear()
    
    def _capture_expected_state(self) -> None:
        """Capture current state for surprise calculation."""
        self._expected_state = self.state.activations.copy()
    
    def _calculate_surprise(self) -> float:
        """
        Calculate surprise as RMS difference between expected and observed.
        
        Returns:
            Surprise level (0.0 = no surprise)
        """
        if self._expected_state is None:
            return 0.0
        
        observed = self.state.activations
        n = min(len(self._expected_state), len(observed))
        
        # Only compare active regions
        active_mask = (self._expected_state[:n] > 0.01) | (observed[:n] > 0.01)
        
        if not np.any(active_mask):
            return 0.0
        
        diff = self._expected_state[:n][active_mask] - observed[:n][active_mask]
        surprise = float(np.sqrt(np.mean(diff ** 2)))
        
        self._last_surprise = surprise
        return surprise
    
    def propagate(self, steps: int = 1) -> Dict[str, float]:
        """
        Run propagation for N steps.
        
        This is the core System 1 computation:
        1. Apply external input
        2. Decay existing activations
        3. Propagate through adjacency matrix
        4. Apply divisive normalization
        5. Apply sigmoid activation
        6. Apply refractory masking
        7. Update refractory counters
        8. Seizure damping if needed
        
        Args:
            steps: Number of ticks to run
            
        Returns:
            Dictionary of active labels → activation values
        """
        # Ensure matrix is synchronized with topology
        self.state.synchronize_matrix(self.topology)
        
        A = self.state.activations
        W = self.state.adjacency
        R = self.state.refractory_counters
        thresholds = self.state.thresholds
        
        cfg = self.config
        
        for _ in range(steps):
            # Capture state for surprise calculation
            self._capture_expected_state()
            
            # Phase 0: Apply external input
            self._apply_external_input()
            
            # Determine working size
            if W is not None and W.shape[0] > 0:
                n = W.shape[0]
            else:
                n = self.state.current_max_idx + 1 if self.state.current_max_idx >= 0 else 0
            
            if n == 0:
                self.tick_count += 1
                continue
            
            # Ensure arrays cover the working size
            if n > len(A):
                self.state.ensure_capacity(n - 1)
                A = self.state.activations
                R = self.state.refractory_counters
                thresholds = self.state.thresholds
            
            # Phase 1: Decay (passive leak)
            A[:n] *= (1 - cfg.decay_delta)
            
            # Phase 2: Propagation through adjacency matrix
            if W is not None and W.nnz > 0:
                # Matrix-vector multiplication: propagated = W @ A
                propagated = W.dot(A[:n]) * cfg.flow_alpha
                A[:n] += propagated
            
            # Phase 3: Divisive normalization (homeostasis)
            total_energy = float(np.sum(A[:n]))
            if total_energy > 0:
                normalization_factor = 1.0 + cfg.norm_beta * total_energy
                A[:n] /= normalization_factor
            
            # Phase 4: Sigmoid activation (squash to 0-1)
            # Scale to useful sigmoid range
            A[:n] = sigmoid(A[:n] * 6 - 3)
            
            # Phase 5: Identify firing nodes
            can_fire = (A[:n] > thresholds[:n]) & (R[:n] == 0)
            firing_indices = np.where(can_fire)[0]
            
            # Store firing set for learning
            self._last_firing_set = set(firing_indices.tolist())
            
            # Phase 6: Apply refractory period to fired nodes
            R[firing_indices] = cfg.refractory_period
            A[firing_indices] = 0.0  # Reset after firing
            
            # Phase 7: Refractory mask - zero out refractory nodes
            refractory_mask = (R[:n] == 0).astype(np.float32)
            A[:n] *= refractory_mask
            
            # Phase 8: Decay refractory counters
            R[R > 0] -= 1
            
            # Phase 9: Seizure damping (global stability)
            current_total = float(np.sum(A[:n]))
            if current_total > cfg.seizure_threshold:
                A[:n] *= cfg.seizure_damping
            
            # Phase 10: Clamp values
            A[:n] = np.clip(A[:n], 0.0, 1.0)
            
            # Zero out very small values (numerical cleanup)
            A[A < 0.001] = 0.0
            
            # Calculate surprise
            self._calculate_surprise()
            
            # Decay novelty scores
            self.state.novelty_scores *= 0.999
            
            self.tick_count += 1
            self.state._tick_count = self.tick_count
        
        # Return active concepts
        return self.get_active_concepts()
    
    def get_active_concepts(self, threshold: float = None) -> Dict[str, float]:
        """
        Get dictionary of active concepts and their activations.
        
        Args:
            threshold: Minimum activation (default: config.active_energy_threshold)
            
        Returns:
            Dict mapping labels to activation values
        """
        if threshold is None:
            threshold = self.config.active_energy_threshold
        
        A = self.state.activations
        active_mask = A > threshold
        active_indices = np.where(active_mask)[0]
        
        result = {}
        for idx in active_indices:
            label = self.topology.registry.get_label(int(idx))
            if label is not None:
                result[label] = float(A[idx])
        
        return result
    
    def get_top_k_active(self, k: int = 10) -> Dict[str, float]:
        """
        Get top K most active concepts.
        
        Args:
            k: Number of top concepts to return
            
        Returns:
            Dict mapping labels to activation values (sorted by activation)
        """
        A = self.state.activations
        
        # Get indices of top k values
        if len(A) <= k:
            top_indices = np.argsort(A)[::-1]
        else:
            # Use argpartition for efficiency on large arrays
            top_indices = np.argpartition(A, -k)[-k:]
            top_indices = top_indices[np.argsort(A[top_indices])][::-1]
        
        result = {}
        for idx in top_indices:
            if A[idx] <= 0.001:
                continue
            label = self.topology.registry.get_label(int(idx))
            if label is not None:
                result[label] = float(A[idx])
        
        return result
    
    def get_firing_set(self) -> Set[str]:
        """Get labels of nodes that fired in the last tick."""
        labels = set()
        for idx in self._last_firing_set:
            label = self.topology.registry.get_label(idx)
            if label:
                labels.add(label)
        return labels
    
    @property
    def last_surprise(self) -> float:
        """Get the surprise level from the last tick."""
        return self._last_surprise
    
    def reset(self) -> None:
        """Reset engine state."""
        self._external_buffer.fill(0.0)
        self._pending_indices.clear()
        self._last_firing_set.clear()
        self._expected_state = None
        self._last_surprise = 0.0
        self.tick_count = 0
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "tick_count": self.tick_count,
            "last_surprise": self._last_surprise,
            "pending_injections": len(self._pending_indices),
            "last_firing_count": len(self._last_firing_set),
            "active_concepts": len(self.get_active_concepts()),
            "total_energy": self.state.get_total_energy(),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PropagationEngine(ticks={stats['tick_count']}, "
            f"active={stats['active_concepts']}, "
            f"energy={stats['total_energy']:.2f})"
        )
