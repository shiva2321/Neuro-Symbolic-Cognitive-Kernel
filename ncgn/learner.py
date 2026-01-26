"""
NCGN v7 Hebbian Learning Module

3-Factor Hebbian learning with reward modulation.

Learning Rule:
    ΔW = η × M × pre × post × exists(W)

Where:
    - η: Learning rate
    - M: Reward modulator (positive = LTP, negative = LTD)
    - pre: Pre-synaptic activation
    - post: Post-synaptic activation
    - exists(W): Only update existing edges (unless genesis threshold exceeded)

This implements biologically-plausible reward-modulated plasticity.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import GraphTopology
    from .state import CognitiveState

from .config import Config, DEFAULT_CONFIG


class HebbianLearner:
    """
    3-Factor Hebbian learning with reward modulation.
    
    Only updates EXISTING synapses (synaptic scaling) by default.
    New synapses require explicit structural plasticity call or
    exceeding the genesis threshold.
    
    The third factor (reward/dopamine) allows:
    - Positive modulation: Strengthen active synapses (LTP)
    - Negative modulation: Weaken active synapses (LTD)
    - Zero modulation: No change (neutral)
    """
    

    def __init__(
        self, 
        state: 'CognitiveState', 
        topology: 'GraphTopology',
        config: Config = DEFAULT_CONFIG,
        confidence_tracker = None
    ):
        self.state = state
        self.topology = topology
        self.config = config
        self.confidence = confidence_tracker
        
        # Co-occurrence history for auto-genesis
        # Key: tuple(sorted(idx1, idx2)), Value: count
        self.co_occurrence_counts: Dict[Tuple[int, int], int] = {}
        
        # Learning statistics
        self._total_updates = 0
        self._total_ltp = 0
        self._total_ltd = 0
        self._last_update_count = 0
        self._genesis_count = 0
        
    def auto_learn(self, active_indices: Set[int], threshold: int = 5) -> int:
        """
        Automatically create edges based on co-occurrence history.
        NO LLM required.
        
        Args:
            active_indices: Set of currently active node indices
            threshold: Number of co-occurrences required to form an edge
            
        Returns:
            Number of new edges created
        """
        sorted_indices = sorted(list(active_indices))
        new_edges = 0
        
        # Compare all pairs (O(N^2) on active set - usually small)
        for i in range(len(sorted_indices)):
            idx1 = sorted_indices[i]
            for j in range(i + 1, len(sorted_indices)):
                idx2 = sorted_indices[j]
                
                pair_key = (idx1, idx2)
                
                # Increment count
                self.co_occurrence_counts[pair_key] = self.co_occurrence_counts.get(pair_key, 0) + 1
                
                # Check for genesis
                if self.co_occurrence_counts[pair_key] >= threshold:
                    label1 = self.topology.registry.get_label(idx1)
                    label2 = self.topology.registry.get_label(idx2)
                    
                    if label1 and label2:
                        # Check if edge exists
                        if not self.topology.get_edge_weight(label1, label2):
                            # Auto-create edge!
                            self.topology.add_connection(label1, label2, weight=0.3)
                            
                            # Initial confidence
                            if self.confidence:
                                self.confidence.reinforce(label1, label2, source_type="auto_hebbian")
                                
                            new_edges += 1
                            self._genesis_count += 1
                            
                            # Reset counter (or keep high to maintain?)
                            # Resetting prevents spamming add_connection
                            # But we might want to keep tracking for confidence.
                            # Let's keep it but only add if not exists.
        
        return new_edges

    def apply_reward(self, modulator: float, active_threshold: float = 0.1) -> int:
        """
        Apply reward-modulated learning to all active synapses.
        
        Args:
            modulator: Reward signal. 
                      - Positive = LTP (strengthen)
                      - Negative = LTD (weaken)
            active_threshold: Minimum activation to be considered "active"
        
        Returns:
            Number of synapses updated
        """
        if abs(modulator) < 0.001:
            return 0
        
        A = self.state.activations
        W = self.state.adjacency
        
        # Determine number of active nodes even if W is empty
        # We use activations array length, but need to know valid range.
        # State usually tracks max index.
        n = len(A)
        active_mask = A[:n] > active_threshold
        active_indices = set(np.where(active_mask)[0].tolist())
        
        # Auto-learn from this activation pattern
        # This MUST happen before returning on empty W
        if modulator > 0 and len(active_indices) >= 2:
             self.auto_learn(active_indices)

        if W is None or W.nnz == 0:
            return 0
        
        if len(active_indices) < 2:
            return 0
        
        updated = 0
        lr = self.config.learning_rate
        
        # Iterate over non-zero entries in CSR matrix
        for i in range(W.shape[0]):
            if i not in active_indices:
                continue
            
            pre_activation = A[i]
            row_start = W.indptr[i]
            row_end = W.indptr[i + 1]
            
            for idx in range(row_start, row_end):
                j = W.indices[idx]
                
                if j >= n or j not in active_indices:
                    continue
                
                post_activation = A[j]
                
                # 3-Factor Hebbian: ΔW = η × M × pre × post
                delta = lr * modulator * pre_activation * post_activation
                
                # Update weight with clamping
                new_weight = W.data[idx] + delta
                W.data[idx] = np.clip(
                    new_weight, 
                    self.config.weight_min, 
                    self.config.weight_max
                )
                
                updated += 1
                
                # Track LTP vs LTD
                if delta > 0:
                    self._total_ltp += 1
                    # Reinforce confidence
                    if self.confidence:
                        l1 = self.topology.registry.get_label(i)
                        l2 = self.topology.registry.get_label(j)
                        if l1 and l2:
                            self.confidence.reinforce(l1, l2, source_type="hebbian_reward")
                else:
                    self._total_ltd += 1
        
        self._total_updates += updated
        self._last_update_count = updated
        
        return updated
    
    def apply_targeted_ltd(
        self, 
        source: str, 
        target: str, 
        factor: float = 0.5
    ) -> bool:
        """
        Apply Long-Term Depression to a specific synapse.
        
        This is used by System 2 to weaken incorrect associations.
        """
        src_idx = self.topology.registry.get_index(source)
        tgt_idx = self.topology.registry.get_index(target)
        
        if src_idx is None or tgt_idx is None:
            return False
        
        W = self.state.adjacency
        if W is None:
            return False
        
        # Find the edge in CSR matrix
        row_start = W.indptr[src_idx]
        row_end = W.indptr[src_idx + 1]
        
        for idx in range(row_start, row_end):
            if W.indices[idx] == tgt_idx:
                W.data[idx] *= factor
                W.data[idx] = max(W.data[idx], self.config.weight_min)
                self._total_ltd += 1
                return True
        
        return False
    
    def apply_targeted_ltp(
        self, 
        source: str, 
        target: str, 
        factor: float = 1.5
    ) -> bool:
        """
        Apply Long-Term Potentiation to a specific synapse.
        """
        src_idx = self.topology.registry.get_index(source)
        tgt_idx = self.topology.registry.get_index(target)
        
        if src_idx is None or tgt_idx is None:
            return False
        
        W = self.state.adjacency
        if W is None:
            return False
        
        # Find the edge in CSR matrix
        row_start = W.indptr[src_idx]
        row_end = W.indptr[src_idx + 1]
        
        for idx in range(row_start, row_end):
            if W.indices[idx] == tgt_idx:
                W.data[idx] *= factor
                W.data[idx] = min(W.data[idx], self.config.weight_max)
                self._total_ltp += 1
                return True
        
        return False
    
    def propose_new_edges(self, threshold: float = None) -> List[Tuple[str, str, float]]:
        # Legacy support or LLM usage - minimal update needed
        return []

    def create_proposed_edges(self, max_new: int = 5) -> int:
        return 0

    def get_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            "total_updates": self._total_updates,
            "total_ltp": self._total_ltp,
            "total_ltd": self._total_ltd,
            "genesis_count": self._genesis_count,
            "auto_pairs_tracked": len(self.co_occurrence_counts),
            "last_update_count": self._last_update_count,
            "learning_rate": self.config.learning_rate,
        }
    
    def reset_stats(self) -> None:
        """Reset learning statistics."""
        self._total_updates = 0
        self._total_ltp = 0
        self._total_ltd = 0
        self._last_update_count = 0
        self._genesis_count = 0
        self.co_occurrence_counts.clear()
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"HebbianLearner(updates={stats['total_updates']}, "
            f"LTP={stats['total_ltp']}, LTD={stats['total_ltd']}, "
            f"Genesis={stats['genesis_count']})"
        )


class RewardModulator:
    """
    Dopamine-like reward signal generator.
    """
    
    def __init__(self, baseline_alpha: float = 0.1):
        self.baseline_alpha = baseline_alpha
        self.expected_reward = 0.0
        
        # History
        self._reward_history: List[float] = []
        self._rpe_history: List[float] = []
    
    def calculate_signal(
        self, 
        actual_reward: float,
        predicted_value: float = 0.0
    ) -> float:
        """
        Calculate the dopamine/reward signal.
        """
        # Use either predicted value or moving baseline
        expected = predicted_value if predicted_value != 0 else self.expected_reward
        
        # RPE = actual - expected
        rpe = actual_reward - expected
        
        # Update baseline with exponential moving average
        self.expected_reward = (
            self.baseline_alpha * actual_reward + 
            (1 - self.baseline_alpha) * self.expected_reward
        )
        
        # Record history
        self._reward_history.append(actual_reward)
        self._rpe_history.append(rpe)
        
        # Keep history bounded
        if len(self._reward_history) > 1000:
            self._reward_history = self._reward_history[-500:]
            self._rpe_history = self._rpe_history[-500:]
        
        return rpe
    
    def reset(self) -> None:
        """Reset reward baseline and history."""
        self.expected_reward = 0.0
        self._reward_history.clear()
        self._rpe_history.clear()
    
    def get_stats(self) -> Dict:
        """Get modulator statistics."""
        return {
            "expected_reward": self.expected_reward,
            "history_length": len(self._reward_history),
            "avg_reward": np.mean(self._reward_history) if self._reward_history else 0.0,
            "avg_rpe": np.mean(self._rpe_history) if self._rpe_history else 0.0,
        }
