"""
Hebbian Learning Module
=======================
Implements Hebbian learning for NSCK cognitive architecture.

"Neurons that fire together, wire together" - Donald Hebb (1949)

This module provides:
1. Oja's rule for normalized Hebbian learning
2. BCM rule for homeostatic plasticity
3. VSA-compatible weight updates
4. Integration with concept formation

Key Features:
- Unsupervised learning from co-activation patterns
- No backpropagation required
- Biologically plausible
- Fast (0.1-1ms per update)
- Works with or without PyTorch

Mathematical Foundation:
    Oja's Rule: Δw_ij = η * (x_i * y_j - y_j² * w_ij)
    This prevents weights from growing unbounded and normalizes them.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

logger = logging.getLogger("nsck.hebbian")


# ===========================================================================
# NumPy Implementation (CPU-only, no dependencies)
# ===========================================================================

class HebbianMatrixNumPy:
    """
    Pure NumPy Hebbian learning matrix with reward modulation.
    
    Fast, lightweight, no external dependencies beyond NumPy.
    Perfect for CPU-only deployments and embedded systems.
    
    Supports both:
    - Standard Hebbian: Δw = η * (pre * post)
    - Reward-modulated: Δw = η * (reward - baseline) * (pre * post)
    """
    
    def __init__(self, in_features: int, out_features: int,
                 learning_rate: float = 0.01,
                 decay: float = 0.001,
                 normalize: bool = True,
                 reward_modulated: bool = False,
                 baseline: float = 0.0,
                 eligibility_decay: float = 0.9):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            learning_rate: Hebbian learning rate (η)
            decay: Weight decay factor
            normalize: Use Oja's normalization to prevent weight explosion
            reward_modulated: Enable reward-modulated plasticity
            baseline: Reward baseline for advantage calculation
            eligibility_decay: Decay factor for eligibility traces (λ)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.lr = learning_rate
        self.decay = decay
        self.normalize = normalize
        self.reward_modulated = reward_modulated
        self.baseline = baseline
        self.eligibility_decay = eligibility_decay
        
        # Initialize weights with small random values
        self.weights = np.random.randn(out_features, in_features) * 0.01
        self.update_count = 0
        
        # Eligibility traces for credit assignment over time
        self.eligibility_trace = np.zeros_like(self.weights)
        
        # Statistics for reward learning
        self.total_reward = 0.0
        self.reward_count = 0
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: y = x @ W.T"""
        # Handle both [batch, in] and [in] shapes
        if x.ndim == 1:
            return self.weights @ x  # Single sample: [out] = [out, in] @ [in]
        else:
            return x @ self.weights.T  # Batch: [batch, out] = [batch, in] @ [in, out]
    
    def hebbian_update(self, pre: np.ndarray, post: np.ndarray, reward: Optional[float] = None):
        """
        Apply Hebbian weight update using Oja's rule with optional reward modulation.
        
        Args:
            pre: Pre-synaptic activations [batch, in_features] or [in_features]
            post: Post-synaptic activations [batch, out_features] or [out_features]
            reward: Optional reward signal for reward-modulated plasticity
        """
        # Handle both batch and single-sample inputs
        if pre.ndim == 1:
            pre = pre.reshape(1, -1)
        if post.ndim == 1:
            post = post.reshape(1, -1)
        
        batch_size = pre.shape[0]
        
        # Compute basic Hebbian term (outer product)
        outer = (post.T @ pre) / batch_size
        
        # Update eligibility trace (exponentially decaying pre*post correlation)
        self.eligibility_trace = self.eligibility_decay * self.eligibility_trace + outer
        
        if self.normalize:
            # Oja's rule: Δw = η * (post ⊗ pre - post² ⊗ w)
            # Normalization term: post² averaged over batch
            post_squared = ((post ** 2).T @ np.ones((batch_size, self.in_features))) / batch_size
            normalization = post_squared * self.weights
            
            if self.reward_modulated and reward is not None:
                # Reward-modulated Oja's rule: Δw = η * (R - b) * (eligibility_trace - normalization)
                advantage = reward - self.baseline
                delta_w = self.lr * advantage * (self.eligibility_trace - normalization)
                
                # Update baseline (exponential moving average of rewards)
                self.total_reward += reward
                self.reward_count += 1
                alpha = 0.1  # Baseline learning rate
                self.baseline = (1 - alpha) * self.baseline + alpha * reward
            else:
                # Standard Oja's rule
                delta_w = self.lr * (outer - normalization)
        else:
            # Standard Hebbian with decay: Δw = η * (post ⊗ pre - decay * w)
            decay_term = self.decay * self.weights
            
            if self.reward_modulated and reward is not None:
                # Reward-modulated: Δw = η * (R - b) * eligibility_trace
                advantage = reward - self.baseline
                delta_w = self.lr * advantage * self.eligibility_trace - decay_term
                
                # Update baseline
                self.total_reward += reward
                self.reward_count += 1
                alpha = 0.1
                self.baseline = (1 - alpha) * self.baseline + alpha * reward
            else:
                # Standard Hebbian
                delta_w = self.lr * (outer - decay_term)
        
        self.weights += delta_w
        self.update_count += 1
    
    def get_weights(self) -> np.ndarray:
        """Return current weight matrix."""
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray):
        """Set weight matrix."""
        assert weights.shape == self.weights.shape
        self.weights = weights.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        avg_reward = self.total_reward / self.reward_count if self.reward_count > 0 else 0.0
        return {
            'update_count': self.update_count,
            'reward_modulated': self.reward_modulated,
            'baseline': self.baseline,
            'avg_reward': avg_reward,
            'reward_count': self.reward_count,
            'weight_norm': np.linalg.norm(self.weights),
            'eligibility_norm': np.linalg.norm(self.eligibility_trace)
        }


# ===========================================================================
# PyTorch Implementation (GPU-compatible)
# ===========================================================================

if TORCH_AVAILABLE:
    class HebbianLayer(nn.Module):
        """
        PyTorch Hebbian learning layer.
        
        Compatible with standard PyTorch workflows and GPU acceleration.
        Can be mixed with gradient-based learning for hybrid training.
        """
        
        def __init__(self, in_features: int, out_features: int,
                     learning_rate: float = 0.01,
                     decay: float = 0.001,
                     normalize: bool = True,
                     use_bias: bool = False):
            """
            Args:
                in_features: Input dimension
                out_features: Output dimension
                learning_rate: Hebbian learning rate
                decay: Weight decay factor
                normalize: Use Oja's normalization
                use_bias: Include bias term (not standard for Hebbian)
            """
            super().__init__()
            
            self.in_features = in_features
            self.out_features = out_features
            self.lr = learning_rate
            self.decay = decay
            self.normalize = normalize
            
            # Weights are Parameters for PyTorch compatibility
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features) * 0.01
            )
            
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
            
            self.update_count = 0
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the layer."""
            return F.linear(x, self.weight, self.bias)
        
        @torch.no_grad()
        def hebbian_update(self, pre: torch.Tensor, post: torch.Tensor):
            """
            Apply Hebbian weight update.
            
            Args:
                pre: Pre-synaptic activations [batch, in_features]
                post: Post-synaptic activations [batch, out_features]
            """
            batch_size = pre.shape[0]
            if batch_size == 0:
                return
            
            if self.normalize:
                # Oja's rule
                outer = torch.einsum("bi,bj->ij", post, pre) / batch_size
                post_squared = (post ** 2).sum(dim=0, keepdim=True).T / batch_size
                normalization = post_squared * self.weight
                delta_w = self.lr * (outer - normalization)
            else:
                # Standard Hebbian with decay
                outer = torch.einsum("bi,bj->ij", post, pre) / batch_size
                decay_term = self.decay * self.weight
                delta_w = self.lr * (outer - decay_term)
            
            self.weight.data += delta_w
            self.update_count += 1
        
        def extra_repr(self) -> str:
            return f'in_features={self.in_features}, out_features={self.out_features}, lr={self.lr}'
else:
    # Stub for when PyTorch is not available
    HebbianLayer = None


# ===========================================================================
# VSA-Integrated Hebbian Learning
# ===========================================================================

class VSAHebbianLearner:
    """
    Hebbian learning integrated with Vector Symbolic Architecture.
    
    Learns associations between hypervectors through co-activation,
    enabling unsupervised concept formation from sensory input.
    
    Supports reward-modulated plasticity for reinforcement learning.
    """
    
    def __init__(self, dimension: int = 10240, n_concepts: int = 100,
                 learning_rate: float = 0.1,
                 reward_modulated: bool = False,
                 baseline: float = 0.0):
        """
        Args:
            dimension: Hypervector dimensionality
            n_concepts: Number of concept slots
            learning_rate: Learning rate for association strength
            reward_modulated: Enable reward-modulated plasticity
            baseline: Reward baseline for advantage calculation
        """
        self.dimension = dimension
        self.n_concepts = n_concepts
        self.lr = learning_rate
        self.reward_modulated = reward_modulated
        self.baseline = baseline
        
        # Association matrix: concept_i → concept_j activation strength
        self.associations = np.zeros((n_concepts, n_concepts), dtype=np.float32)
        
        # Eligibility traces for credit assignment
        self.eligibility = np.zeros((n_concepts, n_concepts), dtype=np.float32)
        self.eligibility_decay = 0.9
        
        # Concept usage tracking
        self.activation_counts = np.zeros(n_concepts, dtype=np.int32)
        
        # Reward statistics
        self.total_reward = 0.0
        self.reward_count = 0
        self.update_count = 0
    
    def update_associations(self, active_concepts: List[int], reward: Optional[float] = None):
        """
        Update association strengths between co-activated concepts.
        
        Args:
            active_concepts: List of concept IDs that are currently active
            reward: Optional reward signal for reward-modulated learning
        """
        if len(active_concepts) < 1:
            return  # Need at least 1 concept
        
        # Decay eligibility traces
        self.eligibility *= self.eligibility_decay
        
        # Update all pairwise associations
        for i in active_concepts:
            for j in active_concepts:
                if i < self.n_concepts and j < self.n_concepts:
                    # Update eligibility trace (pre * post correlation)
                    self.eligibility[i, j] += 1.0
                    
                    if self.reward_modulated and reward is not None:
                        # Reward-modulated update: Δw = η * (R - b) * eligibility
                        advantage = reward - self.baseline
                        delta = self.lr * advantage * self.eligibility[i, j]
                        self.associations[i, j] = np.clip(
                            self.associations[i, j] + delta,
                            0.0, 1.0
                        )
                    elif not self.reward_modulated:
                        # Standard Hebbian update (only if not waiting for reward)
                        if i != j:
                            # Strengthen connection
                            self.associations[i, j] += self.lr * (1.0 - self.associations[i, j])
                    
            # Update activation count
            if i < self.n_concepts:
                self.activation_counts[i] += 1
        
        # Update baseline if reward provided
        if self.reward_modulated and reward is not None:
            self.total_reward += reward
            self.reward_count += 1
            alpha = 0.1  # Baseline learning rate
            self.baseline = (1 - alpha) * self.baseline + alpha * reward
        
        self.update_count += 1
    
    def spread_activation(self, seed_concepts: List[int],
                         n_steps: int = 3,
                         threshold: float = 0.3) -> Dict[int, float]:
        """
        Spread activation through learned associations.
        
        Args:
            seed_concepts: Initial concept activations
            n_steps: Number of spreading iterations
            threshold: Minimum activation to propagate
            
        Returns:
            Dictionary of concept_id → activation_strength
        """
        activations = {cid: 1.0 for cid in seed_concepts if cid < self.n_concepts}
        
        for step in range(n_steps):
            new_activations = activations.copy()
            
            for cid, strength in list(activations.items()):
                if strength < threshold:
                    continue
                
                # Spread to associated concepts
                for target_cid in range(self.n_concepts):
                    if target_cid == cid:
                        continue
                    
                    association_strength = self.associations[cid, target_cid]
                    if association_strength > threshold:
                        # Propagate activation proportional to association strength
                        propagated = strength * association_strength * 0.5  # Decay factor
                        new_activations[target_cid] = max(
                            new_activations.get(target_cid, 0.0),
                            propagated
                        )
            
            activations = new_activations
        
        # Filter by threshold
        return {cid: strength for cid, strength in activations.items() if strength > threshold}
    
    def get_associated_concepts(self, concept_id: int,
                               threshold: float = 0.5,
                               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get concepts most strongly associated with a given concept.
        
        Args:
            concept_id: Query concept ID
            threshold: Minimum association strength
            top_k: Maximum number of results
            
        Returns:
            List of (concept_id, strength) tuples
        """
        if concept_id >= self.n_concepts:
            return []
        
        associations = self.associations[concept_id, :]
        strong_indices = np.where(associations > threshold)[0]
        
        if len(strong_indices) == 0:
            return []
        
        # Sort by strength
        sorted_indices = strong_indices[np.argsort(-associations[strong_indices])]
        top_indices = sorted_indices[:top_k]
        
        return [(int(idx), float(associations[idx])) for idx in top_indices]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        avg_reward = self.total_reward / self.reward_count if self.reward_count > 0 else 0.0
        n_strong_associations = np.sum(self.associations > 0.5)
        avg_association_strength = np.mean(self.associations[self.associations > 0.1])
        
        return {
            'update_count': self.update_count,
            'reward_modulated': self.reward_modulated,
            'baseline': self.baseline,
            'avg_reward': avg_reward,
            'reward_count': self.reward_count,
            'n_strong_associations': int(n_strong_associations),
            'avg_association_strength': float(avg_association_strength) if n_strong_associations > 0 else 0.0,
            'eligibility_norm': float(np.linalg.norm(self.eligibility))
        }


# ===========================================================================
# Factory Functions
# ===========================================================================

def create_hebbian_layer(in_features: int, out_features: int,
                        backend: str = "auto",
                        **kwargs) -> Any:
    """
    Factory function to create a Hebbian learning layer.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        backend: "numpy", "torch", or "auto" (choose based on availability)
        **kwargs: Additional arguments passed to layer constructor
        
    Returns:
        Hebbian learning layer (NumPy or PyTorch)
    """
    if backend == "auto":
        backend = "torch" if TORCH_AVAILABLE else "numpy"
    
    if backend == "torch":
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to NumPy")
            return HebbianMatrixNumPy(in_features, out_features, **kwargs)
        return HebbianLayer(in_features, out_features, **kwargs)
    
    elif backend == "numpy":
        return HebbianMatrixNumPy(in_features, out_features, **kwargs)
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ===========================================================================
# Module Exports
# ===========================================================================

__all__ = [
    "HebbianMatrixNumPy",
    "HebbianLayer",
    "VSAHebbianLearner",
    "create_hebbian_layer",
]
