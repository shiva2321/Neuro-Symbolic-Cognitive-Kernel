"""
Spike-Timing-Dependent Plasticity (STDP)
Implements biologically-inspired local learning rules for spiking networks.

STDP Rule:
- If pre-synaptic neuron fires before post-synaptic: strengthen synapse (LTP)
- If pre-synaptic neuron fires after post-synaptic: weaken synapse (LTD)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STDPLearning:
    """
    Spike-Timing-Dependent Plasticity learning rule.

    Weight update:
        Δw = A_+ * exp(-Δt/τ_+)  if Δt > 0 (LTP)
        Δw = -A_- * exp(Δt/τ_-)  if Δt < 0 (LTD)

    Where Δt = t_post - t_pre
    """

    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0,
                 A_plus: float = 0.01, A_minus: float = 0.01,
                 w_min: float = 0.0, w_max: float = 1.0):
        """
        Initialize STDP learning rule.

        Args:
            tau_plus: Time constant for LTP (ms)
            tau_minus: Time constant for LTD (ms)
            A_plus: Maximum weight increase
            A_minus: Maximum weight decrease
            w_min: Minimum weight value
            w_max: Maximum weight value
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max

        # Trace variables for efficient STDP
        self.pre_trace = None
        self.post_trace = None

    def reset_traces(self, shape: Tuple, device: str = 'cpu'):
        """
        Reset synaptic traces.

        Args:
            shape: Shape of weight matrix
            device: Computation device
        """
        self.pre_trace = torch.zeros(shape[0], device=device)
        self.post_trace = torch.zeros(shape[1], device=device)

    def update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        Update synaptic traces based on spike activity.

        Args:
            pre_spikes: Pre-synaptic spikes (batch_size, n_pre)
            post_spikes: Post-synaptic spikes (batch_size, n_post)
        """
        # Exponential decay
        alpha_plus = np.exp(-1.0 / self.tau_plus)
        alpha_minus = np.exp(-1.0 / self.tau_minus)

        # Update traces (sum over batch)
        self.pre_trace = alpha_plus * self.pre_trace + pre_spikes.sum(dim=0)
        self.post_trace = alpha_minus * self.post_trace + post_spikes.sum(dim=0)

    def compute_weight_update(self, weights: torch.Tensor,
                            pre_spikes: torch.Tensor,
                            post_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute weight updates using STDP rule.

        Args:
            weights: Current weights (n_pre, n_post)
            pre_spikes: Pre-synaptic spikes (batch_size, n_pre)
            post_spikes: Post-synaptic spikes (batch_size, n_post)

        Returns:
            Weight updates (n_pre, n_post)
        """
        device = weights.device

        # Initialize traces if needed
        if self.pre_trace is None:
            self.reset_traces(weights.shape, device)

        # Update traces
        self.update_traces(pre_spikes, post_spikes)

        # Compute weight changes
        # LTP: post spike causes pre_trace to strengthen
        # LTD: pre spike causes post_trace to weaken

        # Average over batch
        pre_sum = pre_spikes.sum(dim=0)
        post_sum = post_spikes.sum(dim=0)

        # Outer products for weight updates
        dw_ltp = self.A_plus * torch.outer(self.pre_trace, post_sum)
        dw_ltd = -self.A_minus * torch.outer(pre_sum, self.post_trace)

        dw = dw_ltp + dw_ltd

        # Apply weight bounds
        new_weights = torch.clamp(weights + dw, self.w_min, self.w_max)

        return new_weights - weights

    def apply_update(self, weights: nn.Parameter,
                    pre_spikes: torch.Tensor,
                    post_spikes: torch.Tensor):
        """
        Apply STDP update to weight parameter.

        Args:
            weights: Weight parameter to update
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes
        """
        with torch.no_grad():
            dw = self.compute_weight_update(weights.data, pre_spikes, post_spikes)
            weights.data += dw


class TripleSTDP:
    """
    Triplet STDP - considers triplets of spikes for more accurate learning.
    """

    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0,
                 tau_x: float = 15.0, tau_y: float = 15.0,
                 A2_plus: float = 0.01, A2_minus: float = 0.01,
                 A3_plus: float = 0.001, A3_minus: float = 0.001):
        """
        Initialize triplet STDP.

        Args:
            tau_plus, tau_minus: Pairwise STDP time constants
            tau_x, tau_y: Triplet time constants
            A2_plus, A2_minus: Pairwise learning rates
            A3_plus, A3_minus: Triplet learning rates
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.A2_plus = A2_plus
        self.A2_minus = A2_minus
        self.A3_plus = A3_plus
        self.A3_minus = A3_minus

        # Traces
        self.r1 = None  # Fast pre trace
        self.r2 = None  # Slow pre trace
        self.o1 = None  # Fast post trace
        self.o2 = None  # Slow post trace

    def reset_traces(self, n_pre: int, n_post: int, device: str = 'cpu'):
        """Reset all traces"""
        self.r1 = torch.zeros(n_pre, device=device)
        self.r2 = torch.zeros(n_pre, device=device)
        self.o1 = torch.zeros(n_post, device=device)
        self.o2 = torch.zeros(n_post, device=device)

    def compute_weight_update(self, weights: torch.Tensor,
                            pre_spikes: torch.Tensor,
                            post_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute weight updates using triplet STDP.

        Args:
            weights: Current weights
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes

        Returns:
            Weight updates
        """
        device = weights.device
        n_pre, n_post = weights.shape

        if self.r1 is None:
            self.reset_traces(n_pre, n_post, device)

        # Decay factors
        alpha_plus = np.exp(-1.0 / self.tau_plus)
        alpha_minus = np.exp(-1.0 / self.tau_minus)
        alpha_x = np.exp(-1.0 / self.tau_x)
        alpha_y = np.exp(-1.0 / self.tau_y)

        # Sum spikes over batch
        pre_sum = pre_spikes.sum(dim=0)
        post_sum = post_spikes.sum(dim=0)

        # Compute weight changes before updating traces
        # Pairwise terms
        dw_pair_ltp = self.A2_plus * torch.outer(self.r1, post_sum)
        dw_pair_ltd = -self.A2_minus * torch.outer(pre_sum, self.o1)

        # Triplet terms
        dw_triplet_ltp = self.A3_plus * torch.outer(self.r2, post_sum) * self.o1.unsqueeze(0)
        dw_triplet_ltd = -self.A3_minus * torch.outer(pre_sum, self.o2) * self.r1.unsqueeze(1)

        dw = dw_pair_ltp + dw_pair_ltd + dw_triplet_ltp + dw_triplet_ltd

        # Update traces
        self.r1 = alpha_plus * self.r1 + pre_sum
        self.r2 = alpha_x * self.r2 + pre_sum
        self.o1 = alpha_minus * self.o1 + post_sum
        self.o2 = alpha_y * self.o2 + post_sum

        return dw


class RewardModulatedSTDP:
    """
    Reward-modulated STDP (R-STDP) for reinforcement learning.
    Weight updates are gated by a global reward signal.
    """

    def __init__(self, base_stdp: STDPLearning, tau_reward: float = 100.0):
        """
        Initialize R-STDP.

        Args:
            base_stdp: Base STDP learning rule
            tau_reward: Time constant for reward signal
        """
        self.base_stdp = base_stdp
        self.tau_reward = tau_reward

        # Eligibility trace
        self.eligibility = None
        self.reward_signal = 0.0

    def reset(self, shape: Tuple, device: str = 'cpu'):
        """Reset eligibility trace"""
        self.eligibility = torch.zeros(shape, device=device)
        self.reward_signal = 0.0

    def compute_weight_update(self, weights: torch.Tensor,
                            pre_spikes: torch.Tensor,
                            post_spikes: torch.Tensor,
                            reward: float) -> torch.Tensor:
        """
        Compute reward-modulated weight updates.

        Args:
            weights: Current weights
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes
            reward: Reward signal

        Returns:
            Weight updates
        """
        device = weights.device

        if self.eligibility is None:
            self.reset(weights.shape, device)

        # Compute base STDP update
        dw_stdp = self.base_stdp.compute_weight_update(weights, pre_spikes, post_spikes)

        # Update eligibility trace
        alpha_reward = np.exp(-1.0 / self.tau_reward)
        self.eligibility = alpha_reward * self.eligibility + dw_stdp

        # Modulate by reward
        dw = reward * self.eligibility

        return dw


class HomeostaticSTDP(STDPLearning):
    """
    STDP with homeostatic regulation to maintain stable firing rates.
    """

    def __init__(self, target_rate: float = 0.1, homeostatic_strength: float = 0.001,
                 **kwargs):
        """
        Initialize homeostatic STDP.

        Args:
            target_rate: Target firing rate
            homeostatic_strength: Strength of homeostatic regulation
            **kwargs: Arguments for base STDP
        """
        super().__init__(**kwargs)

        self.target_rate = target_rate
        self.homeostatic_strength = homeostatic_strength

        # Running average of firing rates
        self.avg_pre_rate = None
        self.avg_post_rate = None
        self.alpha = 0.99  # Exponential moving average factor

    def compute_weight_update(self, weights: torch.Tensor,
                            pre_spikes: torch.Tensor,
                            post_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute weight updates with homeostatic regulation.

        Args:
            weights: Current weights
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes

        Returns:
            Weight updates
        """
        device = weights.device

        # Compute base STDP update
        dw_stdp = super().compute_weight_update(weights, pre_spikes, post_spikes)

        # Compute firing rates
        current_pre_rate = pre_spikes.mean(dim=0)
        current_post_rate = post_spikes.mean(dim=0)

        # Update running averages
        if self.avg_pre_rate is None:
            self.avg_pre_rate = current_pre_rate
            self.avg_post_rate = current_post_rate
        else:
            self.avg_pre_rate = self.alpha * self.avg_pre_rate + (1 - self.alpha) * current_pre_rate
            self.avg_post_rate = self.alpha * self.avg_post_rate + (1 - self.alpha) * current_post_rate

        # Homeostatic regulation
        # If post rate too high, decrease incoming weights
        # If post rate too low, increase incoming weights
        rate_error = self.avg_post_rate - self.target_rate
        dw_homeostatic = -self.homeostatic_strength * rate_error.unsqueeze(0)

        # Combine STDP and homeostatic updates
        dw = dw_stdp + dw_homeostatic

        return dw


class WeightNormalization:
    """
    Weight normalization to prevent runaway potentiation.
    """

    @staticmethod
    def normalize_weights(weights: torch.Tensor, method: str = 'l2',
                         norm_value: float = 1.0) -> torch.Tensor:
        """
        Normalize weights.

        Args:
            weights: Weight matrix (n_pre, n_post)
            method: Normalization method ('l1', 'l2', 'sum')
            norm_value: Target norm value

        Returns:
            Normalized weights
        """
        if method == 'l1':
            # L1 normalization (sum of absolute values)
            norms = weights.abs().sum(dim=0, keepdim=True) + 1e-8
            return weights * (norm_value / norms)

        elif method == 'l2':
            # L2 normalization (Euclidean norm)
            norms = torch.sqrt((weights ** 2).sum(dim=0, keepdim=True)) + 1e-8
            return weights * (norm_value / norms)

        elif method == 'sum':
            # Sum normalization
            sums = weights.sum(dim=0, keepdim=True) + 1e-8
            return weights * (norm_value / sums)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def soft_bounds(weights: torch.Tensor, w_min: float = 0.0,
                   w_max: float = 1.0, sharpness: float = 10.0) -> torch.Tensor:
        """
        Apply soft bounds using sigmoid function.

        Args:
            weights: Weight matrix
            w_min: Minimum weight
            w_max: Maximum weight
            sharpness: Sharpness of sigmoid

        Returns:
            Bounded weights
        """
        # Map to [w_min, w_max] using sigmoid
        normalized = torch.sigmoid(sharpness * (weights - 0.5))
        return w_min + (w_max - w_min) * normalized


if __name__ == "__main__":
    # Test STDP learning
    logger.info("Testing STDP Learning...")

    # Create weight matrix
    n_pre, n_post = 100, 50
    weights = torch.rand(n_pre, n_post) * 0.5

    # Initialize STDP
    stdp = STDPLearning(tau_plus=20.0, tau_minus=20.0,
                       A_plus=0.01, A_minus=0.01)

    # Simulate spike activity
    batch_size = 10
    num_steps = 100

    weight_history = [weights.clone()]

    for step in range(num_steps):
        # Generate random spikes
        pre_spikes = (torch.rand(batch_size, n_pre) < 0.1).float()
        post_spikes = (torch.rand(batch_size, n_post) < 0.1).float()

        # Compute weight update
        dw = stdp.compute_weight_update(weights, pre_spikes, post_spikes)
        weights = weights + dw

        if step % 20 == 0:
            weight_history.append(weights.clone())

    print(f"Initial weight mean: {weight_history[0].mean():.4f}")
    print(f"Final weight mean: {weight_history[-1].mean():.4f}")
    print(f"Weight change: {(weight_history[-1] - weight_history[0]).abs().mean():.4f}")

    # Test homeostatic STDP
    logger.info("\nTesting Homeostatic STDP...")

    h_stdp = HomeostaticSTDP(target_rate=0.1, homeostatic_strength=0.001)

    weights_h = torch.rand(n_pre, n_post) * 0.5

    for step in range(num_steps):
        pre_spikes = (torch.rand(batch_size, n_pre) < 0.15).float()
        post_spikes = (torch.rand(batch_size, n_post) < 0.15).float()

        dw = h_stdp.compute_weight_update(weights_h, pre_spikes, post_spikes)
        weights_h = weights_h + dw

    print(f"Homeostatic STDP - Average post rate: {h_stdp.avg_post_rate.mean():.4f}")
    print(f"Target rate: {h_stdp.target_rate}")

    logger.info("\nSTDP tests complete!")

