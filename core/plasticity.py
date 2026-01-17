"""
Plasticity: Learning rules and neuromodulation.

Separates learning mechanisms from neural computation.
Three independent plasticity systems can be composed:
1. STDP: Spike-timing-dependent plasticity
2. Neuromodulation: Reward/dopamine-based learning gating
3. Homeostasis: Synaptic scaling to maintain target firing rates
"""

from typing import Dict
from .synapse import Synapse


class STDPRule:
    """
    Spike-Timing-Dependent Plasticity.

    Updates synaptic weights based on the relative timing of pre and post spikes.
    Implemented as 3-factor learning: weight_change ∝ trace × postfire × dopamine

    Invariants:
    - Learning only occurs when postsynaptic neuron fires
    - Learning requires non-zero neuromodulatory signal
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize STDP rule.

        Args:
            learning_rate: How fast weights change (typical: 0.01-0.1)
        """
        self.learning_rate = learning_rate

    def apply(self, synapse: Synapse, post_fired: bool, dopamine: float) -> float:
        """
        Apply STDP to a single synapse.

        Returns:
            Delta weight applied
        """
        if not post_fired or dopamine == 0:
            return 0.0

        delta_w = self.learning_rate * synapse.trace * dopamine
        synapse.weight += delta_w
        synapse.weight = max(0.0, min(2.0, synapse.weight))

        return delta_w


class HomeostasisRule:
    """
    Homeostatic Synaptic Scaling.

    Multiplicative scaling of all incoming weights to maintain target firing rate.
    If neuron fires too much, scale down all weights. If too silent, scale up.

    Biological basis: BCM theory (Bienenstock-Cooper-Munro)
    """

    def __init__(self, target_rate: float = 0.25, adaptation_rate: float = 0.005):
        """
        Initialize homeostasis rule.

        Args:
            target_rate: Target firing rate (0.0-1.0, typically 0.25)
            adaptation_rate: Speed of homeostatic adjustment
        """
        self.target_rate = target_rate
        self.adaptation_rate = adaptation_rate

    def apply(self, synapses: Dict[int, Synapse], current_rate: float, node_type: str) -> None:
        """
        Apply homeostatic scaling to all synapses.

        Only applies to non-input neurons (input neurons don't have firing rate targets).

        Args:
            synapses: Dict of incoming synapses
            current_rate: Exponential moving average firing rate
            node_type: "input", "hidden", or "output"
        """
        if node_type == "input":
            return  # Input neurons don't need homeostasis

        # Compute scaling factor
        rate_diff = current_rate - self.target_rate
        scaling_factor = 1.0 - (rate_diff * self.adaptation_rate)

        # Apply to all incoming synapses
        for synapse in synapses.values():
            synapse.weight *= scaling_factor

            # Clamp to safe ranges (prevent complete silencing)
            if node_type == "hidden":
                synapse.weight = max(0.1, min(2.0, synapse.weight))
            else:  # output
                synapse.weight = max(0.1, min(3.0, synapse.weight))


class LearningGate:
    """
    Gating mechanism to enable/disable learning.

    Allows learning to be turned on/off without modifying computation.
    Separates inference (always on) from learning (conditional).

    Invariant: Disabling learning does not change network behavior during inference.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize learning gate.

        Args:
            enabled: Whether learning is active
        """
        self.enabled = enabled

    def is_learning_active(self) -> bool:
        """Check if learning is currently enabled."""
        return self.enabled

    def enable(self) -> None:
        """Turn on learning."""
        self.enabled = True

    def disable(self) -> None:
        """Turn off learning."""
        self.enabled = False

    def toggle(self) -> None:
        """Toggle learning state."""
        self.enabled = not self.enabled


class AdaptivePlasticity:
    """
    Adaptive learning-rate scheduler for plasticity.

    Computes an effective learning rate based on performance and stability signals.
    If no signals are provided, defaults to base rate.
    """

    def __init__(self, base_rate: float = 0.01, min_rate: float = 0.0001, max_rate: float = 0.5):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate

    def rate_scheduler(self, step: int, perf_delta: float = 0.0, stability: float = 0.0) -> float:
        """
        Compute an adapted learning rate.

        Args:
            step: Global step or epoch
            perf_delta: Performance change (-1..+1)
            stability: Stability index (0..1); higher means more stable
        Returns:
            Effective learning rate
        """
        # Simple scheduler: encourage learning when improving, reduce when unstable
        factor = (1.0 + (perf_delta * 0.1)) * (1.0 - (stability * 0.2))
        rate = self.base_rate * max(0.5, min(2.0, factor))
        return max(self.min_rate, min(self.max_rate, rate))


class PlasticityController:
    """
    Unified plasticity system composing STDP, homeostasis, and gating.

    This module owns all learning behavior. Neurons and synapses call this
    to update weights, never directly modifying weights themselves.

    Public Interface:
    - update_learning(synapse, pre_fired, post_fired, dopamine)
    - update_homeostasis(synapses, firing_rate, node_type)
    - disable/enable learning
    """

    def __init__(
        self,
        stdp_learning_rate: float = 0.01,
        target_firing_rate: float = 0.25,
        homeostasis_rate: float = 0.005,
        learning_enabled: bool = True
    ):
        """
        Initialize plasticity controller.

        Args:
            stdp_learning_rate: STDP learning rate
            target_firing_rate: Target for homeostatic regulation
            homeostasis_rate: Speed of homeostatic adaptation
            learning_enabled: Whether learning is active
        """
        self.stdp = STDPRule(stdp_learning_rate)
        self.homeostasis = HomeostasisRule(target_firing_rate, homeostasis_rate)
        self.gate = LearningGate(learning_enabled)
        # Adaptive plasticity and region multiplier (optional)
        self._adaptive = AdaptivePlasticity(base_rate=stdp_learning_rate)
        self._region_lr_multiplier: float = 1.0
        self._external_scheduler = None  # Optional callable(step, perf_delta, stability) -> rate

    def set_region_lr_multiplier(self, multiplier: float) -> None:
        """Set per-region learning rate multiplier (>=0)."""
        self._region_lr_multiplier = max(0.0, multiplier)

    def set_rate_scheduler(self, scheduler_callable) -> None:
        """
        Provide an external scheduler function: f(step, perf_delta, stability) -> rate.
        If None, internal AdaptivePlasticity is used.
        """
        self._external_scheduler = scheduler_callable

    def get_effective_lr(self, step: int, perf_delta: float = 0.0, stability: float = 0.0) -> float:
        """Compute effective learning rate combining scheduler and region multiplier."""
        base = (
            self._external_scheduler(step, perf_delta, stability)
            if callable(self._external_scheduler)
            else self._adaptive.rate_scheduler(step, perf_delta, stability)
        )
        return max(0.0, min(self._adaptive.max_rate, base * self._region_lr_multiplier))

    def update_learning(
        self,
        synapse: Synapse,
        post_fired: bool,
        dopamine: float,
        step: int = 0,
        perf_delta: float = 0.0,
        stability: float = 0.0
    ) -> None:
        """
        Apply STDP learning to a synapse, using effective learning rate if enabled.

        Learning only occurs if gate is enabled.

        Args:
            synapse: Synapse to update
            post_fired: Did postsynaptic neuron fire?
            dopamine: Reward signal
        """
        if not self.gate.is_learning_active():
            return
        # Temporarily adjust STDP learning rate
        original_lr = self.stdp.learning_rate
        self.stdp.learning_rate = self.get_effective_lr(step, perf_delta, stability)
        try:
            self.stdp.apply(synapse, post_fired, dopamine)
        finally:
            # Restore original base rate to avoid leaking state across calls
            self.stdp.learning_rate = original_lr

    def update_homeostasis(
        self,
        synapses: Dict[int, Synapse],
        firing_rate: float,
        node_type: str
    ) -> None:
        """
        Apply homeostatic scaling to synapses.

        Homeostasis is always active (independent of learning gate).
        This maintains network stability and prevents saturation.

        Args:
            synapses: Dict of incoming synapses
            firing_rate: Current firing rate (EMA)
            node_type: Neuron type ("input", "hidden", "output")
        """
        # Homeostasis is always active for stability
        self.homeostasis.apply(synapses, firing_rate, node_type)

    def disable_learning(self) -> None:
        """Disable learning (keep inference running)."""
        self.gate.disable()

    def enable_learning(self) -> None:
        """Enable learning."""
        self.gate.enable()

    def is_learning(self) -> bool:
        """Check if learning is active."""
        return self.gate.is_learning_active()

    def __repr__(self):
        status = "ON" if self.gate.is_learning_active() else "OFF"
        return f"PlasticityController(learning={status})"
