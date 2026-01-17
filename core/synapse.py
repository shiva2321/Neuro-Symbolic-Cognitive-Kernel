"""
Synapse: Biologically-inspired synaptic connection with plasticity support.

Represents a connection between two neurons with:
- Weight (synaptic strength)
- Eligibility trace (memory of presynaptic activity for delayed learning)
- Type classification (excitatory/inhibitory)

Core responsibility: Weight the spike and pass it to postsynaptic neuron.
Learning rules are applied externally by plasticity modules.
"""

from .spike import Spike, WeightedSpike


class Synapse:
    """
    Biologically-inspired synaptic connection.

    Maintains synaptic weight and eligibility trace.
    Transmits incoming spikes to postsynaptic targets.

    Invariants:
    - Weight bounds: [0, 2.0] (non-negative, prevents explosion)
    - Trace always >= 0
    - Type is immutable (excitatory XOR inhibitory)
    """

    def __init__(self, weight: float, is_inhibitory: bool = False):
        """
        Initialize a synapse.

        Args:
            weight: Initial synaptic strength (typical: 0.1-1.5)
            is_inhibitory: If True, this is an inhibitory synapse
        """
        self.weight = max(0.0, min(2.0, weight))  # Clamp to safe range
        self.is_inhibitory = is_inhibitory

        # Eligibility trace: memory of recent presynaptic activity
        self.trace = 0.0

        # STDP parameters (biological timescales)
        self.trace_decay = 0.9  # Exponential decay of trace per timestep
        self.trace_increment = 1.0  # How much trace increases per presynaptic spike

    def transmit(self, spike: Spike) -> WeightedSpike:
        """
        Transmit a presynaptic spike through this synapse.

        Applies synaptic weight to the spike and marks it as inhibitory if needed.

        Args:
            spike: The incoming spike from presynaptic neuron

        Returns:
            WeightedSpike with transmission parameters applied
        """
        return WeightedSpike(
            spike=spike,
            weight=self.weight,
            is_inhibitory=self.is_inhibitory
        )

    def update_trace(self, pre_fired: bool) -> None:
        """
        Update eligibility trace based on presynaptic activity.

        The trace acts as a "memory" of recent presynaptic spikes.
        It is used by learning rules to correlate pre and post activity.

        Biological basis: calcium influx through synaptic receptors

        Args:
            pre_fired: Did the presynaptic neuron fire this timestep?
        """
        # Exponential decay
        self.trace *= self.trace_decay

        # Add new trace if presynaptic neuron fired
        if pre_fired:
            self.trace += self.trace_increment

    def apply_stdp(self, post_fired: bool, dopamine: float, learning_rate: float = 0.01) -> None:
        """
        Apply 3-Factor STDP learning rule.

        Updates weight based on:
        - Eligibility trace (presynaptic timing)
        - Postsynaptic firing (postsynaptic timing)
        - Dopamine (reward signal)

        Learning rule:
            ΔWeight = LearningRate × Trace × Dopamine

        Args:
            post_fired: Did the postsynaptic neuron fire?
            dopamine: Reward signal (positive=strengthening, negative=weakening)
            learning_rate: Learning rate (typical: 0.01-0.1)
        """
        if post_fired and dopamine != 0:
            # Only learn when postsynaptic neuron fires AND dopamine is present
            delta_w = learning_rate * self.trace * dopamine
            self.weight += delta_w

            # Clamp weight to safe bounds
            self.weight = max(0.0, min(2.0, self.weight))

    def get_current(self, pre_fired: bool) -> float:
        """
        Calculate the synaptic current this synapse contributes.

        DEPRECATED: Use transmit() and let postsynaptic neuron handle integration.
        Kept for backward compatibility with old code.

        Args:
            pre_fired: Is presynaptic neuron currently firing?

        Returns:
            Current contribution (can be negative for inhibitory)
        """
        if not pre_fired:
            return 0.0

        return self.weight if not self.is_inhibitory else -self.weight

    def get_output(self) -> float:
        """
        DEPRECATED: Use transmit() instead.

        For backward compatibility, returns weight.
        """
        return self.weight

    def __repr__(self):
        typ = "INH" if self.is_inhibitory else "EXC"
        return f"Synapse({typ}, w={self.weight:.3f}, tr={self.trace:.3f})"

