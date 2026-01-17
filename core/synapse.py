"""
Synapse: Represents a biological synapse connection with eligibility trace.
NO matrix multiplication - each synapse is an independent object.
"""

class Synapse:
    def __init__(self, weight, is_inhibitory=False):
        """
        Args:
            weight: Initial synaptic strength
            is_inhibitory: If True, implements shunting inhibition
        """
        self.weight = weight
        self.trace = 0.0  # Eligibility trace for delayed learning
        self.is_inhibitory = is_inhibitory

        # STDP parameters
        self.trace_decay = 0.9  # How quickly the trace fades
        self.trace_increment = 1.0  # How much trace is added per spike

    def update_trace(self, pre_fired):
        """
        Update the eligibility trace based on presynaptic activity.
        The trace acts as a "memory" of recent activity.
        """
        # Decay existing trace
        self.trace *= self.trace_decay

        # Add new trace if presynaptic neuron fired
        if pre_fired:
            self.trace += self.trace_increment

    def apply_stdp(self, post_fired, dopamine, learning_rate=0.01):
        """
        3-Factor STDP: Weight update based on pre-trace, post-fire, and dopamine.

        Args:
            post_fired: Did the postsynaptic neuron fire?
            dopamine: Global reward signal (positive or negative)
            learning_rate: How fast weights change
        """
        if post_fired and dopamine != 0:
            # Delta_Weight = LR * Trace * Dopamine
            # Positive dopamine strengthens, negative weakens
            delta_w = learning_rate * self.trace * dopamine
            self.weight += delta_w

            # Keep weights in reasonable bounds
            self.weight = max(0.0, min(2.0, self.weight))

    def get_current(self, pre_fired):
        """
        Calculate the current contributed by this synapse.

        Args:
            pre_fired: Is the presynaptic neuron currently firing?

        Returns:
            Current contribution (can be negative for inhibitory)
        """
        if not pre_fired:
            return 0.0

        return self.weight if not self.is_inhibitory else -self.weight

