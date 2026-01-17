"""
BioNode v2.0: A biologically-inspired neuron with membrane potential,
threshold firing, refractory period, shunting inhibition, and HOMEOSTATIC PLASTICITY.
NO backpropagation - learning via local STDP rules + synaptic scaling.
"""

from .synapse import Synapse


class BioNode:
    def __init__(self, node_id, node_type="hidden", threshold=1.0, refractory_period=3):
        """
        Args:
            node_id: Unique identifier
            node_type: "input", "hidden", or "output"
            threshold: Voltage needed to fire
            refractory_period: Ticks during which neuron cannot fire after spiking
        """
        self.id = node_id
        self.type = node_type
        self.threshold = threshold
        self.refractory_period = refractory_period

        # State variables
        self.potential = 0.0  # Membrane potential (voltage)
        self.last_spike_tick = -999  # When did this neuron last fire?
        self.is_firing = False  # Current spike state
        self.external_current = 0.0  # External input (sensors or teacher forcing)

        # HOMEOSTASIS VARIABLES (NEW!)
        self.avg_firing_rate = 0.0  # Running average of firing activity
        self.target_rate = 0.25     # Target firing rate (fire ~25% of time for selectivity)
        self.homeostasis_rate = 0.005  # Speed of homeostatic adjustment

        # Graph-based connectivity (NO matrices!)
        self.inputs = {}  # Dict[source_id, Synapse]

        # Biological parameters
        self.decay_rate = 0.8  # Leak: potential decays toward 0
        self.resting_potential = 0.0
        self.spike_value = 1.0  # Output when firing

    def add_input(self, source_id, weight, is_inhibitory=False):
        """
        Create a synapse from source_id to this node.
        Uses adjacency list (dictionary) for sparse connectivity.
        """
        self.inputs[source_id] = Synapse(weight, is_inhibitory)

    def set_external_input(self, value):
        """
        Set external current for this node.
        Stored separately and applied during tick to avoid decay issues.
        """
        self.external_current = value

    def tick(self, current_tick, network_state, global_dopamine=0.0, learning_rate=0.01):
        """
        One timestep of the neuron's lifecycle with HOMEOSTATIC PLASTICITY.

        Args:
            current_tick: Current simulation time
            network_state: Dict[node_id, is_firing] - state of all neurons
            global_dopamine: Reward signal for learning
            learning_rate: STDP learning rate
        """
        # Step 0: Update Firing Rate (Exponential Moving Average)
        # Track "how busy am I?" over time for homeostatic regulation
        instant_rate = 1.0 if self.is_firing else 0.0
        self.avg_firing_rate = (0.99 * self.avg_firing_rate) + (0.01 * instant_rate)

        # Save previous firing state for STDP before resetting
        was_firing = self.is_firing

        # Reset firing state
        self.is_firing = False

        # Step 1: Update Eligibility Traces (always, even in refractory)
        for source_id, synapse in self.inputs.items():
            pre_fired = network_state.get(source_id, False)
            synapse.update_trace(pre_fired)

        # Step 2: Check Refractory Period
        if current_tick - self.last_spike_tick < self.refractory_period:
            # Cannot fire, potential held at resting
            self.potential = self.resting_potential
            # Still apply plasticity (learning + homeostasis) during refractory!
            self.apply_plasticity(global_dopamine, learning_rate)
            return

        # Step 3: Apply Leak (Membrane Decay)
        self.potential *= self.decay_rate

        # Step 4: Apply External Input (after decay, so it's not diminished)
        if self.type == "input":
            # Input nodes: external input directly sets potential
            if self.external_current != 0.0:
                self.potential = self.external_current
        else:
            # Other nodes: external input adds to potential (teacher forcing)
            self.potential += self.external_current

        # Clear external current after use (must be set each tick if needed)
        self.external_current = 0.0

        # Step 5: Integrate Inputs
        excitatory_current = 0.0
        has_inhibitory_input = False

        for source_id, synapse in self.inputs.items():
            pre_fired = network_state.get(source_id, False)

            # Get current contribution (trace already updated above)
            current = synapse.get_current(pre_fired)

            if synapse.is_inhibitory and pre_fired:
                # Shunting inhibition: active inhibition clamps potential
                has_inhibitory_input = True
            else:
                excitatory_current += current

        # Step 6: Handle Shunting Inhibition
        if has_inhibitory_input:
            # Shunting: multiply potential by a fraction (strong suppression)
            self.potential *= 0.1
        else:
            # Normal integration
            self.potential += excitatory_current

        # Step 7: Check Threshold and Fire
        if self.potential >= self.threshold:
            self.is_firing = True
            self.last_spike_tick = current_tick
            # Reset potential after spike
            self.potential = self.resting_potential

        # Step 8: Apply STDP + Homeostasis
        self.apply_plasticity(global_dopamine, learning_rate)

    def apply_plasticity(self, dopamine, learning_rate):
        """
        Apply both STDP (specific learning) and Homeostasis (global regulation).

        STDP: Strengthens synapses when pre-spike + post-spike + dopamine co-occur
        Homeostasis: Scales all weights to maintain target firing rate
        """
        # A. Calculate Synaptic Scaling Factor (Homeostasis)
        # If firing rate > target: scale DOWN to reduce excitability
        # If firing rate < target: scale UP to increase excitability
        rate_diff = self.avg_firing_rate - self.target_rate
        scaling_factor = 1.0 - (rate_diff * self.homeostasis_rate)

        for synapse in self.inputs.values():
            # 1. Apply STDP (Specific Learning)
            synapse.apply_stdp(self.is_firing, dopamine, learning_rate)

            # 2. Apply Synaptic Scaling (Global Regulation)
            # Only apply to non-input neurons, and only to their INCOMING weights
            # Do NOT apply to input node synapses (they don't have firing rate targets)
            if self.type != "input":
                synapse.weight *= scaling_factor
                # Clamp to safe range (prevent complete silencing or explosion)
                # BUT: For hidden nodes, keep lower ceiling to allow differentiation
                if self.type == "hidden":
                    synapse.weight = max(0.1, min(2.0, synapse.weight))  # Tighter ceiling for hidden
                else:
                    synapse.weight = max(0.1, min(3.0, synapse.weight))  # More permissive for output

    def get_output(self):
        """
        Returns the spike output (1.0 if firing, 0.0 otherwise).
        """
        return self.spike_value if self.is_firing else 0.0

    def __repr__(self):
        return f"BioNode({self.id}, V={self.potential:.3f}, firing={self.is_firing})"

