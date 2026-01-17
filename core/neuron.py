"""
Neuron: Refactored neuron model with separated concerns.

Responsibilities:
- State management (potential, firing status, refractory period)
- Integration (summing weighted inputs)
- Spike emission (threshold detection and firing)

Plasticity is NOT the neuron's responsibility - it's delegated to PlasticityController.
"""

from typing import Dict, Optional
from .synapse import Synapse
from .spike import Spike, WeightedSpike


class Neuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Core behavior:
    - Membrane potential decays toward resting potential (leak)
    - Receives weighted inputs from presynaptic neurons
    - Fires if potential exceeds threshold
    - Enters refractory period after spike (cannot fire)

    Invariants:
    - Potential bounded: [-1, threshold+1]
    - is_firing is boolean
    - last_spike_tick is monotonically non-decreasing
    - Neurons only know their own state + incoming connections

    Separation of Concerns:
    - Computation (this class): integrate and fire
    - Plasticity (PlasticityController): weight updates
    - Routing (Dispatcher): spike delivery
    """

    def __init__(
        self,
        neuron_id: int,
        neuron_type: str = "hidden",
        threshold: float = 1.0,
        refractory_period: int = 3
    ):
        """
        Initialize neuron.

        Args:
            neuron_id: Unique identifier
            neuron_type: "input", "hidden", or "output"
            threshold: Voltage threshold for spiking
            refractory_period: Timesteps during which neuron cannot fire after spike
        """
        # Identity
        self.id = neuron_id
        self.type = neuron_type

        # Parameters
        self.threshold = threshold
        self.refractory_period = refractory_period

        # State: membrane potential and firing
        self.potential = 0.0
        self.is_firing = False
        self.last_spike_tick = -999

        # Synaptic inputs: {source_id -> Synapse}
        self.inputs: Dict[int, Synapse] = {}

        # External input (for input nodes and teacher forcing)
        self.external_current = 0.0

        # Biophysical parameters
        self.decay_rate = 0.8  # Exponential leak
        self.resting_potential = 0.0
        self.spike_magnitude = 1.0

    def add_input(self, source_id: int, weight: float, is_inhibitory: bool = False) -> None:
        """
        Create a synapse from source neuron.

        Args:
            source_id: Presynaptic neuron ID
            weight: Initial synaptic weight
            is_inhibitory: If True, synapse is inhibitory
        """
        self.inputs[source_id] = Synapse(weight, is_inhibitory)

    def set_external_input(self, value: float) -> None:
        """
        Set external input current.

        For input nodes: directly sets potential
        For other nodes: adds to potential (teacher forcing)

        Args:
            value: Input current magnitude
        """
        self.external_current = value

    def integrate(
        self,
        current_tick: int,
        weighted_spikes: list
    ) -> Optional[Spike]:
        """
        Single timestep of neural computation.

        Steps:
        1. Check refractory period (prevents firing if refractory)
        2. Apply membrane leak (exponential decay)
        3. Apply external input
        4. Integrate incoming weighted spikes
        5. Check threshold and emit spike if needed

        Args:
            current_tick: Simulation timestep
            weighted_spikes: List of (source_id, WeightedSpike) arriving this tick

        Returns:
            Spike if neuron fires, None otherwise
        """
        # Reset firing state from previous tick
        self.is_firing = False

        # Step 1: Check Refractory Period
        in_refractory = (current_tick - self.last_spike_tick) < self.refractory_period
        if in_refractory:
            self.potential = self.resting_potential
            return None

        # Step 2: Apply Leak (Exponential decay toward resting potential)
        self.potential *= self.decay_rate

        # Step 3: Apply External Input
        if self.type == "input":
            # Input nodes: external input directly sets potential
            if self.external_current != 0.0:
                self.potential = self.external_current
        else:
            # Other nodes: external input adds to potential (teacher forcing)
            self.potential += self.external_current

        self.external_current = 0.0  # Clear for next tick

        # Step 4: Integrate Weighted Incoming Spikes
        excitatory_current = 0.0
        has_inhibitory_input = False

        for source_id, weighted_spike in weighted_spikes:
            if weighted_spike.is_inhibitory:
                has_inhibitory_input = True
            else:
                excitatory_current += weighted_spike.weight

        # Step 5: Apply Shunting Inhibition
        # If inhibitory input present, multiplicatively reduce potential
        if has_inhibitory_input:
            self.potential *= 0.1  # Strong suppression
        else:
            self.potential += excitatory_current

        # Step 6: Threshold Check and Fire
        if self.potential >= self.threshold:
            self.is_firing = True
            self.last_spike_tick = current_tick
            spike = Spike(self.id, current_tick, self.spike_magnitude)
            self.potential = self.resting_potential  # Reset potential after spike
            return spike

        return None

    def update_traces(self, network_state: Dict[int, bool]) -> None:
        """
        Update eligibility traces for all incoming synapses.

        This must be called every timestep, even during refractory period.
        The trace encodes the history of presynaptic activity.

        Args:
            network_state: Dict[source_id -> is_firing]
        """
        for source_id, synapse in self.inputs.items():
            pre_fired = network_state.get(source_id, False)
            synapse.update_trace(pre_fired)

    def reset(self) -> None:
        """
        Reset neuron to baseline state.

        Clears membrane potential and refractory status.
        Also clears all eligibility traces.
        Used between training episodes to prevent cross-contamination.
        """
        self.potential = self.resting_potential
        self.is_firing = False
        self.last_spike_tick = -999

        for synapse in self.inputs.values():
            synapse.trace = 0.0

    def get_firing_rate_contribution(self) -> float:
        """
        Get this timestep's contribution to exponential moving average firing rate.

        Returns:
            1.0 if firing, 0.0 otherwise
        """
        return 1.0 if self.is_firing else 0.0

    def __repr__(self):
        status = "FIRE" if self.is_firing else "rest"
        return f"Neuron({self.id}[{self.type}], V={self.potential:.3f}, {status})"
