"""
Dispatcher: Fan-out routing of spikes to target neurons.

Routes spikes from firing neurons to their postsynaptic targets.
Implements the spike->synapse->current_injection pathway.

Separation of concerns:
- Dispatcher handles only routing (WHO gets the spike)
- Synapse handles transmission (HOW the spike is modified)
- Neuron handles integration (WHAT happens to the current)
"""

from typing import Dict, List, Tuple, Optional
from .spike import Spike, WeightedSpike


class Dispatcher:
    """
    Routes spikes from source neurons to target neurons via synapses.

    Maintains outgoing connectivity graph:
    {source_id: [(target_id, synapse_index), ...], ...}

    Public Interface:
    - register_outgoing(source_id, target_id, synapse_index)
    - get_targets(source_id) -> Iterable[(target_id, synapse_index)]
    - fan_out(spike, get_synapse_func) -> Iterable[Tuple[target_id, weighted_spike]]
    """

    def __init__(self):
        """Initialize empty dispatcher."""
        self._outgoing: Dict[int, List[Tuple[int, int]]] = {}
        self._route_count = 0

    def register_outgoing(self, source_id: int, target_id: int, synapse_index: int = 0) -> None:
        """
        Register that source_id sends to target_id.

        Args:
            source_id: Presynaptic neuron ID
            target_id: Postsynaptic neuron ID
            synapse_index: Index of synapse in target's input array (for multi-synapse targets)
        """
        if source_id not in self._outgoing:
            self._outgoing[source_id] = []

        self._outgoing[source_id].append((target_id, synapse_index))
        self._route_count += 1

    def get_targets(self, source_id: int) -> List[Tuple[int, int]]:
        """
        Get all targets that source_id connects to.

        Args:
            source_id: Neuron to query

        Returns:
            List of (target_id, synapse_index) tuples
        """
        return self._outgoing.get(source_id, [])

    def fan_out(
        self,
        spike: Spike,
        get_synapse_func
    ) -> List[Tuple[int, WeightedSpike]]:
        """
        Route a spike to all target neurons.

        For each target, calls get_synapse_func(target_id, synapse_index)
        to retrieve the synapse object, then transmits the spike through it.

        Args:
            spike: The spike event to route
            get_synapse_func: Callable that returns synapse(target_id, synapse_index)

        Returns:
            List of (target_id, weighted_spike) tuples ready for injection
        """
        targets = self.get_targets(spike.source_id)

        weighted_spikes = []
        for target_id, synapse_index in targets:
            synapse = get_synapse_func(target_id, synapse_index)
            weighted_spike = synapse.transmit(spike)
            weighted_spikes.append((target_id, weighted_spike))

        return weighted_spikes

    def has_targets(self, source_id: int) -> bool:
        """Check if source_id has any outgoing connections."""
        return source_id in self._outgoing and len(self._outgoing[source_id]) > 0

    def get_fanout_count(self, source_id: int) -> int:
        """Get number of targets for source_id."""
        return len(self._outgoing.get(source_id, []))

    def get_route_count(self) -> int:
        """Get total number of registered routes."""
        return self._route_count

    def reset(self) -> None:
        """Clear all routes."""
        self._outgoing.clear()
        self._route_count = 0

    def __repr__(self):
        return f"Dispatcher(routes={self._route_count}, sources={len(self._outgoing)})"
