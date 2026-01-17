"""
Region: Spatial organization of neurons.

A Region groups neurons together with local properties:
- Local plasticity controller
- Local metrics
- Input/output ports
- Can be frozen independently
- Can be analyzed in isolation

Philosophy:
- Brain is not flat - it has structure (visual cortex, motor cortex, etc.)
- Different regions learn at different rates
- Some regions can be protected from forgetting
- Enables hierarchical reasoning

Invariants:
- Region owns its neurons (no sharing)
- Connections between regions only through ports
- Each region has independent plasticity gate
- Region metrics are local (no global state)
"""

from typing import Dict, Set, List, Optional
from dataclasses import dataclass
from enum import Enum
from .neuron import Neuron
from .synapse import Synapse
from .plasticity import PlasticityController


class RegionType(Enum):
    """Types of brain regions."""
    SENSORY = "sensory"      # Receives external input
    HIDDEN = "hidden"        # Internal processing
    MOTOR = "motor"          # Produces output
    MEMORY = "memory"        # Stores information
    CONTROL = "control"      # Decision making


@dataclass
class RegionPort:
    """Connection point between regions."""
    port_id: int
    neuron_id: int
    direction: str  # "in" or "out"
    capacity: int = 100  # Max connections through this port


@dataclass
class RegionMetrics:
    """Local metrics for a region."""
    firing_rate: float = 0.0
    mean_weight: float = 0.5
    weight_variance: float = 0.1
    active_neurons: int = 0
    plasticity_enabled: bool = True
    timestamp: int = 0


class Region:
    """
    A region of the neural network.

    Represents a coherent functional area with:
    - Local neuron population
    - Local plasticity rules
    - Input/output ports for inter-region communication
    - Independent learning enable/disable
    - Local statistics
    """

    def __init__(
        self,
        region_id: int,
        region_type: RegionType,
        neurons: Dict[int, Neuron],
        synapses: Dict[int, Dict[int, Synapse]],
        plasticity_controller: Optional[PlasticityController] = None
    ):
        """
        Initialize a region.

        Args:
            region_id: Unique region identifier
            region_type: Type of region (sensory, hidden, etc.)
            neurons: Dict of neuron_id -> Neuron for this region
            synapses: Dict of target_id -> {source_id -> Synapse} for this region
            plasticity_controller: Optional plasticity controller (defaults to new)
        """
        self.region_id = region_id
        self.region_type = region_type
        self.neurons = neurons
        self.synapses = synapses

        # Plasticity (local to region)
        self.plasticity = plasticity_controller or PlasticityController()
        # Region-specific learning rule configuration
        self._learning_rule_id: str = "stdp"  # default
        self._lr_multiplier: float = 1.0

        # Ports (connections to other regions)
        self._input_ports: Dict[int, RegionPort] = {}
        self._output_ports: Dict[int, RegionPort] = {}

        # State
        self._frozen = False
        self._metrics = RegionMetrics()
        self._firing_history: List[Set[int]] = []
        self._weight_history: List[Dict[int, float]] = []

    def add_input_port(self, port_id: int, neuron_id: int) -> None:
        """
        Add input port (receives spikes from other regions).

        Args:
            port_id: Unique port identifier
            neuron_id: Neuron that receives external input
        """
        if neuron_id not in self.neurons:
            raise ValueError(f"Neuron {neuron_id} not in region {self.region_id}")

        self._input_ports[port_id] = RegionPort(
            port_id=port_id,
            neuron_id=neuron_id,
            direction="in"
        )

    def add_output_port(self, port_id: int, neuron_id: int) -> None:
        """
        Add output port (sends spikes to other regions).

        Args:
            port_id: Unique port identifier
            neuron_id: Neuron whose spikes exit region
        """
        if neuron_id not in self.neurons:
            raise ValueError(f"Neuron {neuron_id} not in region {self.region_id}")

        self._output_ports[port_id] = RegionPort(
            port_id=port_id,
            neuron_id=neuron_id,
            direction="out"
        )

    def get_input_ports(self) -> Dict[int, RegionPort]:
        """Get all input ports."""
        return self._input_ports.copy()

    def get_output_ports(self) -> Dict[int, RegionPort]:
        """Get all output ports."""
        return self._output_ports.copy()

    def freeze(self) -> None:
        """
        Freeze region (disable learning, prevent weight changes).

        Used to protect learned knowledge from being overwritten.
        """
        self._frozen = True
        self.plasticity.disable_learning()

    def unfreeze(self) -> None:
        """
        Unfreeze region (re-enable learning).
        """
        self._frozen = False
        self.plasticity.enable_learning()

    def is_frozen(self) -> bool:
        """Check if region is frozen."""
        return self._frozen

    def enable_plasticity(self) -> None:
        """Enable learning for this region."""
        if not self._frozen:
            self.plasticity.enable_learning()

    def disable_plasticity(self) -> None:
        """Disable learning for this region (but don't freeze)."""
        self.plasticity.disable_learning()

    def is_learning(self) -> bool:
        """Check if learning is active in this region."""
        return self.plasticity.is_learning()

    def update_metrics(self, firing_neurons: Set[int], timestamp: int) -> None:
        """
        Update local metrics based on current activity.

        Args:
            firing_neurons: Neurons that fired this timestep
            timestamp: Current simulation time
        """
        self._metrics.timestamp = timestamp
        self._metrics.active_neurons = len(firing_neurons)
        self._metrics.plasticity_enabled = self.plasticity.is_learning()

        # Update firing rate (EMA)
        local_firing = len(firing_neurons & set(self.neurons.keys()))
        local_rate = local_firing / max(len(self.neurons), 1)
        self._metrics.firing_rate = 0.9 * self._metrics.firing_rate + 0.1 * local_rate

        # Update weight statistics
        all_weights = []
        for target_synapses in self.synapses.values():
            for synapse in target_synapses.values():
                all_weights.append(synapse.weight)

        if all_weights:
            self._metrics.mean_weight = sum(all_weights) / len(all_weights)
            mean_val = self._metrics.mean_weight
            variance = sum((w - mean_val) ** 2 for w in all_weights) / len(all_weights)
            self._metrics.weight_variance = variance

        # Track history
        self._firing_history.append(firing_neurons.copy())
        if len(self._firing_history) > 100:
            self._firing_history.pop(0)

    def get_metrics(self) -> RegionMetrics:
        """Get current metrics for this region."""
        return self._metrics

    def get_size(self) -> int:
        """Get number of neurons in region."""
        return len(self.neurons)

    def get_neuron_ids(self) -> Set[int]:
        """Get all neuron IDs in region."""
        return set(self.neurons.keys())

    def contains_neuron(self, neuron_id: int) -> bool:
        """Check if neuron is in this region."""
        return neuron_id in self.neurons

    def get_connectivity_count(self) -> int:
        """Get number of synapses in region."""
        return sum(len(synapses) for synapses in self.synapses.values())

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics for region."""
        return {
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "neurons": self.get_size(),
            "synapses": self.get_connectivity_count(),
            "frozen": self._frozen,
            "learning_enabled": self.plasticity.is_learning(),
            "metrics": {
                "firing_rate": self._metrics.firing_rate,
                "mean_weight": self._metrics.mean_weight,
                "weight_variance": self._metrics.weight_variance,
                "active_neurons": self._metrics.active_neurons,
            },
            "input_ports": len(self._input_ports),
            "output_ports": len(self._output_ports),
        }

    def __repr__(self):
        frozen_str = "FROZEN" if self._frozen else "active"
        learning_str = "learning" if self.plasticity.is_learning() else "static"
        return (
            f"Region(id={self.region_id}, type={self.region_type.value}, "
            f"neurons={self.get_size()}, {frozen_str}, {learning_str})"
        )

    def set_learning_rule(self, rule_id: str) -> None:
        """
        Set the local learning rule identifier (e.g., 'stdp', 'oja', 'hebbian').
        Currently scaffolded; defaults to 'stdp'.
        """
        self._learning_rule_id = rule_id or "stdp"

    def set_learning_rate_multiplier(self, multiplier: float) -> None:
        """Set a per-region learning rate multiplier (>=0)."""
        self._lr_multiplier = max(0.0, multiplier)
        self.plasticity.set_region_lr_multiplier(self._lr_multiplier)

    def apply_learning(self, target_id: int, post_fired: bool, dopamine: float, step: int = 0, perf_delta: float = 0.0, stability: float = 0.0) -> None:
        """
        Apply the configured learning rule to all incoming synapses of a target neuron.
        For now, dispatches to STDP via PlasticityController with adaptive rate.
        """
        if target_id not in self.synapses:
            return
        # For each incoming synapse, apply learning
        for syn in self.synapses[target_id].values():
            # Future: switch based on self._learning_rule_id
            self.plasticity.update_learning(synapse=syn, post_fired=post_fired, dopamine=dopamine, step=step, perf_delta=perf_delta, stability=stability)


class RegionNetwork:
    """
    Network of regions with inter-region connectivity.

    Manages:
    - Multiple regions
    - Connections between regions
    - Global vs local learning
    - Region coordination
    """

    def __init__(self):
        """Initialize region network."""
        self._regions: Dict[int, Region] = {}
        self._inter_region_connections: List[tuple] = []
        self._global_metrics: Dict = {}

    def add_region(self, region: Region) -> None:
        """
        Add region to network.

        Args:
            region: Region to add

        Raises:
            ValueError: If region ID already exists
        """
        if region.region_id in self._regions:
            raise ValueError(f"Region {region.region_id} already exists")

        self._regions[region.region_id] = region

    def get_region(self, region_id: int) -> Optional[Region]:
        """Get region by ID."""
        return self._regions.get(region_id)

    def get_regions(self) -> Dict[int, Region]:
        """Get all regions."""
        return self._regions.copy()

    def connect_regions(
        self,
        source_region_id: int,
        target_region_id: int,
        source_port_id: int,
        target_port_id: int
    ) -> None:
        """
        Connect two regions through ports.

        Args:
            source_region_id: Source region
            target_region_id: Target region
            source_port_id: Output port in source
            target_port_id: Input port in target
        """
        source_region = self.get_region(source_region_id)
        target_region = self.get_region(target_region_id)

        if not source_region or not target_region:
            raise ValueError("Invalid region IDs")

        # Verify ports exist
        if source_port_id not in source_region.get_output_ports():
            raise ValueError(f"Output port {source_port_id} not in source region")
        if target_port_id not in target_region.get_input_ports():
            raise ValueError(f"Input port {target_port_id} not in target region")

        self._inter_region_connections.append((
            source_region_id, source_port_id,
            target_region_id, target_port_id
        ))

    def freeze_region(self, region_id: int) -> None:
        """Freeze a region (prevent learning)."""
        region = self.get_region(region_id)
        if region:
            region.freeze()

    def unfreeze_region(self, region_id: int) -> None:
        """Unfreeze a region (enable learning)."""
        region = self.get_region(region_id)
        if region:
            region.unfreeze()

    def freeze_all(self) -> None:
        """Freeze all regions."""
        for region in self._regions.values():
            region.freeze()

    def unfreeze_all(self) -> None:
        """Unfreeze all regions."""
        for region in self._regions.values():
            region.unfreeze()

    def enable_learning(self, region_id: Optional[int] = None) -> None:
        """
        Enable learning in region(s).

        Args:
            region_id: Specific region, or None for all
        """
        if region_id is None:
            for region in self._regions.values():
                if not region.is_frozen():
                    region.enable_plasticity()
        else:
            region = self.get_region(region_id)
            if region and not region.is_frozen():
                region.enable_plasticity()

    def disable_learning(self, region_id: Optional[int] = None) -> None:
        """
        Disable learning in region(s).

        Args:
            region_id: Specific region, or None for all
        """
        if region_id is None:
            for region in self._regions.values():
                region.disable_plasticity()
        else:
            region = self.get_region(region_id)
            if region:
                region.disable_plasticity()

    def update_all_metrics(self, firing_neurons: Set[int], timestamp: int) -> None:
        """
        Update metrics for all regions.

        Args:
            firing_neurons: Global set of neurons that fired
            timestamp: Current simulation time
        """
        for region in self._regions.values():
            region.update_metrics(firing_neurons, timestamp)

    def get_global_statistics(self) -> Dict:
        """Get statistics for entire region network."""
        return {
            "num_regions": len(self._regions),
            "total_neurons": sum(r.get_size() for r in self._regions.values()),
            "total_synapses": sum(r.get_connectivity_count() for r in self._regions.values()),
            "inter_region_connections": len(self._inter_region_connections),
            "regions": {
                rid: r.get_statistics()
                for rid, r in self._regions.items()
            }
        }

    def __repr__(self):
        return (
            f"RegionNetwork({len(self._regions)} regions, "
            f"{sum(r.get_size() for r in self._regions.values())} neurons)"
        )
