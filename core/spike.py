"""
Spike: Immutable spike event with causality guarantees.

Defines the fundamental unit of neural communication.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Spike:
    """
    Immutable spike event.

    Represents a discrete action potential emission from a neuron.
    The frozen flag ensures causality: once created, spike state cannot change.

    Attributes:
        source_id: Unique identifier of neuron that fired
        timestamp: When spike was emitted (simulator ticks)
        magnitude: Spike amplitude (typically 1.0 for standard spike)
    """
    source_id: int
    timestamp: int
    magnitude: float = 1.0

    def __post_init__(self):
        """Validate spike invariants at creation time."""
        if self.source_id < 0:
            raise ValueError(f"source_id must be non-negative, got {self.source_id}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if self.magnitude <= 0:
            raise ValueError(f"magnitude must be positive, got {self.magnitude}")

    def __repr__(self):
        return f"Spike(src={self.source_id}, t={self.timestamp}, mag={self.magnitude:.3f})"


@dataclass(frozen=True)
class WeightedSpike:
    """
    Spike after transmission through a synapse.

    Represents a spike that has been weighted by synaptic strength.
    Used internally by synapses when transmitting spikes to postsynaptic targets.

    Attributes:
        spike: Original spike event
        weight: Synaptic transmission strength
        is_inhibitory: If True, this is an inhibitory synapse
    """
    spike: Spike
    weight: float
    is_inhibitory: bool = False

    def __post_init__(self):
        """Validate weighted spike invariants."""
        if self.weight < -5.0 or self.weight > 5.0:
            raise ValueError(f"weight out of reasonable bounds: {self.weight}")

    @property
    def source_id(self):
        return self.spike.source_id

    @property
    def timestamp(self):
        return self.spike.timestamp

    def __repr__(self):
        typ = "INH" if self.is_inhibitory else "EXC"
        return f"WeightedSpike({typ}, src={self.source_id}, w={self.weight:.3f})"
