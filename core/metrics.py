"""
Metrics & Probes: Observability layer for neural simulation.

Probes collect metrics without intruding into core computation.
Three probe types:
1. SpikeProbe: Counts spikes, latencies
2. WeightProbe: Tracks synaptic weight changes
3. StateProbe: Monitors neuron membrane potential

Probes attach to orchestrator and are called after events.
All metrics are optional - computation works without them.

Invariants:
- Probes never modify computation
- Probe attachment is runtime-safe (can attach/detach mid-simulation)
- Metric collection has negligible overhead
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from .spike import Spike
from collections import deque


class ProbeType(Enum):
    """Types of probes available."""
    SPIKE = "spike"
    WEIGHT = "weight"
    STATE = "state"


@dataclass
class SpikeMetric:
    """Recorded spike event with metadata."""
    source_id: int
    timestamp: int
    targets: int  # Number of neurons this spike reached
    magnitude: float


@dataclass
class WeightMetric:
    """Recorded weight change."""
    target_id: int
    synapse_index: int
    old_weight: float
    new_weight: float
    delta: float
    timestamp: int


@dataclass
class StateMetric:
    """Recorded neuron state."""
    neuron_id: int
    potential: float
    is_firing: bool
    timestamp: int


@dataclass
class ProbeSnapshot:
    """Snapshot of all metrics collected so far."""
    spike_count: int = 0
    total_spikes_routed: int = 0
    spike_log: List[SpikeMetric] = field(default_factory=list)

    weight_updates: int = 0
    weight_log: List[WeightMetric] = field(default_factory=list)

    state_samples: int = 0
    state_log: List[StateMetric] = field(default_factory=list)

    simulation_time: int = 0


class SpikeProbe:
    """
    Probe for spike events.

    Records:
    - When spikes are emitted
    - How many targets each spike reaches
    - Spike arrival latency (timestamp delay)
    """

    def __init__(self):
        """Initialize spike probe."""
        self.spike_log: List[SpikeMetric] = []
        self.spike_count = 0

    def record_spike(self, spike: Spike, target_count: int) -> None:
        """
        Record a spike event.

        Args:
            spike: The spike that was emitted
            target_count: Number of targets this spike reached
        """
        metric = SpikeMetric(
            source_id=spike.source_id,
            timestamp=spike.timestamp,
            targets=target_count,
            magnitude=spike.magnitude
        )
        self.spike_log.append(metric)
        self.spike_count += 1

    def get_firing_rate(self, neuron_id: int, total_ticks: int) -> float:
        """
        Get average firing rate for a neuron.

        Args:
            neuron_id: Neuron to query
            total_ticks: Total simulation time

        Returns:
            Firing rate as spikes per tick (0.0-1.0)
        """
        if total_ticks == 0:
            return 0.0

        count = sum(1 for m in self.spike_log if m.source_id == neuron_id)
        return count / total_ticks

    def get_fanout_stats(self) -> Dict[str, float]:
        """
        Get statistics about spike routing.

        Returns:
            Dictionary with mean, min, max fanout
        """
        if not self.spike_log:
            return {"mean": 0.0, "min": 0, "max": 0}

        targets = [m.targets for m in self.spike_log]
        return {
            "mean": sum(targets) / len(targets),
            "min": min(targets),
            "max": max(targets),
        }

    def clear(self) -> None:
        """Clear all recorded spikes."""
        self.spike_log.clear()
        self.spike_count = 0


class WeightProbe:
    """
    Probe for synaptic weight changes.

    Records:
    - Each weight update (old → new)
    - Magnitude of change
    - When updates occur
    """

    def __init__(self):
        """Initialize weight probe."""
        self.weight_log: List[WeightMetric] = []
        self.weight_updates = 0

    def record_update(
        self,
        target_id: int,
        synapse_index: int,
        old_weight: float,
        new_weight: float,
        timestamp: int
    ) -> None:
        """
        Record a weight change.

        Args:
            target_id: Postsynaptic neuron
            synapse_index: Index of updated synapse
            old_weight: Weight before update
            new_weight: Weight after update
            timestamp: When update occurred
        """
        metric = WeightMetric(
            target_id=target_id,
            synapse_index=synapse_index,
            old_weight=old_weight,
            new_weight=new_weight,
            delta=new_weight - old_weight,
            timestamp=timestamp
        )
        self.weight_log.append(metric)
        self.weight_updates += 1

    def get_learning_curve(self, target_id: int, synapse_index: int) -> List[float]:
        """
        Get weight trajectory for a synapse.

        Args:
            target_id: Postsynaptic neuron
            synapse_index: Index of synapse

        Returns:
            List of weight values in temporal order
        """
        relevant = [
            m for m in self.weight_log
            if m.target_id == target_id and m.synapse_index == synapse_index
        ]
        relevant.sort(key=lambda m: m.timestamp)
        return [m.new_weight for m in relevant]

    def get_weight_stats(self) -> Dict[str, float]:
        """
        Get statistics about weight changes.

        Returns:
            Dictionary with mean delta, total change, etc.
        """
        if not self.weight_log:
            return {"mean_delta": 0.0, "total_change": 0.0}

        deltas = [m.delta for m in self.weight_log]
        return {
            "mean_delta": sum(deltas) / len(deltas),
            "total_change": sum(abs(d) for d in deltas),
        }

    def clear(self) -> None:
        """Clear all recorded weights."""
        self.weight_log.clear()
        self.weight_updates = 0


class StateProbe:
    """
    Probe for neuron membrane state.

    Records:
    - Membrane potential at each step
    - Firing status
    - Useful for debugging and visualization
    """

    def __init__(self):
        """Initialize state probe."""
        self.state_log: List[StateMetric] = []
        self.state_samples = 0

    def record_state(
        self,
        neuron_id: int,
        potential: float,
        is_firing: bool,
        timestamp: int
    ) -> None:
        """
        Record neuron state.

        Args:
            neuron_id: Neuron being recorded
            potential: Membrane potential value
            is_firing: Whether neuron is spiking
            timestamp: When state was sampled
        """
        metric = StateMetric(
            neuron_id=neuron_id,
            potential=potential,
            is_firing=is_firing,
            timestamp=timestamp
        )
        self.state_log.append(metric)
        self.state_samples += 1

    def get_neuron_trace(self, neuron_id: int) -> List[float]:
        """
        Get potential trace for a neuron.

        Args:
            neuron_id: Neuron to query

        Returns:
            List of potential values in temporal order
        """
        relevant = [m for m in self.state_log if m.neuron_id == neuron_id]
        relevant.sort(key=lambda m: m.timestamp)
        return [m.potential for m in relevant]

    def get_firing_events(self, neuron_id: int) -> List[int]:
        """
        Get timestamps when neuron fired.

        Args:
            neuron_id: Neuron to query

        Returns:
            List of timestamps when is_firing was True
        """
        return [
            m.timestamp for m in self.state_log
            if m.neuron_id == neuron_id and m.is_firing
        ]

    def clear(self) -> None:
        """Clear all recorded states."""
        self.state_log.clear()
        self.state_samples = 0


class MetricsCollector:
    """
    Aggregator for all probe metrics.

    Provides unified interface for attaching/detaching probes and
    collecting snapshots of all metrics.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.spike_probe = SpikeProbe()
        self.weight_probe = WeightProbe()
        self.state_probe = StateProbe()

        self._spike_callbacks: List[Callable] = []
        self._weight_callbacks: List[Callable] = []
        self._state_callbacks: List[Callable] = []

    def attach_spike_callback(self, callback: Callable) -> None:
        """
        Attach a callback to spike recording.

        Callback signature: (spike: Spike, target_count: int) -> None

        Args:
            callback: Function to call on each spike
        """
        self._spike_callbacks.append(callback)

    def attach_weight_callback(self, callback: Callable) -> None:
        """
        Attach a callback to weight updates.

        Callback signature: (target_id, synapse_index, old_w, new_w, t) -> None

        Args:
            callback: Function to call on each weight change
        """
        self._weight_callbacks.append(callback)

    def attach_state_callback(self, callback: Callable) -> None:
        """
        Attach a callback to state sampling.

        Callback signature: (neuron_id, potential, is_firing, t) -> None

        Args:
            callback: Function to call on each state sample
        """
        self._state_callbacks.append(callback)

    def record_spike(self, spike: Spike, target_count: int) -> None:
        """Record spike in probe and notify callbacks."""
        self.spike_probe.record_spike(spike, target_count)
        for callback in self._spike_callbacks:
            try:
                callback(spike, target_count)
            except Exception:
                pass  # Silently ignore callback errors

    def record_weight(
        self,
        target_id: int,
        synapse_index: int,
        old_weight: float,
        new_weight: float,
        timestamp: int
    ) -> None:
        """Record weight change in probe and notify callbacks."""
        self.weight_probe.record_update(target_id, synapse_index, old_weight, new_weight, timestamp)
        for callback in self._weight_callbacks:
            try:
                callback(target_id, synapse_index, old_weight, new_weight, timestamp)
            except Exception:
                pass

    def record_state(
        self,
        neuron_id: int,
        potential: float,
        is_firing: bool,
        timestamp: int
    ) -> None:
        """Record state in probe and notify callbacks."""
        self.state_probe.record_state(neuron_id, potential, is_firing, timestamp)
        for callback in self._state_callbacks:
            try:
                callback(neuron_id, potential, is_firing, timestamp)
            except Exception:
                pass

    def get_snapshot(self, simulation_time: int) -> ProbeSnapshot:
        """
        Get comprehensive metrics snapshot.

        Args:
            simulation_time: Current simulation timestamp

        Returns:
            ProbeSnapshot with all collected metrics
        """
        return ProbeSnapshot(
            spike_count=self.spike_probe.spike_count,
            total_spikes_routed=len(self.spike_probe.spike_log),
            spike_log=self.spike_probe.spike_log.copy(),

            weight_updates=self.weight_probe.weight_updates,
            weight_log=self.weight_probe.weight_log.copy(),

            state_samples=self.state_probe.state_samples,
            state_log=self.state_probe.state_log.copy(),

            simulation_time=simulation_time,
        )

    def clear_all(self) -> None:
        """Clear all probes."""
        self.spike_probe.clear()
        self.weight_probe.clear()
        self.state_probe.clear()


class SignalComputer:
    """
    Computes meta-learning signals: confidence, novelty, error, stability.

    These signals feed into MetaLearner.update to adapt plasticity.
    """

    def __init__(self, window_size: int = 100):
        """Initialize signal computer with tracking window."""
        self.window_size = window_size
        self._weight_history: deque = deque(maxlen=window_size)
        self._firing_rate_history: deque = deque(maxlen=window_size)
        self._error_samples: deque = deque(maxlen=window_size)
        self._seen_patterns: set = set()

    def update(
        self,
        current_weights: List[float],
        current_firing_rate: float,
        prediction_error: float = 0.0,
        observed_pattern_id: str = ""
    ) -> None:
        """
        Update signal tracking with current observations.

        Args:
            current_weights: List of synaptic weights
            current_firing_rate: Firing rate [0..1]
            prediction_error: Prediction error magnitude [0..1]
            observed_pattern_id: Pattern identifier for novelty tracking
        """
        self._weight_history.append(current_weights.copy())
        self._firing_rate_history.append(current_firing_rate)
        self._error_samples.append(prediction_error)
        if observed_pattern_id:
            self._seen_patterns.add(observed_pattern_id)

    def compute_confidence(self) -> float:
        """
        Compute confidence as inverse of weight/firing variance.
        High confidence: weights and firing rate are stable.
        """
        if len(self._weight_history) < 2:
            return 0.5
        # Flatten all weights and compute variance
        all_weights = []
        for w_list in self._weight_history:
            all_weights.extend(w_list)
        if not all_weights:
            return 0.5
        mean_w = sum(all_weights) / len(all_weights)
        variance = sum((w - mean_w) ** 2 for w in all_weights) / len(all_weights)
        # Confidence = 1 / (1 + variance)
        return min(1.0, 1.0 / (1.0 + variance))

    def compute_novelty(self, new_pattern_id: str = "") -> float:
        """
        Compute novelty as fraction of new patterns seen.
        High novelty: many unseen patterns.
        """
        if not new_pattern_id:
            return 0.0
        if new_pattern_id not in self._seen_patterns:
            return 1.0
        # Approximate: novelty = (unseen / total) in recent window
        # For simplicity, return 0.0 for seen patterns
        return 0.0

    def compute_error(self) -> float:
        """Compute prediction error as running average of samples."""
        if not self._error_samples:
            return 0.0
        return sum(self._error_samples) / len(self._error_samples)

    def compute_stability(self) -> float:
        """
        Compute stability as inverse of firing rate variance.
        High stability: firing rate is consistent.
        """
        if len(self._firing_rate_history) < 2:
            return 0.5
        rates = list(self._firing_rate_history)
        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        # Stability = 1 / (1 + variance)
        return min(1.0, 1.0 / (1.0 + variance))

    def get_signals(self, new_pattern_id: str = "") -> Dict[str, float]:
        """Get all signals for MetaLearner.update()."""
        return {
            'confidence': self.compute_confidence(),
            'novelty': self.compute_novelty(new_pattern_id),
            'error': self.compute_error(),
            'stability': self.compute_stability(),
            'performance_delta': 0.0,  # Will be set by caller
        }

    def compute_intrinsic_reward(
        self,
        novelty: float,
        pred_error_delta: float,
        beta_intrinsic: float = 0.1
    ) -> float:
        """
        Compute intrinsic reward from novelty and prediction error change (Phase 5).

        Intrinsic motivation combines:
        - Novelty: Reward for exploring new patterns
        - Prediction error change: Reward for learning (reducing errors)

        Formula: intrinsic = (novelty * 0.5 + |pred_error_delta| * 0.5) * beta_intrinsic
        Capped at [0, 1]

        Args:
            novelty: Novelty score [0, 1]
            pred_error_delta: Change in prediction error (positive or negative)
            beta_intrinsic: Weight on intrinsic component [0, 1]

        Returns:
            Intrinsic reward [0, 1]
        """
        # Combine novelty and prediction error change
        raw_intrinsic = (novelty * 0.5) + (abs(pred_error_delta) * 0.5)
        # Apply beta weight and cap at 1.0
        return min(1.0, raw_intrinsic * beta_intrinsic)

    def clear(self) -> None:
        """Clear all history."""
        self._weight_history.clear()
        self._firing_rate_history.clear()
        self._error_samples.clear()
        self._seen_patterns.clear()
