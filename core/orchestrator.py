"""
Orchestrator: Event-driven simulation engine.

Coordinates all Phase 1 modules into a cohesive neural simulation:
- EventQueue: temporal spike ordering
- Dispatcher: spike routing
- Neuron: integration and spike emission
- Synapse: weight and transmission
- PlasticityController: learning rules

The orchestrator is the "glue" that runs a simulation step by step,
processing events in causally-correct order and applying learning.

State Machine:
  1. RESTING: No events, waiting for external stimuli
  2. PROCESSING: Pulling events from queue, computing neuron responses
  3. LEARNING: Applying plasticity updates to synapses
  4. IDLE: Waiting for next batch

Invariants:
- No spike can be processed before it was emitted
- All plasticity updates happen after spike transmission
- Simulation time always moves forward
- Same event sequence always produces same output (if seeded)
"""

from typing import Dict, List, Optional, Tuple, Callable, Set
from enum import Enum
from .spike import Spike, WeightedSpike
from .event_queue import EventQueue
from .dispatcher import Dispatcher
from .neuron import Neuron
from .synapse import Synapse
from .plasticity import STDPRule, HomeostasisRule
from .metrics import MetricsCollector
from .control_layer import ControlLayer, LearningContext


class SimulationState(Enum):
    """Orchestrator execution state."""
    RESTING = "resting"
    PROCESSING = "processing"
    LEARNING = "learning"
    IDLE = "idle"


class Orchestrator:
    """
    Event-driven neural simulation engine.

    Coordinates:
    - Spike injection and temporal ordering (EventQueue)
    - Spike routing to targets (Dispatcher)
    - Neural integration and spike emission (Neuron)
    - Synaptic transmission (Synapse)
    - Learning updates (PlasticityController)

    Public Interface:
    - inject_spike(spike) -> None
    - step() -> bool (True if processing, False if idle)
    - run_until(timeout_ticks) -> int (number of steps executed)
    - get_neuron(neuron_id) -> Neuron
    - get_synapse(target_id, synapse_index) -> Synapse
    - enable_learning(flag) -> None
    """

    def __init__(
        self,
        network_neurons: Dict[int, Neuron],
        network_synapses: Dict[int, Dict[int, Synapse]],
        enable_control_layer: bool = True,
        novelty_threshold: float = 0.5,
        output_neuron_ids: Optional[Set[int]] = None
    ):
        """
        Initialize orchestrator with a network topology.

        Args:
            network_neurons: Dict mapping neuron_id -> Neuron instance
            network_synapses: Dict mapping target_id -> {synapse_index -> Synapse}
            enable_control_layer: Enable Phase 3 cognitive control (default True)
            novelty_threshold: Novelty threshold for plasticity gating
            output_neuron_ids: Set of output neuron IDs for confidence/conflict monitoring
        """
        self.neurons = network_neurons
        self.synapses = network_synapses

        self.event_queue = EventQueue()
        self.dispatcher = Dispatcher()

        # Track pre/post spikes for plasticity
        self._pre_spikes: Dict[int, Spike] = {}  # neuron_id -> last spike
        self._post_spikes: Dict[int, Spike] = {}  # neuron_id -> last spike

        # Plasticity rules
        self._stdp_rule = STDPRule(learning_rate=0.01)
        self._homeostasis_rule = HomeostasisRule(target_rate=0.25, adaptation_rate=0.005)
        self._learning_enabled = True

        # Metrics collection
        self.metrics = MetricsCollector()

        # Phase 3: Control Layer (System 1.5)
        self.control_layer_enabled = enable_control_layer
        self.control_layer = ControlLayer(
            novelty_threshold=novelty_threshold,
            default_learning=False  # Phase 3: Learning OFF by default
        ) if enable_control_layer else None

        self.output_neuron_ids = output_neuron_ids or set()
        self._current_firing: Set[int] = set()
        self._all_neuron_ids = set(network_neurons.keys())

        # State tracking
        self.state = SimulationState.RESTING
        self.current_tick = 0
        self.steps_executed = 0
        self.spikes_processed = 0

        # Metrics
        self._spike_log: List[Spike] = []
        self._fired_neurons: Dict[int, int] = {}  # neuron_id -> fire count

    def inject_spike(self, spike: Spike) -> None:
        """
        Inject a spike event into the simulation.

        Spikes are ordered by EventQueue. Can be injected from external stimuli,
        test inputs, or spontaneous activity.

        Args:
            spike: The spike to inject

        Raises:
            ValueError: If spike timestamp < current simulation time (causality violation)
        """
        self.event_queue.inject(spike)
        self.state = SimulationState.PROCESSING

    def enable_learning(self, flag: bool) -> None:
        """
        Enable or disable learning globally.

        When disabled, synapses still transmit spikes but weights don't change.
        Useful for testing inference vs. learning modes.

        Args:
            flag: True to enable learning, False to disable
        """
        self._learning_enabled = flag

    def step(self) -> bool:
        """
        Execute one simulation step.

        Process the next spike in the event queue:
        1. Pop spike from queue
        2. Route spike to target neurons via dispatcher
        3. Each target neuron integrates the weighted spike
        4. If target fires, inject its spike into the queue
        5. Apply plasticity updates (if control layer permits)

        Phase 3: Control layer monitors firing patterns and gates plasticity.

        Returns:
            True if a spike was processed, False if queue is empty
        """
        # Try to pop next event
        spike = self.event_queue.pop_next()
        if spike is None:
            self.state = SimulationState.IDLE
            return False

        self.state = SimulationState.PROCESSING
        self.current_tick = spike.timestamp
        self.steps_executed += 1
        self.spikes_processed += 1

        # Store pre-spike for plasticity
        self._pre_spikes[spike.source_id] = spike

        # Track firing for control layer
        self._current_firing.clear()
        self._current_firing.add(spike.source_id)

        # Route spike to all targets
        if self.dispatcher.has_targets(spike.source_id):
            weighted_spikes = self.dispatcher.fan_out(spike, self._get_synapse)

            # Record spike routing in metrics
            self.metrics.record_spike(spike, len(weighted_spikes))

            # Each target integrates and potentially fires
            for target_id, weighted_spike in weighted_spikes:
                target_neuron = self.neurons[target_id]

                # Integrate weighted spike
                # neuron.integrate expects: [(source_id, WeightedSpike), ...]
                post_spike = target_neuron.integrate(self.current_tick, [(spike.source_id, weighted_spike)])

                # If target fires, inject into queue
                if post_spike is not None:
                    self.event_queue.inject(post_spike)
                    self._post_spikes[target_id] = post_spike
                    self._fired_neurons[target_id] = self._fired_neurons.get(target_id, 0) + 1
                    self._current_firing.add(target_id)

                    # Phase 3: Control layer decides if learning should occur
                    if self._should_learn_now():
                        self._apply_plasticity(target_id, spike, post_spike)

        self.state = SimulationState.LEARNING

        # Phase 3: Update control layer with current state
        if self.control_layer_enabled and self.control_layer:
            self._update_control_layer()

        return True

    def run_until(self, timeout_ticks: int) -> int:
        """
        Run simulation until a timeout or queue empty.

        Repeatedly calls step() until current_tick >= timeout_ticks or queue is empty.

        Args:
            timeout_ticks: Stop when simulation time reaches this value

        Returns:
            Number of steps executed
        """
        start_steps = self.steps_executed

        while self.current_tick < timeout_ticks:
            if not self.step():
                break

        return self.steps_executed - start_steps

    def register_connection(self, source_id: int, target_id: int, synapse_index: int = 0) -> None:
        """
        Register a synaptic connection in the dispatcher.

        Called during network construction to wire neurons together.

        Args:
            source_id: Presynaptic neuron
            target_id: Postsynaptic neuron
            synapse_index: Index in target's input array
        """
        self.dispatcher.register_outgoing(source_id, target_id, synapse_index)

    def get_neuron(self, neuron_id: int) -> Neuron:
        """Get a neuron by ID."""
        return self.neurons[neuron_id]

    def get_synapse(self, target_id: int, synapse_index: int) -> Synapse:
        """Get a synapse by target and index."""
        return self.synapses[target_id][synapse_index]

    def _get_synapse(self, target_id: int, synapse_index: int) -> Synapse:
        """Internal helper for dispatcher fan_out."""
        return self.get_synapse(target_id, synapse_index)

    def _apply_plasticity(self, post_neuron_id: int, pre_spike: Spike, post_spike: Spike) -> None:
        """
        Apply learning rules to synapses that contributed to a post-spike.

        Called after a neuron fires. Updates all incoming synapses using STDP
        and homeostasis rules.

        Args:
            post_neuron_id: Neuron that just fired
            pre_spike: The presynaptic spike that triggered postfire
            post_spike: The postsynaptic spike that just occurred
        """
        if not self._learning_enabled:
            return

        incoming_synapses = self.synapses.get(post_neuron_id, {})

        # Apply STDP to each incoming synapse
        for synapse in incoming_synapses.values():
            self._stdp_rule.apply(synapse, post_fired=True, dopamine=1.0)

        # Apply homeostasis to regulate firing rate
        neuron = self.neurons[post_neuron_id]
        current_rate = neuron.get_firing_rate_contribution()
        self._homeostasis_rule.apply(incoming_synapses, current_rate, neuron.type)

    def _should_learn_now(self) -> bool:
        """
        Phase 3: Determine if learning should occur at this timestep.

        If control layer is disabled, use global learning flag.
        If control layer is enabled, defer to control layer decision.

        Returns:
            True if learning should be active
        """
        if not self.control_layer_enabled or self.control_layer is None:
            return self._learning_enabled

        return self.control_layer.is_learning_active()

    def _update_control_layer(self) -> None:
        """
        Phase 3: Update control layer with current network state.

        Provides control layer with:
        - Firing neurons this timestep
        - All neuron IDs
        - Output neuron states
        - Reward signal (if available)
        - Prediction error (if available)
        """
        if not self.control_layer:
            return

        # Get output neuron states
        output_states = {}
        for neuron_id in self.output_neuron_ids:
            output_states[neuron_id] = neuron_id in self._current_firing

        # Update control layer (no reward/error by default)
        self.control_layer.update(
            firing_neurons=self._current_firing,
            all_neurons=self._all_neuron_ids,
            output_neurons=output_states,
            reward=0.0,
            prediction_error=0.0
        )

    def set_reward_signal(self, reward: float) -> None:
        """
        Phase 3: Inject reward signal for next control layer update.

        Args:
            reward: Reward value (>0 for positive, 0 for none)
        """
        if self.control_layer:
            # Reward will be used in next update
            # For now, force learning if reward > 0
            if reward > 0.0:
                self.control_layer.force_learning(True)

    def get_spike_log(self) -> List[Spike]:
        """Return list of all spikes processed."""
        return self._spike_log.copy()

    def get_firing_counts(self) -> Dict[int, int]:
        """Return dict of neuron_id -> spike count."""
        return self._fired_neurons.copy()

    def get_metrics(self) -> Dict:
        """
        Get simulation metrics.

        Returns:
            Dictionary with:
            - steps_executed: Total simulation steps
            - spikes_processed: Total spikes routed
            - current_tick: Current simulation time
            - learning_enabled: Is learning active
            - state: Current orchestrator state
            - control_layer: Phase 3 control layer statistics (if enabled)
        """
        metrics = {
            "steps_executed": self.steps_executed,
            "spikes_processed": self.spikes_processed,
            "current_tick": self.current_tick,
            "learning_enabled": self._learning_enabled,
            "state": self.state.value,
        }

        # Phase 3: Add control layer statistics
        if self.control_layer_enabled and self.control_layer:
            metrics["control_layer"] = self.control_layer.get_full_statistics()

        return metrics
