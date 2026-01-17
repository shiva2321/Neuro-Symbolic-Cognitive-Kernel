"""
Experiment Base: Declarative interface for neural simulations.

Provides a standardized way to:
1. Set up a network topology
2. Run training/testing phases
3. Inject stimuli and measure responses
4. Collect metrics and results

Replaces ad-hoc experiment code with structured, repeatable protocols.

Invariants:
- Each experiment phase (setup, train, test) is explicit
- Metrics are always collected and accessible
- No global state - experiments are independent
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from core.neuron import Neuron
from core.synapse import Synapse
from core.spike import Spike
from core.orchestrator import Orchestrator
from core.graph_store import GraphStore
from core.metrics import MetricsCollector, ProbeSnapshot
from core.event_queue import EventQueue


class ExperimentPhase(Enum):
    """Experiment execution phase."""
    SETUP = "setup"
    TRAINING = "training"
    TESTING = "testing"
    COMPLETE = "complete"


@dataclass
class ExperimentResult:
    """Results from an experiment run."""
    name: str
    phase: str
    duration_ticks: int
    spikes_processed: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class ExperimentBase(ABC):
    """
    Abstract base class for neural experiments.

    Subclasses must implement:
    - build_network(): Construct network topology
    - create_stimuli(): Define input patterns
    - run_training(): Execute learning phase
    - run_test(optional): Execute evaluation phase

    The framework handles:
    - Network initialization
    - Spike injection
    - Metrics collection
    - Result logging
    """

    def __init__(self, name: str, verbose: bool = False):
        """
        Initialize experiment.

        Args:
            name: Unique experiment identifier
            verbose: Print progress during execution
        """
        self.name = name
        self.verbose = verbose

        # Network components
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[int, Dict[int, Synapse]] = {}
        self.graph_store: Optional[GraphStore] = None
        self.orchestrator: Optional[Orchestrator] = None

        # Experiment state
        self.phase = ExperimentPhase.SETUP
        self.results: List[ExperimentResult] = []

        # Stimulus definition
        self.stimuli: Dict[str, List[Spike]] = {}  # stimulus_name -> [spikes]

    @abstractmethod
    def build_network(self) -> None:
        """
        Construct the neural network topology.

        Must create neurons and synapses and populate:
        - self.neurons
        - self.synapses
        - self.graph_store (optional, can be auto-generated)

        Example:
            self.neurons[0] = Neuron(0, neuron_type="input")
            self.neurons[1] = Neuron(1, neuron_type="hidden")
            self.synapses[1] = {0: Synapse(weight=0.5)}
        """
        pass

    @abstractmethod
    def create_stimuli(self) -> None:
        """
        Define input patterns for the experiment.

        Must populate self.stimuli with named spike sequences.

        Example:
            self.stimuli['cs'] = [
                Spike(source_id=0, timestamp=0, magnitude=1.0),
                Spike(source_id=0, timestamp=1, magnitude=1.0),
            ]
        """
        pass

    @abstractmethod
    def run_training(self, duration: int) -> ExperimentResult:
        """
        Execute the training phase.

        Args:
            duration: Maximum simulation ticks

        Returns:
            ExperimentResult with metrics
        """
        pass

    def run_test(self, duration: int) -> ExperimentResult:
        """
        Execute the testing phase (optional).

        Override in subclass if test phase needed.

        Args:
            duration: Maximum simulation ticks

        Returns:
            ExperimentResult with metrics
        """
        return ExperimentResult(
            name=self.name,
            phase="test",
            duration_ticks=duration,
            spikes_processed=0,
            success=True,
        )

    def execute(self, train_duration: int, test_duration: Optional[int] = None) -> List[ExperimentResult]:
        """
        Execute the complete experiment.

        Steps:
        1. Build network
        2. Create stimuli
        3. Run training
        4. Run test (if test_duration provided)
        5. Return results

        Args:
            train_duration: Duration of training phase
            test_duration: Duration of test phase (None = skip test)

        Returns:
            List of ExperimentResult for each phase
        """
        try:
            # Phase 1: Setup
            if self.verbose:
                print(f"[{self.name}] Building network...")
            self.phase = ExperimentPhase.SETUP
            self.build_network()
            self._initialize_orchestrator()

            if self.verbose:
                print(f"[{self.name}] Creating stimuli...")
            self.create_stimuli()

            # Phase 2: Training
            if self.verbose:
                print(f"[{self.name}] Running training ({train_duration} ticks)...")
            self.phase = ExperimentPhase.TRAINING
            train_result = self.run_training(train_duration)
            self.results.append(train_result)

            # Phase 3: Testing (optional)
            if test_duration is not None:
                if self.verbose:
                    print(f"[{self.name}] Running test ({test_duration} ticks)...")
                self.phase = ExperimentPhase.TESTING
                test_result = self.run_test(test_duration)
                self.results.append(test_result)

            self.phase = ExperimentPhase.COMPLETE
            if self.verbose:
                print(f"[{self.name}] Complete!")

        except Exception as e:
            result = ExperimentResult(
                name=self.name,
                phase=self.phase.value,
                duration_ticks=0,
                spikes_processed=0,
                success=False,
                error_message=str(e)
            )
            self.results.append(result)
            if self.verbose:
                print(f"[{self.name}] ERROR: {e}")

        return self.results

    def _initialize_orchestrator(self) -> None:
        """Initialize orchestrator from network components."""
        self.orchestrator = Orchestrator(self.neurons, self.synapses)

        # Register all connections in dispatcher
        if self.graph_store:
            for edge in self.graph_store.get_edges():
                self.orchestrator.register_connection(
                    edge.source_id,
                    edge.target_id,
                    edge.synapse_index
                )

    def inject_stimulus(self, stimulus_name: str) -> int:
        """
        Inject a named stimulus into the network.

        Args:
            stimulus_name: Name of stimulus in self.stimuli

        Returns:
            Number of spikes injected

        Raises:
            KeyError: If stimulus not found
            RuntimeError: If orchestrator not initialized
        """
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized - call build_network() first")

        if stimulus_name not in self.stimuli:
            raise KeyError(f"Stimulus '{stimulus_name}' not found")

        spikes = self.stimuli[stimulus_name]
        for spike in spikes:
            self.orchestrator.inject_spike(spike)

        return len(spikes)

    def run_simulation(self, duration: int) -> int:
        """
        Run simulation until timeout or queue empty.

        Args:
            duration: Maximum simulation ticks

        Returns:
            Number of steps executed
        """
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")

        return self.orchestrator.run_until(duration)

    def get_metrics(self) -> ProbeSnapshot:
        """Get current metrics snapshot."""
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")

        return self.orchestrator.metrics.get_snapshot(self.orchestrator.current_tick)

    def get_results(self) -> List[ExperimentResult]:
        """Get all experiment results."""
        return self.results.copy()

    def reset_network(self) -> None:
        """Reset all neurons to baseline state for next trial."""
        if self.orchestrator is None:
            return

        for neuron in self.neurons.values():
            neuron.reset()

        self.orchestrator.metrics.clear_all()
        # Reinitialize event queue
        self.orchestrator.event_queue = EventQueue()

    def get_neuron_potential(self, neuron_id: int) -> float:
        """Get current membrane potential of a neuron."""
        return self.neurons[neuron_id].potential

    def get_neuron_is_firing(self, neuron_id: int) -> bool:
        """Check if a neuron is currently firing."""
        return self.neurons[neuron_id].is_firing

    def get_synapse_weight(self, target_id: int, synapse_index: int) -> float:
        """Get current weight of a synapse."""
        return self.synapses[target_id][synapse_index].weight

    def set_learning_enabled(self, enabled: bool) -> None:
        """Enable or disable learning globally."""
        if self.orchestrator:
            self.orchestrator.enable_learning(enabled)

    def __repr__(self):
        return f"Experiment({self.name}, phase={self.phase.value})"
