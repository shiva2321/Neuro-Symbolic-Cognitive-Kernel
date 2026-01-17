"""
Pavlov Experiment (Phase 2 Implementation): Classical conditioning with new API.

Demonstrates the ExperimentBase framework with a simple classical conditioning task:
1. Setup: Create 3-neuron network (input, hidden, output)
2. Training: Pair CS (conditional stimulus) with US (unconditional stimulus)
3. Test: Present CS alone and measure output neuron response

This replaces the legacy pavlov_experiment.py with the structured Phase 2 API.
All learning behavior is preserved.
"""

import sys
sys.path.insert(0, r'D:\Node_network')

from core.neuron import Neuron
from core.synapse import Synapse
from core.spike import Spike
from core.graph_store import GraphStore
from experiments.base_experiment import ExperimentBase, ExperimentResult, ExperimentPhase


class PavlovExperimentPhase2(ExperimentBase):
    """
    Classical conditioning experiment using Phase 2 architecture.

    Network topology:
        CS (neuron 0) ─┐
                       ├──> Hidden (neuron 1) ──> Output (neuron 2)
        US (neuron 2) ─┘

    Protocol:
        1. Conditioning: Present CS and US together for 5 trials
        2. Testing: Present CS alone and measure output response
    """

    def build_network(self) -> None:
        """Create simple 3-neuron classical conditioning circuit."""
        # Neuron 0: CS input
        self.neurons[0] = Neuron(0, neuron_type="input", threshold=0.5, refractory_period=2)

        # Neuron 1: Hidden (integrator)
        self.neurons[1] = Neuron(1, neuron_type="hidden", threshold=1.0, refractory_period=3)

        # Neuron 2: Output (responds to CS after conditioning)
        self.neurons[2] = Neuron(2, neuron_type="output", threshold=0.8, refractory_period=3)

        # CS -> Hidden: weak synapse that will strengthen
        self.synapses[1] = {0: Synapse(weight=0.3, is_inhibitory=False)}

        # US -> Hidden: strong baseline synapse
        self.synapses[1][1] = Synapse(weight=0.8, is_inhibitory=False)

        # Hidden -> Output: transmission synapse
        self.synapses[2] = {0: Synapse(weight=0.6, is_inhibitory=False)}

        # Build graph store for persistence
        self.graph_store = GraphStore()
        self.graph_store.add_node(0, node_type="input", threshold=0.5)
        self.graph_store.add_node(1, node_type="hidden", threshold=1.0)
        self.graph_store.add_node(2, node_type="output", threshold=0.8)

        self.graph_store.add_edge(0, 1, weight=0.3, is_inhibitory=False)
        self.graph_store.add_edge(1, 1, weight=0.8, is_inhibitory=False)
        self.graph_store.add_edge(1, 2, weight=0.6, is_inhibitory=False)

    def create_stimuli(self) -> None:
        """Define stimulus patterns for conditioning."""
        # CS: 3 spikes from neuron 0
        self.stimuli['cs'] = [
            Spike(source_id=0, timestamp=0, magnitude=1.0),
            Spike(source_id=0, timestamp=1, magnitude=1.0),
            Spike(source_id=0, timestamp=2, magnitude=1.0),
        ]

        # US: 2 spikes from neuron 1 (unconditional stimulus)
        self.stimuli['us'] = [
            Spike(source_id=1, timestamp=3, magnitude=1.0),
            Spike(source_id=1, timestamp=4, magnitude=1.0),
        ]

        # Paired: CS + US together
        self.stimuli['paired'] = self.stimuli['cs'] + self.stimuli['us']

    def run_training(self, duration: int) -> ExperimentResult:
        """
        Execute classical conditioning trials.

        Pairs CS + US repeatedly to strengthen CS -> output connection.
        """
        num_trials = 5
        trial_duration = 20  # ticks per trial

        for trial in range(num_trials):
            if self.verbose:
                print(f"  Trial {trial + 1}/{num_trials}")

            # Inject paired stimulus
            self.inject_stimulus('paired')

            # Run simulation
            self.run_simulation(trial_duration)

            # Reset network between trials
            self.reset_network()

        # Collect metrics
        metrics = self.get_metrics()

        return ExperimentResult(
            name=self.name,
            phase="training",
            duration_ticks=num_trials * trial_duration,
            spikes_processed=metrics.total_spikes_routed,
            metrics={
                "num_trials": num_trials,
                "trial_duration": trial_duration,
                "cs_input_weight": self.get_synapse_weight(1, 0),
            },
            success=True,
        )

    def run_test(self, duration: int) -> ExperimentResult:
        """
        Test if CS alone elicits output response.

        After conditioning, CS should trigger hidden -> output firing.
        """
        # Inject CS alone
        self.inject_stimulus('cs')

        # Run simulation
        self.run_simulation(duration)

        # Measure if output fired
        metrics = self.get_metrics()

        # Check if we got output spikes
        output_spike_count = sum(
            1 for m in metrics.spike_log if m.source_id == 2
        )

        return ExperimentResult(
            name=self.name,
            phase="test",
            duration_ticks=duration,
            spikes_processed=metrics.total_spikes_routed,
            metrics={
                "output_spikes": output_spike_count,
                "cs_input_weight_final": self.get_synapse_weight(1, 0),
                "conditioning_success": output_spike_count > 0,
            },
            success=True,
        )


def run_pavlov_phase2(verbose: bool = True) -> ExperimentResult:
    """
    Execute Pavlov experiment using Phase 2 API.

    Returns aggregate results from training and test phases.
    """
    exp = PavlovExperimentPhase2(
        name="Pavlov_Phase2",
        verbose=verbose
    )

    # Run full experiment protocol
    results = exp.execute(
        train_duration=100,  # 5 trials × 20 ticks
        test_duration=20      # CS alone test
    )

    # Log results
    if verbose:
        print("\n=== Pavlov Experiment Results ===")
        for result in results:
            print(f"{result.phase.upper()}: {result.duration_ticks} ticks, "
                  f"{result.spikes_processed} spikes processed")
            if result.metrics:
                for key, value in result.metrics.items():
                    print(f"  {key}: {value}")

    return results


if __name__ == "__main__":
    results = run_pavlov_phase2(verbose=True)
    print("\n✅ Pavlov Phase 2 experiment completed successfully!")
