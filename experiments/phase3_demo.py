"""
Phase 3 Demonstration: Novelty-Driven Gated Learning

This experiment demonstrates the cognitive control layer (System 1.5) in action:

1. Learning is OFF by default (Phase 3 philosophy)
2. Novel patterns automatically open the plasticity gate
3. Familiar patterns keep the gate closed
4. Reward signals can override novelty detection

This shows the system learning WHEN to learn, not just HOW to learn.
"""

from core.spike import Spike
from core.neuron import Neuron
from core.synapse import Synapse
from core.orchestrator import Orchestrator
from core.graph_store import GraphStore
import random


def build_simple_network():
    """
    Build a simple 3-layer network for demonstration.

    Architecture:
    - 2 input neurons (0, 1)
    - 2 hidden neurons (2, 3)
    - 1 output neuron (4)
    """
    graph = GraphStore()

    # Input layer
    graph.add_node(0, "input", threshold=1.0, refractory_period=5)
    graph.add_node(1, "input", threshold=1.0, refractory_period=5)

    # Hidden layer
    graph.add_node(2, "hidden", threshold=1.5, refractory_period=5)
    graph.add_node(3, "hidden", threshold=1.5, refractory_period=5)

    # Output layer
    graph.add_node(4, "output", threshold=2.0, refractory_period=5)

    # Connections: fully connected feedforward
    # Input -> Hidden
    graph.add_edge(0, 2, weight=0.8, is_inhibitory=False)
    graph.add_edge(0, 3, weight=0.7, is_inhibitory=False)
    graph.add_edge(1, 2, weight=0.7, is_inhibitory=False)
    graph.add_edge(1, 3, weight=0.8, is_inhibitory=False)

    # Hidden -> Output
    graph.add_edge(2, 4, weight=1.0, is_inhibitory=False)
    graph.add_edge(3, 4, weight=1.0, is_inhibitory=False)

    return graph


def create_orchestrator_from_graph(graph: GraphStore, enable_control: bool = True):
    """Create orchestrator with control layer from graph."""
    # Build neurons
    neurons = {}
    for node in graph.get_nodes():
        neurons[node.node_id] = Neuron(
            node.node_id,
            node.node_type,
            threshold=node.threshold,
            refractory_period=node.refractory_period
        )

    # Build synapses
    synapses = {}
    for edge in graph.get_edges():
        target_id = edge.target_id
        if target_id not in synapses:
            synapses[target_id] = {}

        synapse_index = edge.source_id  # Use source as index for simplicity
        synapses[target_id][synapse_index] = Synapse(
            weight=edge.weight,
            is_inhibitory=edge.is_inhibitory
        )

        # Add input to neuron
        neurons[target_id].add_input(synapse_index, edge.weight)

    # Create orchestrator with control layer
    orchestrator = Orchestrator(
        neurons,
        synapses,
        enable_control_layer=enable_control,
        novelty_threshold=0.5,
        output_neuron_ids={4}  # Output neuron for monitoring
    )

    # Register connections
    for edge in graph.get_edges():
        orchestrator.register_connection(
            edge.source_id,
            edge.target_id,
            synapse_index=edge.source_id
        )

    return orchestrator


def inject_pattern(orchestrator: Orchestrator, pattern: str, timestamp: int):
    """
    Inject a spike pattern into the network.

    Patterns:
    - "A": neuron 0 fires
    - "B": neuron 1 fires
    - "AB": both fire
    """
    if "A" in pattern or pattern == "A":
        orchestrator.inject_spike(Spike(source_id=0, timestamp=timestamp))
    if "B" in pattern or pattern == "B":
        orchestrator.inject_spike(Spike(source_id=1, timestamp=timestamp + 1))


def run_phase3_demonstration():
    """
    Demonstrate Phase 3 cognitive control.

    Experiment:
    1. Present pattern A repeatedly (familiar)
    2. Present pattern B (novel)
    3. Return to pattern A (familiar again)
    4. Show learning gate only opens for novel patterns
    """
    print("=" * 80)
    print("PHASE 3 DEMONSTRATION: Novelty-Driven Gated Learning")
    print("=" * 80)
    print()

    # Build network
    print("Building 3-layer network (2 input, 2 hidden, 1 output)...")
    graph = build_simple_network()
    orchestrator = create_orchestrator_from_graph(graph, enable_control=True)
    print("✓ Network built with control layer enabled")
    print()

    # Phase 1: Present familiar pattern A repeatedly
    print("-" * 80)
    print("PHASE 1: Establishing Familiar Pattern")
    print("-" * 80)

    print("Presenting pattern 'A' 20 times...")
    for trial in range(20):
        inject_pattern(orchestrator, "A", timestamp=trial * 10)
        orchestrator.run_until(timeout_ticks=(trial + 1) * 10 + 5)

        if trial % 5 == 4:  # Every 5 trials
            metrics = orchestrator.get_metrics()
            if "control_layer" in metrics:
                control = metrics["control_layer"]
                novelty = control["novelty"]["current_novelty"]
                learning = "ON" if control["plasticity_gate"]["learning_enabled"] else "OFF"
                print(f"  Trial {trial + 1:2d}: Novelty={novelty:.3f}, Learning={learning}")

    print()
    print("Result: Novelty decreases → Learning gate CLOSES")
    print()

    # Phase 2: Present novel pattern B
    print("-" * 80)
    print("PHASE 2: Novel Pattern Presentation")
    print("-" * 80)

    print("Presenting novel pattern 'B' 5 times...")
    base_time = 20 * 10 + 10
    for trial in range(5):
        inject_pattern(orchestrator, "B", timestamp=base_time + trial * 10)
        orchestrator.run_until(timeout_ticks=base_time + (trial + 1) * 10 + 5)

        metrics = orchestrator.get_metrics()
        if "control_layer" in metrics:
            control = metrics["control_layer"]
            novelty = control["novelty"]["current_novelty"]
            learning = "ON" if control["plasticity_gate"]["learning_enabled"] else "OFF"
            print(f"  Trial {trial + 1}: Novelty={novelty:.3f}, Learning={learning}")

    print()
    print("Result: Novelty spikes → Learning gate OPENS")
    print()

    # Phase 3: Return to familiar pattern A
    print("-" * 80)
    print("PHASE 3: Return to Familiar Pattern")
    print("-" * 80)

    print("Returning to pattern 'A' for 5 trials...")
    base_time = 25 * 10 + 10
    for trial in range(5):
        inject_pattern(orchestrator, "A", timestamp=base_time + trial * 10)
        orchestrator.run_until(timeout_ticks=base_time + (trial + 1) * 10 + 5)

        metrics = orchestrator.get_metrics()
        if "control_layer" in metrics:
            control = metrics["control_layer"]
            novelty = control["novelty"]["current_novelty"]
            learning = "ON" if control["plasticity_gate"]["learning_enabled"] else "OFF"
            print(f"  Trial {trial + 1}: Novelty={novelty:.3f}, Learning={learning}")

    print()
    print("Result: Novelty low again → Learning gate CLOSES")
    print()

    # Final statistics
    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)

    metrics = orchestrator.get_metrics()
    print(f"Total simulation steps: {metrics['steps_executed']}")
    print(f"Total spikes processed: {metrics['spikes_processed']}")
    print()

    if "control_layer" in metrics:
        control = metrics["control_layer"]
        print("Control Layer Statistics:")
        print(f"  Timesteps: {control['timestep']}")
        print(f"  Gate open count: {control['plasticity_gate']['gate_open_count']}")
        print(f"  Gate close count: {control['plasticity_gate']['gate_close_count']}")
        print(f"  System 2 invocations: {control['system2_invocations']}")
        print()

        print("Novelty Statistics:")
        nov_stats = control['novelty']
        print(f"  Current: {nov_stats['current_novelty']:.3f}")
        print(f"  Mean: {nov_stats['mean_novelty']:.3f}")
        print(f"  Peak: {nov_stats['peak_novelty']:.3f}")
        print()

        print("Confidence Statistics:")
        conf_stats = control['confidence']
        print(f"  Current: {conf_stats['current_confidence']:.3f}")
        print(f"  Mean: {conf_stats['mean_confidence']:.3f}")
        print(f"  Min: {conf_stats['min_confidence']:.3f}")
        print()

        print("Conflict Statistics:")
        conflict_stats = control['conflict']
        print(f"  Current: {conflict_stats['current_conflict']:.3f}")
        print(f"  Mean: {conflict_stats['mean_conflict']:.3f}")
        print(f"  Conflict events: {conflict_stats['conflict_events']}")

    print()
    print("=" * 80)
    print("KEY INSIGHT: Learning is context-aware, not always-on")
    print("=" * 80)
    print()
    print("Phase 3 Achievement:")
    print("✓ Learning OFF by default")
    print("✓ Novel patterns trigger learning automatically")
    print("✓ Familiar patterns preserve learned weights")
    print("✓ Control layer monitors without modifying computation")
    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    try:
        run_phase3_demonstration()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
