"""
Integration tests for Phase 2: Event-Driven Orchestrator

Verifies:
1. Orchestrator correctly coordinates spike routing
2. Learning can be enabled/disabled
3. Metrics collection works
4. Backward compatibility with Phase 1 modules
"""

import sys
sys.path.insert(0, r'D:\Node_network')

from core.neuron import Neuron
from core.synapse import Synapse
from core.spike import Spike
from core.orchestrator import Orchestrator


def test_orchestrator_spike_routing():
    """Test that orchestrator correctly routes spikes between neurons."""
    # Setup: Create two neurons with a synapse
    neuron_0 = Neuron(0, neuron_type="input", threshold=0.5)
    neuron_1 = Neuron(1, neuron_type="hidden", threshold=0.8)

    neurons = {0: neuron_0, 1: neuron_1}
    synapses = {1: {0: Synapse(weight=1.0, is_inhibitory=False)}}

    # Create orchestrator
    orch = Orchestrator(neurons, synapses)
    orch.register_connection(0, 1, 0)

    # Inject spike from neuron 0
    spike = Spike(source_id=0, timestamp=0, magnitude=1.0)
    orch.inject_spike(spike)

    # Step should process the spike
    processed = orch.step()
    assert processed, "Should process injected spike"
    assert orch.spikes_processed == 1, "Should count 1 spike processed"

    print("✓ test_orchestrator_spike_routing passed")


def test_orchestrator_event_causality():
    """Test that orchestrator enforces event causality."""
    neuron_0 = Neuron(0, neuron_type="input", threshold=0.5)
    neurons = {0: neuron_0}
    synapses = {}

    orch = Orchestrator(neurons, synapses)

    # Inject spike at t=5
    spike1 = Spike(source_id=0, timestamp=5, magnitude=1.0)
    orch.inject_spike(spike1)

    # Step to advance time
    orch.step()
    assert orch.current_tick == 5

    # Try to inject spike at t=3 (violates causality)
    spike2 = Spike(source_id=0, timestamp=3, magnitude=1.0)
    try:
        orch.inject_spike(spike2)
        assert False, "Should raise ValueError for retrograde spike"
    except ValueError as e:
        assert "past" in str(e).lower()

    print("✓ test_orchestrator_event_causality passed")


def test_orchestrator_learning_disable():
    """Test that learning can be disabled and re-enabled."""
    neuron_0 = Neuron(0, neuron_type="input")
    neuron_1 = Neuron(1, neuron_type="hidden")

    neurons = {0: neuron_0, 1: neuron_1}
    synapse = Synapse(weight=0.5)
    synapses = {1: {0: synapse}}

    orch = Orchestrator(neurons, synapses)

    # Learning enabled by default
    assert orch._learning_enabled == True
    initial_weight = synapse.weight

    # Disable learning
    orch.enable_learning(False)
    assert orch._learning_enabled == False

    # Re-enable
    orch.enable_learning(True)
    assert orch._learning_enabled == True

    print("✓ test_orchestrator_learning_disable passed")


def test_orchestrator_metrics():
    """Test that orchestrator collects metrics."""
    neuron_0 = Neuron(0, neuron_type="input", threshold=0.5)
    neurons = {0: neuron_0}
    synapses = {}

    orch = Orchestrator(neurons, synapses)

    # Check initial metrics
    metrics = orch.get_metrics()
    assert metrics["steps_executed"] == 0
    assert metrics["spikes_processed"] == 0
    assert metrics["learning_enabled"] == True

    # Inject and process spike
    spike = Spike(source_id=0, timestamp=0, magnitude=1.0)
    orch.inject_spike(spike)
    orch.step()

    # Check metrics after step
    metrics = orch.get_metrics()
    assert metrics["steps_executed"] == 1
    assert metrics["spikes_processed"] == 1
    assert metrics["current_tick"] == 0

    print("✓ test_orchestrator_metrics passed")


def test_orchestrator_run_until():
    """Test run_until() executes until timeout."""
    neuron_0 = Neuron(0, neuron_type="input", threshold=0.5)
    neurons = {0: neuron_0}
    synapses = {}

    orch = Orchestrator(neurons, synapses)

    # Inject spikes at different times
    for t in range(5):
        spike = Spike(source_id=0, timestamp=t, magnitude=1.0)
        orch.inject_spike(spike)

    # Run until t=3
    steps = orch.run_until(timeout_ticks=3)
    assert steps > 0
    assert orch.current_tick <= 3

    print("✓ test_orchestrator_run_until passed")


def test_phase1_module_compatibility():
    """Verify Phase 1 modules work with Orchestrator."""
    from core.event_queue import EventQueue
    from core.dispatcher import Dispatcher
    from core.plasticity import STDPRule, HomeostasisRule

    # All Phase 1 modules should be usable
    eq = EventQueue()
    assert eq is not None

    disp = Dispatcher()
    assert disp is not None

    stdp = STDPRule()
    assert stdp is not None

    homeostasis = HomeostasisRule()
    assert homeostasis is not None

    print("✓ test_phase1_module_compatibility passed")


if __name__ == "__main__":
    test_orchestrator_spike_routing()
    test_orchestrator_event_causality()
    test_orchestrator_learning_disable()
    test_orchestrator_metrics()
    test_orchestrator_run_until()
    test_phase1_module_compatibility()

    print("\n✅ All Phase 2 orchestrator integration tests passed!")
