"""
Tests for Phase 2: Metrics and Persistence

Verifies:
1. Metrics collection doesn't interfere with computation
2. GraphStore persistence (save/load round-trip)
3. Probe attachment and callback execution
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, r'D:\Node_network')

from core.metrics import MetricsCollector, SpikeProbe, WeightProbe, StateProbe
from core.graph_store import GraphStore
from core.spike import Spike


def test_metrics_collector_recording():
    """Test that metrics collector records events correctly."""
    collector = MetricsCollector()

    # Record a spike
    spike = Spike(source_id=0, timestamp=5, magnitude=1.0)
    collector.record_spike(spike, target_count=3)

    assert collector.spike_probe.spike_count == 1
    assert len(collector.spike_probe.spike_log) == 1

    snapshot = collector.get_snapshot(5)
    assert snapshot.spike_count == 1
    assert snapshot.total_spikes_routed == 1

    print("✓ test_metrics_collector_recording passed")


def test_metrics_spike_probe():
    """Test SpikeProbe functionality."""
    probe = SpikeProbe()

    # Record several spikes
    for i in range(5):
        spike = Spike(source_id=i % 2, timestamp=i, magnitude=1.0)
        probe.record_spike(spike, target_count=2)

    assert probe.spike_count == 5

    # Get firing rate
    rate_0 = probe.get_firing_rate(neuron_id=0, total_ticks=10)
    assert 0 < rate_0 < 1

    # Get fanout stats
    stats = probe.get_fanout_stats()
    assert stats["mean"] == 2.0
    assert stats["min"] == 2
    assert stats["max"] == 2

    print("✓ test_metrics_spike_probe passed")


def test_metrics_weight_probe():
    """Test WeightProbe functionality."""
    probe = WeightProbe()

    # Record weight updates
    probe.record_update(
        target_id=1,
        synapse_index=0,
        old_weight=0.5,
        new_weight=0.6,
        timestamp=10
    )

    probe.record_update(
        target_id=1,
        synapse_index=0,
        old_weight=0.6,
        new_weight=0.7,
        timestamp=20
    )

    assert probe.weight_updates == 2

    # Get learning curve
    curve = probe.get_learning_curve(target_id=1, synapse_index=0)
    assert curve == [0.6, 0.7]

    # Get stats
    stats = probe.get_weight_stats()
    assert abs(stats["mean_delta"] - 0.1) < 0.001

    print("✓ test_metrics_weight_probe passed")


def test_metrics_state_probe():
    """Test StateProbe functionality."""
    probe = StateProbe()

    # Record states
    for t in range(5):
        probe.record_state(
            neuron_id=0,
            potential=float(t) * 0.2,
            is_firing=(t == 3),
            timestamp=t
        )

    assert probe.state_samples == 5

    # Get neuron trace
    trace = probe.get_neuron_trace(neuron_id=0)
    assert len(trace) == 5

    # Get firing events
    fires = probe.get_firing_events(neuron_id=0)
    assert fires == [3]

    print("✓ test_metrics_state_probe passed")


def test_metrics_callbacks():
    """Test that metrics callbacks are invoked."""
    collector = MetricsCollector()

    spike_calls = []
    def spike_callback(spike, count):
        spike_calls.append((spike.source_id, count))

    collector.attach_spike_callback(spike_callback)

    spike = Spike(source_id=5, timestamp=10, magnitude=1.0)
    collector.record_spike(spike, target_count=2)

    assert len(spike_calls) == 1
    assert spike_calls[0] == (5, 2)

    print("✓ test_metrics_callbacks passed")


def test_graphstore_persist():
    """Test GraphStore persistence to file."""
    # Create a small graph
    store = GraphStore()
    store.add_node(0, node_type="input", threshold=0.5, refractory_period=2)
    store.add_node(1, node_type="hidden", threshold=1.0, refractory_period=3)
    store.add_node(2, node_type="output", threshold=1.5, refractory_period=3)

    store.add_edge(0, 1, weight=0.8, is_inhibitory=False)
    store.add_edge(1, 2, weight=0.6, is_inhibitory=False)
    store.add_edge(0, 2, weight=0.3, is_inhibitory=True)

    # Persist to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name

    try:
        store.persist(filepath)
        assert Path(filepath).exists()

        # Verify file is valid JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
            assert "nodes" in data
            assert "edges" in data
            assert len(data["nodes"]) == 3
            assert len(data["edges"]) == 3

        print("✓ test_graphstore_persist passed")
    finally:
        Path(filepath).unlink()


def test_graphstore_restore():
    """Test GraphStore restore from file."""
    # Create original graph
    store1 = GraphStore()
    store1.add_node(0, node_type="input", threshold=0.5, refractory_period=2)
    store1.add_node(1, node_type="hidden", threshold=1.0, refractory_period=3)
    store1.add_edge(0, 1, weight=0.8, is_inhibitory=False)

    # Persist
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name

    try:
        store1.persist(filepath)

        # Restore
        store2 = GraphStore.restore(filepath)

        # Verify topology matches
        assert store2.get_node_count() == 2
        assert store2.get_edge_count() == 1

        node0 = store2.get_node(0)
        assert node0.node_type == "input"
        assert node0.threshold == 0.5

        edges = list(store2.get_edges())
        assert edges[0].source_id == 0
        assert edges[0].target_id == 1
        assert edges[0].weight == 0.8
        assert edges[0].is_inhibitory == False

        print("✓ test_graphstore_restore passed")
    finally:
        Path(filepath).unlink()


def test_graphstore_round_trip():
    """Test that save→load produces identical graph."""
    # Create complex graph
    store1 = GraphStore()
    for i in range(5):
        store1.add_node(i, node_type="hidden", threshold=1.0)

    for i in range(4):
        store1.add_edge(i, i+1, weight=0.5, is_inhibitory=(i % 2 == 0))

    # Save and load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name

    try:
        store1.persist(filepath)
        store2 = GraphStore.restore(filepath)

        # Verify equivalence
        assert store1.get_node_count() == store2.get_node_count()
        assert store1.get_edge_count() == store2.get_edge_count()

        edges1 = sorted([(e.source_id, e.target_id) for e in store1.get_edges()])
        edges2 = sorted([(e.source_id, e.target_id) for e in store2.get_edges()])
        assert edges1 == edges2

        print("✓ test_graphstore_round_trip passed")
    finally:
        Path(filepath).unlink()


if __name__ == "__main__":
    test_metrics_collector_recording()
    test_metrics_spike_probe()
    test_metrics_weight_probe()
    test_metrics_state_probe()
    test_metrics_callbacks()
    test_graphstore_persist()
    test_graphstore_restore()
    test_graphstore_round_trip()

    print("\n✅ All Phase 2 metrics and persistence tests passed!")
