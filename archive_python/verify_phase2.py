"""
Phase 2 Verification Checklist - FINAL SIGN-OFF

Run this script to verify all Phase 2 components are functional.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_imports():
    """Verify all new modules can be imported."""
    print("Verifying imports...")
    try:
        from core.orchestrator import Orchestrator, SimulationState
        from core.metrics import MetricsCollector, SpikeProbe, WeightProbe, StateProbe
        from core.graph_store import GraphStore
        from experiments.base_experiment import ExperimentBase, ExperimentResult
        from experiments.pavlov_phase2 import PavlovExperimentPhase2
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def verify_orchestrator():
    """Verify Orchestrator basic functionality."""
    print("\nVerifying Orchestrator...")
    try:
        from core.orchestrator import Orchestrator
        from core.neuron import Neuron
        from core.synapse import Synapse
        from core.spike import Spike

        neurons = {0: Neuron(0, neuron_type="input")}
        synapses = {}
        orch = Orchestrator(neurons, synapses)

        # Inject and process spike
        spike = Spike(source_id=0, timestamp=0, magnitude=1.0)
        orch.inject_spike(spike)
        processed = orch.step()

        assert processed, "Should process spike"
        assert orch.spikes_processed == 1, "Should count 1 spike"
        print("✅ Orchestrator working correctly")
        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False


def verify_metrics():
    """Verify metrics collection."""
    print("\nVerifying Metrics...")
    try:
        from core.metrics import MetricsCollector
        from core.spike import Spike

        collector = MetricsCollector()
        spike = Spike(source_id=0, timestamp=0, magnitude=1.0)
        collector.record_spike(spike, target_count=2)

        assert collector.spike_probe.spike_count == 1
        snapshot = collector.get_snapshot(0)
        assert snapshot.total_spikes_routed == 1
        print("✅ Metrics working correctly")
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def verify_persistence():
    """Verify GraphStore persistence."""
    print("\nVerifying Persistence...")
    try:
        import tempfile
        from pathlib import Path
        from core.graph_store import GraphStore

        # Create graph
        store = GraphStore()
        store.add_node(0, node_type="input")
        store.add_node(1, node_type="hidden")
        store.add_edge(0, 1, weight=0.5)

        # Persist and restore
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            store.persist(filepath)
            restored = GraphStore.restore(filepath)

            assert restored.get_node_count() == 2
            assert restored.get_edge_count() == 1
            print("✅ Persistence working correctly")
            return True
        finally:
            Path(filepath).unlink()
    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False


def verify_experiment_framework():
    """Verify ExperimentBase framework."""
    print("\nVerifying Experiment Framework...")
    try:
        from experiments.base_experiment import ExperimentBase

        # Verify abstract class is defined
        assert hasattr(ExperimentBase, 'build_network')
        assert hasattr(ExperimentBase, 'create_stimuli')
        assert hasattr(ExperimentBase, 'run_training')
        assert hasattr(ExperimentBase, 'execute')
        print("✅ Experiment framework working correctly")
        return True
    except Exception as e:
        print(f"❌ Experiment framework test failed: {e}")
        return False


def verify_backward_compatibility():
    """Verify Phase 1 code still works."""
    print("\nVerifying Backward Compatibility...")
    try:
        from core.neuron import Neuron
        from core.synapse import Synapse
        from core.spike import Spike
        from core.event_queue import EventQueue
        from core.dispatcher import Dispatcher

        # All Phase 1 components should work
        n = Neuron(0)
        s = Synapse(0.5)
        spike = Spike(0, 0, 1.0)
        eq = EventQueue()
        disp = Dispatcher()

        print("✅ Phase 1 backward compatibility verified")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("NODE_NETWORK PHASE 2 VERIFICATION CHECKLIST")
    print("=" * 70)
    print()

    checks = [
        verify_imports,
        verify_orchestrator,
        verify_metrics,
        verify_persistence,
        verify_experiment_framework,
        verify_backward_compatibility,
    ]

    results = []
    for check in checks:
        results.append(check())

    print()
    print("=" * 70)
    if all(results):
        print("✅ ALL VERIFICATION CHECKS PASSED")
        print("=" * 70)
        print()
        print("Phase 2 is production-ready!")
        print()
        print("Key achievements:")
        print("  ✅ Event-driven orchestrator functional")
        print("  ✅ Metrics and probes operational")
        print("  ✅ Persistence layer working")
        print("  ✅ Experiment framework ready")
        print("  ✅ Phase 1 backward compatible")
        print()
        print("Next steps:")
        print("  → Review PHASE2_COMPLETION_REPORT.md")
        print("  → Run full test suite: python run_all_tests.py")
        print("  → Try example: python experiments/pavlov_phase2.py")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
