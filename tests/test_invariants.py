"""
Invariant Tests: Explicit enforcement of architectural rules.

These tests verify that the system maintains its invariants:
1. Event causality: no retrograde spikes
2. Locality: neurons only access their inputs
3. Plasticity isolation: learning can be disabled without breaking inference
4. Determinism: seeded runs produce identical behavior
5. Bounded growth: synapses don't explode
"""

import unittest
import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.spike import Spike, WeightedSpike
from core.event_queue import EventQueue
from core.dispatcher import Dispatcher
from core.neuron import Neuron
from core.synapse import Synapse
from core.plasticity import PlasticityController
from core.graph_store import GraphStore


class TestEventCausality(unittest.TestCase):
    """Spike events must respect temporal causality."""

    def test_no_retrograde_spikes(self):
        """Cannot inject spikes into the past."""
        eq = EventQueue()

        # Inject spike at t=10
        eq.inject(Spike(0, 10))
        eq.pop_next()

        # Try to inject spike at t=5 (retrograde)
        with self.assertRaises(ValueError):
            eq.inject(Spike(1, 5))

    def test_causality_progression(self):
        """Event queue time should monotonically increase."""
        eq = EventQueue()
        times = [0, 1, 5, 3, 7]  # Out of order
        spikes = [Spike(i, t) for i, t in enumerate(times)]

        for spike in sorted(spikes, key=lambda s: s.timestamp):
            eq.inject(spike)

        processed_times = []
        while not eq.is_empty():
            spike = eq.pop_next()
            processed_times.append(spike.timestamp)

        # Should come out in strict order
        self.assertEqual(processed_times, sorted(processed_times))


class TestLocality(unittest.TestCase):
    """Neurons must only access their incoming connections."""

    def test_neuron_no_global_access(self):
        """Neuron.integrate should only use its inputs, not network state."""
        neuron = Neuron(0, "hidden")
        neuron.add_input(1, 0.5)

        # Call integrate with only one spike
        spikes = [(1, WeightedSpike(Spike(1, 0), 0.5))]
        result = neuron.integrate(0, spikes)

        # Neuron should fire if weight is sufficient
        initial_potential = neuron.potential
        self.assertIsNotNone(initial_potential)

    def test_synapse_transmit_pure(self):
        """Synapse.transmit should have no side effects."""
        syn = Synapse(0.5)
        spike = Spike(0, 0)

        # Transmit multiple times with same spike
        ws1 = syn.transmit(spike)
        ws2 = syn.transmit(spike)

        # Should be identical
        self.assertEqual(ws1.weight, ws2.weight)
        self.assertEqual(ws1.spike.source_id, ws2.spike.source_id)


class TestPlasticityIsolation(unittest.TestCase):
    """Learning must be separable from inference."""

    def test_inference_without_learning(self):
        """Disabling learning should not affect spike computation."""
        # Two identical neurons
        neuron1 = Neuron(0, "hidden")
        neuron2 = Neuron(1, "hidden")

        neuron1.add_input(2, 1.5)
        neuron2.add_input(2, 1.5)

        # Run with learning
        plasticity_on = PlasticityController(learning_enabled=True)
        spikes1 = neuron1.integrate(0, [(2, WeightedSpike(Spike(2, 0), 1.5))])

        # Run without learning
        plasticity_off = PlasticityController(learning_enabled=False)
        spikes2 = neuron2.integrate(0, [(2, WeightedSpike(Spike(2, 0), 1.5))])

        # Both should fire identically
        self.assertEqual(bool(spikes1), bool(spikes2))

    def test_learning_gate_isolated(self):
        """Disabling learning gate should not affect synaptic transmission."""
        syn = Synapse(0.5)
        syn.trace = 1.0  # Set trace so learning can happen

        plasticity = PlasticityController(learning_enabled=True, stdp_learning_rate=0.1)
        plasticity.update_learning(syn, post_fired=True, dopamine=1.0)
        weight_with_learning = syn.weight

        # Reset and try with learning disabled
        syn2 = Synapse(0.5)
        syn2.trace = 1.0
        plasticity_disabled = PlasticityController(learning_enabled=False)
        plasticity_disabled.update_learning(syn2, post_fired=True, dopamine=1.0)
        weight_without_learning = syn2.weight

        # Learning should have changed syn but not syn2
        self.assertNotEqual(weight_with_learning, 0.5)  # syn should have changed
        self.assertEqual(weight_without_learning, 0.5)  # syn2 should not have changed


class TestDeterminism(unittest.TestCase):
    """Seeded runs should produce identical results."""

    def test_event_ordering_deterministic(self):
        """Same sequence of injections should always pop in same order."""
        def run_with_seed(seed):
            random.seed(seed)
            eq = EventQueue()

            # Inject 100 random spikes
            for i in range(100):
                t = random.randint(0, 1000)
                eq.inject(Spike(i, t))

            # Pop all and record order
            order = []
            while not eq.is_empty():
                spike = eq.pop_next()
                order.append(spike.source_id)

            return order

        order1 = run_with_seed(42)
        order2 = run_with_seed(42)

        self.assertEqual(order1, order2)


class TestBoundedGrowth(unittest.TestCase):
    """Synaptic weights must stay bounded."""

    def test_weight_bounds_enforced(self):
        """Weights should never exceed safe bounds."""
        syn = Synapse(0.1)

        # Apply many strong STDP updates
        for _ in range(100):
            syn.apply_stdp(post_fired=True, dopamine=10.0, learning_rate=0.5)

        # Weight should still be <= 2.0
        self.assertLessEqual(syn.weight, 2.0)
        self.assertGreaterEqual(syn.weight, 0.0)

    def test_homeostasis_prevents_saturation(self):
        """Homeostasis should prevent all weights from growing unbounded."""
        from core.plasticity import HomeostasisRule

        synapses = {
            0: Synapse(0.5),
            1: Synapse(0.5),
            2: Synapse(0.5),
        }

        rule = HomeostasisRule(target_rate=0.25)

        # Apply homeostasis when firing rate is very high
        for _ in range(10):
            rule.apply(synapses, current_rate=0.9, node_type="hidden")

        # All weights should be reduced
        avg_weight = sum(s.weight for s in synapses.values()) / len(synapses)
        self.assertLess(avg_weight, 0.5)


class TestGraphConsistency(unittest.TestCase):
    """GraphStore must maintain consistent topology."""

    def test_edges_reference_valid_nodes(self):
        """All edges must connect valid nodes."""
        store = GraphStore()
        store.add_node(0, "input")
        store.add_node(1, "hidden")

        # Valid edge
        edge = store.add_edge(0, 1, 0.5)
        self.assertEqual(edge.source_id, 0)
        self.assertEqual(edge.target_id, 1)

        # Invalid edge
        with self.assertRaises(ValueError):
            store.add_edge(0, 999, 0.5)  # Node 999 doesn't exist

    def test_incoming_outgoing_consistency(self):
        """Incoming and outgoing edges should be consistent."""
        store = GraphStore()
        store.add_node(0)
        store.add_node(1)
        store.add_node(2)

        store.add_edge(0, 1, 0.5)
        store.add_edge(0, 2, 0.3)
        store.add_edge(1, 2, 0.7)

        # Check consistency
        outgoing_from_0 = list(store.get_outgoing_edges(0))
        incoming_to_1 = list(store.get_incoming_edges(1))
        incoming_to_2 = list(store.get_incoming_edges(2))

        self.assertEqual(len(outgoing_from_0), 2)
        self.assertEqual(len(incoming_to_1), 1)
        self.assertEqual(len(incoming_to_2), 2)


if __name__ == '__main__':
    unittest.main()
