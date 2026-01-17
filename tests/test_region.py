"""
Phase 3.2 Region Architecture Tests

Tests for region-based spatial organization:
1. Region creation and neuron ownership
2. Port-based inter-region communication
3. Region freezing/unfreezing
4. Per-region plasticity control
5. Regional metrics
6. Region network coordination

Invariants:
- Regions own their neurons (no sharing)
- Connections between regions only through ports
- Frozen regions cannot learn
- Region metrics are local
- Region network coordinates without global state
"""

import unittest
from core.neuron import Neuron
from core.synapse import Synapse
from core.plasticity import PlasticityController
from core.region import Region, RegionType, RegionNetwork, RegionPort


class TestRegionBasics(unittest.TestCase):
    """Test basic region creation and properties."""

    def setUp(self):
        """Set up test region."""
        self.neurons = {
            0: Neuron(0, "hidden"),
            1: Neuron(1, "hidden"),
            2: Neuron(2, "hidden"),
        }
        self.synapses = {
            1: {0: Synapse(0.8)},
            2: {1: Synapse(0.9)},
        }
        self.region = Region(
            region_id=0,
            region_type=RegionType.HIDDEN,
            neurons=self.neurons,
            synapses=self.synapses
        )

    def test_region_creation(self):
        """Region should be created with correct properties."""
        self.assertEqual(self.region.region_id, 0)
        self.assertEqual(self.region.region_type, RegionType.HIDDEN)
        self.assertEqual(self.region.get_size(), 3)

    def test_region_neuron_ownership(self):
        """Region should own its neurons."""
        self.assertTrue(self.region.contains_neuron(0))
        self.assertTrue(self.region.contains_neuron(1))
        self.assertTrue(self.region.contains_neuron(2))
        self.assertFalse(self.region.contains_neuron(99))

    def test_region_connectivity(self):
        """Region should report connectivity count."""
        self.assertEqual(self.region.get_connectivity_count(), 2)

    def test_region_neuron_ids(self):
        """Region should return all neuron IDs."""
        neuron_ids = self.region.get_neuron_ids()
        self.assertEqual(neuron_ids, {0, 1, 2})


class TestRegionPorts(unittest.TestCase):
    """Test port-based inter-region communication."""

    def setUp(self):
        """Set up region with ports."""
        neurons = {0: Neuron(0, "hidden"), 1: Neuron(1, "hidden")}
        synapses = {}
        self.region = Region(
            region_id=0,
            region_type=RegionType.HIDDEN,
            neurons=neurons,
            synapses=synapses
        )

    def test_add_input_port(self):
        """Should add input port to neuron."""
        self.region.add_input_port(port_id=0, neuron_id=0)
        ports = self.region.get_input_ports()

        self.assertIn(0, ports)
        self.assertEqual(ports[0].neuron_id, 0)
        self.assertEqual(ports[0].direction, "in")

    def test_add_output_port(self):
        """Should add output port from neuron."""
        self.region.add_output_port(port_id=0, neuron_id=1)
        ports = self.region.get_output_ports()

        self.assertIn(0, ports)
        self.assertEqual(ports[0].neuron_id, 1)
        self.assertEqual(ports[0].direction, "out")

    def test_invalid_port_neuron(self):
        """Should reject ports for non-existent neurons."""
        with self.assertRaises(ValueError):
            self.region.add_input_port(port_id=0, neuron_id=99)

    def test_multiple_ports(self):
        """Region should support multiple ports."""
        self.region.add_input_port(port_id=0, neuron_id=0)
        self.region.add_input_port(port_id=1, neuron_id=1)
        self.region.add_output_port(port_id=0, neuron_id=0)

        input_ports = self.region.get_input_ports()
        output_ports = self.region.get_output_ports()

        self.assertEqual(len(input_ports), 2)
        self.assertEqual(len(output_ports), 1)


class TestRegionFreezing(unittest.TestCase):
    """Test region freezing and learning control."""

    def setUp(self):
        """Set up region."""
        neurons = {0: Neuron(0, "hidden"), 1: Neuron(1, "hidden")}
        synapses = {1: {0: Synapse(0.5)}}
        self.region = Region(
            region_id=0,
            region_type=RegionType.HIDDEN,
            neurons=neurons,
            synapses=synapses
        )

    def test_initial_not_frozen(self):
        """Region should not be frozen initially."""
        self.assertFalse(self.region.is_frozen())

    def test_freeze_region(self):
        """Freezing should disable learning."""
        self.assertTrue(self.region.is_learning())
        self.region.freeze()
        self.assertTrue(self.region.is_frozen())
        self.assertFalse(self.region.is_learning())

    def test_unfreeze_region(self):
        """Unfreezing should re-enable learning."""
        self.region.freeze()
        self.region.unfreeze()
        self.assertFalse(self.region.is_frozen())
        self.assertTrue(self.region.is_learning())

    def test_freeze_prevents_learning(self):
        """Frozen region should not apply plasticity."""
        self.region.freeze()
        original_weight = self.region.synapses[1][0].weight

        # Try to enable learning while frozen
        self.region.enable_plasticity()

        # Learning should still be disabled
        self.assertFalse(self.region.is_learning())


class TestRegionPlasticity(unittest.TestCase):
    """Test per-region plasticity control."""

    def setUp(self):
        """Set up region."""
        neurons = {0: Neuron(0, "hidden"), 1: Neuron(1, "hidden")}
        synapses = {}
        self.region = Region(
            region_id=0,
            region_type=RegionType.HIDDEN,
            neurons=neurons,
            synapses=synapses
        )

    def test_learning_initially_on(self):
        """Learning should be enabled initially."""
        self.assertTrue(self.region.is_learning())

    def test_disable_plasticity(self):
        """Should disable learning without freezing."""
        self.region.disable_plasticity()
        self.assertFalse(self.region.is_learning())
        self.assertFalse(self.region.is_frozen())

    def test_enable_plasticity(self):
        """Should re-enable learning."""
        self.region.disable_plasticity()
        self.region.enable_plasticity()
        self.assertTrue(self.region.is_learning())

    def test_cannot_enable_when_frozen(self):
        """Cannot enable plasticity when frozen."""
        self.region.freeze()
        self.region.enable_plasticity()
        self.assertFalse(self.region.is_learning())


class TestRegionMetrics(unittest.TestCase):
    """Test regional metrics collection."""

    def setUp(self):
        """Set up region."""
        neurons = {
            0: Neuron(0, "hidden"),
            1: Neuron(1, "hidden"),
            2: Neuron(2, "hidden"),
        }
        synapses = {
            1: {0: Synapse(0.8)},
            2: {1: Synapse(0.6)},
        }
        self.region = Region(
            region_id=0,
            region_type=RegionType.HIDDEN,
            neurons=neurons,
            synapses=synapses
        )

    def test_metrics_initial_state(self):
        """Metrics should initialize properly."""
        metrics = self.region.get_metrics()
        self.assertEqual(metrics.firing_rate, 0.0)
        self.assertGreater(metrics.mean_weight, 0)

    def test_update_metrics_with_firing(self):
        """Metrics should update with firing patterns."""
        firing = {0, 1}  # Two neurons firing
        self.region.update_metrics(firing, timestamp=0)

        metrics = self.region.get_metrics()
        self.assertGreater(metrics.firing_rate, 0)
        self.assertEqual(metrics.active_neurons, 2)
        self.assertEqual(metrics.timestamp, 0)

    def test_update_metrics_multiple_times(self):
        """Firing rate should be EMA over time."""
        # First update
        self.region.update_metrics({0}, timestamp=0)
        rate1 = self.region.get_metrics().firing_rate

        # Second update
        self.region.update_metrics({0, 1}, timestamp=1)
        rate2 = self.region.get_metrics().firing_rate

        # Rate should increase (EMA of 0.33, 0.67)
        self.assertGreater(rate2, rate1)

    def test_weight_statistics(self):
        """Metrics should compute weight statistics."""
        self.region.update_metrics({0}, timestamp=0)
        metrics = self.region.get_metrics()

        # Should compute mean and variance
        self.assertGreater(metrics.mean_weight, 0)
        self.assertGreaterEqual(metrics.weight_variance, 0)

    def test_statistics_dictionary(self):
        """Should return comprehensive statistics."""
        self.region.update_metrics({0, 1}, timestamp=5)
        stats = self.region.get_statistics()

        self.assertIn("region_id", stats)
        self.assertIn("neurons", stats)
        self.assertIn("synapses", stats)
        self.assertIn("metrics", stats)
        self.assertEqual(stats["region_id"], 0)
        self.assertEqual(stats["neurons"], 3)


class TestRegionNetwork(unittest.TestCase):
    """Test multi-region networks."""

    def setUp(self):
        """Set up region network."""
        self.network = RegionNetwork()

        # Create sensory region
        sensory_neurons = {0: Neuron(0, "input"), 1: Neuron(1, "input")}
        sensory = Region(
            region_id=0,
            region_type=RegionType.SENSORY,
            neurons=sensory_neurons,
            synapses={}
        )
        sensory.add_output_port(0, 0)
        sensory.add_output_port(1, 1)

        # Create hidden region
        hidden_neurons = {2: Neuron(2, "hidden"), 3: Neuron(3, "hidden")}
        hidden = Region(
            region_id=1,
            region_type=RegionType.HIDDEN,
            neurons=hidden_neurons,
            synapses={}
        )
        hidden.add_input_port(0, 2)
        hidden.add_input_port(1, 3)

        # Create motor region
        motor_neurons = {4: Neuron(4, "output")}
        motor = Region(
            region_id=2,
            region_type=RegionType.MOTOR,
            neurons=motor_neurons,
            synapses={}
        )
        motor.add_input_port(0, 4)

        self.network.add_region(sensory)
        self.network.add_region(hidden)
        self.network.add_region(motor)

    def test_add_regions(self):
        """Network should add regions."""
        self.assertEqual(len(self.network.get_regions()), 3)

    def test_get_region(self):
        """Should retrieve regions by ID."""
        region = self.network.get_region(0)
        self.assertIsNotNone(region)
        self.assertEqual(region.region_id, 0)

    def test_duplicate_region_id(self):
        """Should reject duplicate region IDs."""
        region = Region(
            region_id=0,
            region_type=RegionType.SENSORY,
            neurons={5: Neuron(5, "input")},
            synapses={}
        )

        with self.assertRaises(ValueError):
            self.network.add_region(region)

    def test_connect_regions(self):
        """Should connect regions through ports."""
        self.network.connect_regions(
            source_region_id=0,
            target_region_id=1,
            source_port_id=0,
            target_port_id=0
        )

        # Connection should be recorded
        self.assertGreater(len(self.network._inter_region_connections), 0)

    def test_freeze_region(self):
        """Network should freeze regions."""
        region = self.network.get_region(0)
        region.enable_plasticity()

        self.network.freeze_region(0)

        self.assertTrue(region.is_frozen())
        self.assertFalse(region.is_learning())

    def test_unfreeze_region(self):
        """Network should unfreeze regions."""
        self.network.freeze_region(0)
        self.network.unfreeze_region(0)

        region = self.network.get_region(0)
        self.assertFalse(region.is_frozen())

    def test_freeze_all(self):
        """Should freeze all regions."""
        self.network.freeze_all()

        for region in self.network.get_regions().values():
            self.assertTrue(region.is_frozen())

    def test_enable_learning_global(self):
        """Should enable learning in all regions."""
        self.network.disable_learning()
        self.network.enable_learning()

        for region in self.network.get_regions().values():
            if not region.is_frozen():
                self.assertTrue(region.is_learning())

    def test_enable_learning_regional(self):
        """Should enable learning in specific region."""
        self.network.disable_learning()
        self.network.enable_learning(region_id=0)

        region0 = self.network.get_region(0)
        region1 = self.network.get_region(1)

        self.assertTrue(region0.is_learning())
        self.assertFalse(region1.is_learning())

    def test_update_all_metrics(self):
        """Network should update metrics for all regions."""
        firing = {0, 2, 4}
        self.network.update_all_metrics(firing, timestamp=10)

        for region in self.network.get_regions().values():
            metrics = region.get_metrics()
            self.assertEqual(metrics.timestamp, 10)

    def test_global_statistics(self):
        """Network should provide global statistics."""
        stats = self.network.get_global_statistics()

        self.assertEqual(stats["num_regions"], 3)
        self.assertGreater(stats["total_neurons"], 0)
        self.assertIn("regions", stats)


class TestRegionIsolation(unittest.TestCase):
    """Test region isolation invariants."""

    def test_regions_independent_plasticity(self):
        """Regions should have independent plasticity."""
        neurons1 = {0: Neuron(0, "hidden")}
        neurons2 = {1: Neuron(1, "hidden")}

        region1 = Region(0, RegionType.HIDDEN, neurons1, {})
        region2 = Region(1, RegionType.HIDDEN, neurons2, {})

        region1.disable_plasticity()

        self.assertFalse(region1.is_learning())
        self.assertTrue(region2.is_learning())

    def test_regions_independent_freezing(self):
        """Freezing one region should not freeze others."""
        neurons1 = {0: Neuron(0, "hidden")}
        neurons2 = {1: Neuron(1, "hidden")}

        region1 = Region(0, RegionType.HIDDEN, neurons1, {})
        region2 = Region(1, RegionType.HIDDEN, neurons2, {})

        region1.freeze()

        self.assertTrue(region1.is_frozen())
        self.assertFalse(region2.is_frozen())

    def test_regions_independent_metrics(self):
        """Regions should have independent metrics."""
        neurons1 = {0: Neuron(0, "hidden"), 1: Neuron(1, "hidden")}
        neurons2 = {2: Neuron(2, "hidden"), 3: Neuron(3, "hidden")}

        region1 = Region(0, RegionType.HIDDEN, neurons1, {})
        region2 = Region(1, RegionType.HIDDEN, neurons2, {})

        region1.update_metrics({0, 1}, timestamp=0)
        region2.update_metrics({2}, timestamp=0)

        metrics1 = region1.get_metrics()
        metrics2 = region2.get_metrics()

        # Different firing should produce different metrics
        self.assertGreater(metrics1.active_neurons, metrics2.active_neurons)


class TestRegionTypes(unittest.TestCase):
    """Test different region types."""

    def test_sensory_region(self):
        """Should create sensory region."""
        neurons = {0: Neuron(0, "input")}
        region = Region(0, RegionType.SENSORY, neurons, {})

        self.assertEqual(region.region_type, RegionType.SENSORY)

    def test_motor_region(self):
        """Should create motor region."""
        neurons = {0: Neuron(0, "output")}
        region = Region(0, RegionType.MOTOR, neurons, {})

        self.assertEqual(region.region_type, RegionType.MOTOR)

    def test_memory_region(self):
        """Should create memory region."""
        neurons = {0: Neuron(0, "hidden")}
        region = Region(0, RegionType.MEMORY, neurons, {})

        self.assertEqual(region.region_type, RegionType.MEMORY)


if __name__ == "__main__":
    unittest.main()
