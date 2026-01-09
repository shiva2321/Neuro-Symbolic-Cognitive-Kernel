"""
Comprehensive Unit Tests for Spiking Neural Network Components
Tests Phase 2: Neuromorphic Core
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from ncgn.spiking_neurons import (
    LIFNeuron, IzhikevichNeuron, SpikingLayer,
    PoissonEncoder, RateEncoder, TemporalEncoder,
    SpikingGraphConvolution
)


class TestLIFNeuron(unittest.TestCase):
    """Test Leaky Integrate-and-Fire neuron"""

    def setUp(self):
        """Set up test fixtures"""
        self.neuron = LIFNeuron(
            tau=10.0,
            v_threshold=1.0,
            v_reset=0.0,
            v_rest=0.0
        )

    def test_initialization(self):
        """Test LIF neuron initialization"""
        self.assertEqual(self.neuron.v_threshold, 1.0)
        self.assertEqual(self.neuron.v_reset, 0.0)
        self.assertEqual(self.neuron.v_rest, 0.0)
        self.assertIsNone(self.neuron.membrane_potential)
        print("✓ LIF initialization test passed")

    def test_state_reset(self):
        """Test neuron state reset"""
        self.neuron.reset_state(batch_size=2, num_neurons=5, device='cpu')

        self.assertIsNotNone(self.neuron.membrane_potential)
        self.assertEqual(self.neuron.membrane_potential.shape, (2, 5))
        self.assertTrue(torch.allclose(
            self.neuron.membrane_potential,
            torch.zeros(2, 5)
        ))
        print("✓ LIF state reset test passed")

    def test_forward_no_spike(self):
        """Test forward pass with input below threshold"""
        self.neuron.reset_state(1, 3, 'cpu')

        # Input below threshold
        input_current = torch.tensor([[0.1, 0.2, 0.3]])
        spikes, v = self.neuron(input_current)

        # No spikes should occur
        self.assertEqual(spikes.sum().item(), 0)
        self.assertTrue(torch.all(spikes == 0))
        print("✓ LIF no spike test passed")

    def test_forward_with_spike(self):
        """Test forward pass with spike generation"""
        self.neuron.reset_state(1, 3, 'cpu')

        # Large input to cause spike
        input_current = torch.tensor([[5.0, 5.0, 5.0]])
        spikes, v = self.neuron(input_current)

        # Should produce spikes
        self.assertGreater(spikes.sum().item(), 0)
        print("✓ LIF spike generation test passed")

    def test_spike_causes_reset(self):
        """Test that spiking resets membrane potential"""
        self.neuron.reset_state(1, 1, 'cpu')

        # Cause spike
        input_current = torch.tensor([[5.0]])
        spikes, v = self.neuron(input_current)

        if spikes[0, 0] == 1:
            # Membrane potential should be reset
            self.assertAlmostEqual(v[0, 0].item(), self.neuron.v_reset, places=5)
        print("✓ LIF reset after spike test passed")

    def test_refractory_period(self):
        """Test refractory period behavior"""
        neuron = LIFNeuron(refractory_period=5)
        neuron.reset_state(1, 1, 'cpu')

        # Cause spike
        input_current = torch.tensor([[5.0]])
        spikes1, v1 = neuron(input_current)

        # During refractory period, should not spike again
        spikes2, v2 = neuron(input_current)

        if spikes1.sum() > 0:
            self.assertEqual(spikes2.sum().item(), 0)
        print("✓ LIF refractory period test passed")

    def test_batch_processing(self):
        """Test processing multiple samples"""
        self.neuron.reset_state(4, 10, 'cpu')

        input_current = torch.randn(4, 10) * 2
        spikes, v = self.neuron(input_current)

        self.assertEqual(spikes.shape, (4, 10))
        self.assertEqual(v.shape, (4, 10))
        print("✓ LIF batch processing test passed")


class TestIzhikevichNeuron(unittest.TestCase):
    """Test Izhikevich neuron model"""

    def setUp(self):
        """Set up test fixtures"""
        self.neuron = IzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8.0)

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.neuron.a, 0.02)
        self.assertEqual(self.neuron.b, 0.2)
        self.assertEqual(self.neuron.c, -65.0)
        self.assertEqual(self.neuron.d, 8.0)
        print("✓ Izhikevich initialization test passed")

    def test_state_reset(self):
        """Test state reset"""
        self.neuron.reset_state(2, 5, 'cpu')

        self.assertIsNotNone(self.neuron.v)
        self.assertIsNotNone(self.neuron.u)
        self.assertEqual(self.neuron.v.shape, (2, 5))
        self.assertEqual(self.neuron.u.shape, (2, 5))
        print("✓ Izhikevich state reset test passed")

    def test_forward_pass(self):
        """Test forward pass"""
        self.neuron.reset_state(1, 3, 'cpu')

        input_current = torch.tensor([[10.0, 20.0, 30.0]])
        spikes, v, u = self.neuron(input_current)

        self.assertEqual(spikes.shape, (1, 3))
        self.assertEqual(v.shape, (1, 3))
        self.assertEqual(u.shape, (1, 3))
        print("✓ Izhikevich forward pass test passed")

    def test_different_neuron_types(self):
        """Test different neuron type configurations"""
        # Regular spiking (RS)
        rs_neuron = IzhikevichNeuron(a=0.02, b=0.2, c=-65, d=8)

        # Fast spiking (FS)
        fs_neuron = IzhikevichNeuron(a=0.1, b=0.2, c=-65, d=2)

        # Both should work
        rs_neuron.reset_state(1, 1, 'cpu')
        fs_neuron.reset_state(1, 1, 'cpu')

        input_current = torch.tensor([[50.0]])
        rs_spikes, _, _ = rs_neuron(input_current)
        fs_spikes, _, _ = fs_neuron(input_current)

        # Both should produce valid outputs
        self.assertEqual(rs_spikes.shape, (1, 1))
        self.assertEqual(fs_spikes.shape, (1, 1))
        print("✓ Izhikevich neuron types test passed")


class TestPoissonEncoder(unittest.TestCase):
    """Test Poisson spike encoder"""

    def test_encode_rates(self):
        """Test encoding firing rates"""
        rates = torch.tensor([0.5, 0.8, 0.2])
        time_steps = 100

        spikes = PoissonEncoder.encode(rates, time_steps=time_steps)

        self.assertEqual(spikes.shape, (time_steps, 3))
        self.assertTrue(torch.all((spikes == 0) | (spikes == 1)))
        print("✓ Poisson encoding test passed")

    def test_encode_batch(self):
        """Test batch encoding"""
        rates = torch.rand(4, 10)
        time_steps = 50

        spikes = PoissonEncoder.encode(rates, time_steps=time_steps)

        self.assertEqual(spikes.shape, (time_steps, 4, 10))
        print("✓ Poisson batch encoding test passed")

    def test_rate_correspondence(self):
        """Test that spike rate corresponds to input rate"""
        # High rate should produce more spikes
        high_rate = torch.tensor([0.9])
        low_rate = torch.tensor([0.1])
        time_steps = 1000

        high_spikes = PoissonEncoder.encode(high_rate, time_steps)
        low_spikes = PoissonEncoder.encode(low_rate, time_steps)

        high_count = high_spikes.sum().item()
        low_count = low_spikes.sum().item()

        self.assertGreater(high_count, low_count)
        print("✓ Poisson rate correspondence test passed")


class TestRateEncoder(unittest.TestCase):
    """Test rate-based encoder"""

    def test_encode_features(self):
        """Test encoding feature vectors"""
        features = torch.tensor([[0.5, 0.8, 0.2]])
        time_steps = 50

        spikes = RateEncoder.encode(features, time_steps=time_steps)

        self.assertEqual(spikes.shape, (time_steps, 1, 3))
        print("✓ Rate encoding test passed")

    def test_normalization(self):
        """Test feature normalization"""
        features = torch.tensor([[1.0, 5.0, 10.0]])
        spikes = RateEncoder.encode(features, time_steps=100, normalize=True)

        # Normalized features should produce valid spikes
        self.assertTrue(torch.all((spikes == 0) | (spikes == 1)))
        print("✓ Rate normalization test passed")


class TestTemporalEncoder(unittest.TestCase):
    """Test temporal/latency encoder"""

    def test_encode_latency(self):
        """Test time-to-first-spike encoding"""
        features = torch.tensor([[0.1, 0.5, 0.9]])
        max_time = 50

        spikes = TemporalEncoder.encode(features, max_time=max_time)

        self.assertEqual(spikes.shape[1:], (1, 3))
        self.assertTrue(spikes.shape[0] <= max_time)
        print("✓ Temporal encoding test passed")

    def test_latency_ordering(self):
        """Test that higher values spike earlier"""
        features = torch.tensor([[0.9, 0.1]])  # High, then low
        spikes = TemporalEncoder.encode(features, max_time=100)

        # Find first spike times
        spike_times = []
        for i in range(features.shape[1]):
            spike_time = torch.where(spikes[:, 0, i] == 1)[0]
            if len(spike_time) > 0:
                spike_times.append(spike_time[0].item())

        if len(spike_times) == 2:
            # Higher value should spike first (earlier)
            self.assertLess(spike_times[0], spike_times[1])
        print("✓ Temporal latency ordering test passed")


class TestSpikingLayer(unittest.TestCase):
    """Test spiking neural network layer"""

    def setUp(self):
        """Set up test fixtures"""
        self.layer = SpikingLayer(
            in_features=10,
            out_features=5,
            neuron_model='lif'
        )

    def test_initialization(self):
        """Test layer initialization"""
        self.assertEqual(self.layer.in_features, 10)
        self.assertEqual(self.layer.out_features, 5)
        self.assertIsNotNone(self.layer.weight)
        print("✓ SpikingLayer initialization test passed")

    def test_forward_single_timestep(self):
        """Test forward pass for single timestep"""
        spikes_in = torch.randint(0, 2, (2, 10)).float()
        spikes_out = self.layer(spikes_in)

        self.assertEqual(spikes_out.shape, (2, 5))
        self.assertTrue(torch.all((spikes_out == 0) | (spikes_out == 1)))
        print("✓ SpikingLayer single timestep test passed")

    def test_forward_temporal(self):
        """Test forward pass for temporal sequence"""
        time_steps = 20
        spikes_in = torch.randint(0, 2, (time_steps, 2, 10)).float()

        outputs = []
        for t in range(time_steps):
            out = self.layer(spikes_in[t])
            outputs.append(out)

        outputs = torch.stack(outputs)
        self.assertEqual(outputs.shape, (time_steps, 2, 5))
        print("✓ SpikingLayer temporal test passed")

    def test_reset_state(self):
        """Test state reset"""
        spikes_in = torch.randint(0, 2, (2, 10)).float()
        self.layer(spikes_in)

        # Reset
        self.layer.reset_state()

        # Should work after reset
        spikes_out = self.layer(spikes_in)
        self.assertEqual(spikes_out.shape, (2, 5))
        print("✓ SpikingLayer reset test passed")

    def test_different_neuron_models(self):
        """Test different neuron models"""
        lif_layer = SpikingLayer(10, 5, neuron_model='lif')
        izh_layer = SpikingLayer(10, 5, neuron_model='izhikevich')

        spikes_in = torch.randint(0, 2, (2, 10)).float()

        lif_out = lif_layer(spikes_in)
        izh_out = izh_layer(spikes_in)

        self.assertEqual(lif_out.shape, (2, 5))
        self.assertEqual(izh_out.shape, (2, 5))
        print("✓ SpikingLayer neuron models test passed")


class TestSpikingGraphConvolution(unittest.TestCase):
    """Test spiking graph convolution"""

    def setUp(self):
        """Set up test fixtures"""
        self.conv = SpikingGraphConvolution(
            in_features=8,
            out_features=4,
            neuron_model='lif'
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertIsNotNone(self.conv)
        print("✓ SpikingGraphConv initialization test passed")

    def test_forward_with_adjacency(self):
        """Test forward pass with adjacency matrix"""
        batch_size = 2
        num_nodes = 5

        spikes_in = torch.randint(0, 2, (batch_size, num_nodes, 8)).float()
        adjacency = torch.rand(num_nodes, num_nodes)
        adjacency = (adjacency + adjacency.t()) / 2  # Symmetric

        spikes_out = self.conv(spikes_in, adjacency)

        self.assertEqual(spikes_out.shape, (batch_size, num_nodes, 4))
        print("✓ SpikingGraphConv forward test passed")


class TestSpikingNetworkIntegration(unittest.TestCase):
    """Integration tests for spiking networks"""

    def test_encoder_to_layer_pipeline(self):
        """Test complete pipeline from encoding to processing"""
        # Input features
        features = torch.rand(2, 10)

        # Encode
        spikes_in = PoissonEncoder.encode(features, time_steps=50)

        # Process through layer
        layer = SpikingLayer(10, 5, neuron_model='lif')

        outputs = []
        for t in range(50):
            out = layer(spikes_in[t])
            outputs.append(out)

        outputs = torch.stack(outputs)
        self.assertEqual(outputs.shape, (50, 2, 5))
        print("✓ Encoder-to-layer pipeline test passed")

    def test_multi_layer_network(self):
        """Test multi-layer spiking network"""
        layer1 = SpikingLayer(10, 8, neuron_model='lif')
        layer2 = SpikingLayer(8, 5, neuron_model='lif')

        spikes_in = torch.randint(0, 2, (2, 10)).float()

        hidden = layer1(spikes_in)
        output = layer2(hidden)

        self.assertEqual(output.shape, (2, 5))
        print("✓ Multi-layer network test passed")

    def test_energy_efficiency(self):
        """Test that spiking networks are sparse (energy efficient)"""
        layer = SpikingLayer(100, 50, neuron_model='lif')

        # Low input rate
        features = torch.rand(1, 100) * 0.3
        spikes_in = PoissonEncoder.encode(features, time_steps=100)

        spike_counts = []
        for t in range(100):
            out = layer(spikes_in[t])
            spike_counts.append(out.sum().item())

        # Should be sparse (< 50% activation)
        avg_spikes = np.mean(spike_counts)
        self.assertLess(avg_spikes, 25)  # Less than 50% of 50 neurons
        print("✓ Energy efficiency test passed")


def run_spiking_neuron_tests():
    """Run all spiking neuron tests"""
    print("\n" + "=" * 80)
    print("SPIKING NEURAL NETWORK TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLIFNeuron))
    suite.addTests(loader.loadTestsFromTestCase(TestIzhikevichNeuron))
    suite.addTests(loader.loadTestsFromTestCase(TestPoissonEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestRateEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestSpikingLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestSpikingGraphConvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestSpikingNetworkIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_spiking_neuron_tests()
    exit(0 if success else 1)

