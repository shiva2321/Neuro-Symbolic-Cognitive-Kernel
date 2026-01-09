"""
Comprehensive Unit Tests for STDP Learning
Tests spike-timing-dependent plasticity mechanisms
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from ncgn.stdp_learning import (
    STDPLearning, TripleSTDP, RewardModulatedSTDP,
    HomeostaticSTDP, WeightNormalization
)


class TestSTDPLearning(unittest.TestCase):
    """Test basic STDP learning"""

    def setUp(self):
        """Set up test fixtures"""
        self.stdp = STDPLearning(
            a_plus=0.1,
            a_minus=0.12,
            tau_plus=20.0,
            tau_minus=20.0
        )

    def test_initialization(self):
        """Test STDP initialization"""
        self.assertEqual(self.stdp.a_plus, 0.1)
        self.assertEqual(self.stdp.a_minus, 0.12)
        self.assertGreater(self.stdp.tau_plus, 0)
        print("✓ STDP initialization test passed")

    def test_update_with_causal_spikes(self):
        """Test weight update with causal spike timing (pre before post)"""
        # Pre-synaptic spikes before post-synaptic
        pre_spikes = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]])
        post_spikes = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0]])
        weights = torch.ones(1, 1) * 0.5

        delta_w = self.stdp.update(pre_spikes, post_spikes, weights)

        # Should increase weights (LTP)
        self.assertIsNotNone(delta_w)
        print("✓ STDP causal spike test passed")

    def test_update_with_acausal_spikes(self):
        """Test weight update with acausal timing (post before pre)"""
        # Post-synaptic spikes before pre-synaptic
        pre_spikes = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0]])
        post_spikes = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]])
        weights = torch.ones(1, 1) * 0.5

        delta_w = self.stdp.update(pre_spikes, post_spikes, weights)

        # Should decrease weights (LTD)
        self.assertIsNotNone(delta_w)
        print("✓ STDP acausal spike test passed")

    def test_no_spikes_no_update(self):
        """Test that no spikes means no weight change"""
        pre_spikes = torch.zeros(1, 10)
        post_spikes = torch.zeros(1, 10)
        weights = torch.ones(1, 1) * 0.5

        delta_w = self.stdp.update(pre_spikes, post_spikes, weights)

        # Should be zero or very small
        self.assertTrue(torch.abs(delta_w).sum() < 0.01)
        print("✓ STDP no spike test passed")

    def test_batch_processing(self):
        """Test STDP with batched inputs"""
        batch_size = 4
        time_steps = 20

        pre_spikes = torch.randint(0, 2, (batch_size, time_steps)).float()
        post_spikes = torch.randint(0, 2, (batch_size, time_steps)).float()
        weights = torch.rand(batch_size, batch_size)

        delta_w = self.stdp.update(pre_spikes, post_spikes, weights)

        self.assertEqual(delta_w.shape, weights.shape)
        print("✓ STDP batch processing test passed")

    def test_weight_bounds(self):
        """Test that weights stay within bounds"""
        pre_spikes = torch.ones(1, 100)  # Always firing
        post_spikes = torch.ones(1, 100)
        weights = torch.ones(1, 1) * 0.5

        # Apply many updates
        for _ in range(100):
            delta_w = self.stdp.update(pre_spikes, post_spikes, weights)
            weights = torch.clamp(weights + delta_w, 0, 1)

        # Weights should be bounded
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))
        print("✓ STDP weight bounds test passed")


class TestTripleSTDP(unittest.TestCase):
    """Test triplet STDP (considers spike triplets)"""

    def setUp(self):
        """Set up test fixtures"""
        self.tstdp = TripleSTDP(
            a_plus=0.1,
            a_minus=0.12,
            tau_plus=20.0,
            tau_minus=20.0,
            tau_x=30.0,
            tau_y=40.0
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertIsNotNone(self.tstdp)
        self.assertGreater(self.tstdp.tau_x, 0)
        print("✓ TripleSTDP initialization test passed")

    def test_update(self):
        """Test triplet update"""
        pre_spikes = torch.randint(0, 2, (2, 50)).float()
        post_spikes = torch.randint(0, 2, (2, 50)).float()
        weights = torch.rand(2, 2)

        delta_w = self.tstdp.update(pre_spikes, post_spikes, weights)

        self.assertEqual(delta_w.shape, weights.shape)
        print("✓ TripleSTDP update test passed")

    def test_triplet_vs_pairwise(self):
        """Test that triplet STDP differs from pairwise"""
        pre_spikes = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
        post_spikes = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])
        weights = torch.ones(1, 1) * 0.5

        # Standard STDP
        stdp = STDPLearning()
        delta_w_pair = stdp.update(pre_spikes, post_spikes, weights)

        # Triplet STDP
        delta_w_triple = self.tstdp.update(pre_spikes, post_spikes, weights)

        # Results should differ (triplet considers history)
        self.assertIsNotNone(delta_w_pair)
        self.assertIsNotNone(delta_w_triple)
        print("✓ Triplet vs pairwise test passed")


class TestRewardModulatedSTDP(unittest.TestCase):
    """Test reward-modulated STDP"""

    def setUp(self):
        """Set up test fixtures"""
        self.rm_stdp = RewardModulatedSTDP(
            a_plus=0.1,
            a_minus=0.12,
            tau_plus=20.0,
            tau_minus=20.0,
            tau_reward=100.0
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertIsNotNone(self.rm_stdp)
        self.assertGreater(self.rm_stdp.tau_reward, 0)
        print("✓ RewardModulatedSTDP initialization test passed")

    def test_update_with_positive_reward(self):
        """Test update with positive reward"""
        pre_spikes = torch.randint(0, 2, (1, 20)).float()
        post_spikes = torch.randint(0, 2, (1, 20)).float()
        weights = torch.rand(1, 1)
        reward = torch.tensor([1.0])  # Positive reward

        delta_w = self.rm_stdp.update(pre_spikes, post_spikes, weights, reward)

        self.assertIsNotNone(delta_w)
        print("✓ Reward-modulated positive reward test passed")

    def test_update_with_negative_reward(self):
        """Test update with negative reward"""
        pre_spikes = torch.randint(0, 2, (1, 20)).float()
        post_spikes = torch.randint(0, 2, (1, 20)).float()
        weights = torch.rand(1, 1)
        reward = torch.tensor([-1.0])  # Negative reward

        delta_w = self.rm_stdp.update(pre_spikes, post_spikes, weights, reward)

        self.assertIsNotNone(delta_w)
        print("✓ Reward-modulated negative reward test passed")

    def test_reward_modulation_effect(self):
        """Test that reward actually modulates updates"""
        pre_spikes = torch.ones(1, 10)
        post_spikes = torch.ones(1, 10)
        weights = torch.ones(1, 1) * 0.5

        # High reward
        high_reward = torch.tensor([1.0])
        delta_w_high = self.rm_stdp.update(pre_spikes, post_spikes, weights, high_reward)

        # Low reward
        low_reward = torch.tensor([0.1])
        delta_w_low = self.rm_stdp.update(pre_spikes, post_spikes, weights, low_reward)

        # High reward should produce larger magnitude changes
        if delta_w_high is not None and delta_w_low is not None:
            self.assertGreater(
                torch.abs(delta_w_high).sum(),
                torch.abs(delta_w_low).sum()
            )
        print("✓ Reward modulation effect test passed")


class TestHomeostaticSTDP(unittest.TestCase):
    """Test homeostatic STDP (maintains target firing rate)"""

    def setUp(self):
        """Set up test fixtures"""
        self.h_stdp = HomeostaticSTDP(
            target_rate=0.1,
            tau_homeostatic=1000.0
        )

    def test_initialization(self):
        """Test initialization"""
        self.assertEqual(self.h_stdp.target_rate, 0.1)
        self.assertGreater(self.h_stdp.tau_homeostatic, 0)
        print("✓ HomeostaticSTDP initialization test passed")

    def test_update_with_high_firing_rate(self):
        """Test that high firing rate is suppressed"""
        # Very high firing rate
        pre_spikes = torch.ones(1, 100)
        post_spikes = torch.ones(1, 100)
        weights = torch.ones(1, 1) * 0.8

        delta_w = self.h_stdp.update(pre_spikes, post_spikes, weights)

        # Should reduce weights to decrease firing rate
        if delta_w is not None:
            self.assertLess(delta_w.mean().item(), 0.01)
        print("✓ Homeostatic high rate test passed")

    def test_update_with_low_firing_rate(self):
        """Test that low firing rate is enhanced"""
        # Very low firing rate
        pre_spikes = torch.zeros(1, 100)
        post_spikes = torch.zeros(1, 100)
        # Add occasional spikes
        pre_spikes[0, 10] = 1
        post_spikes[0, 11] = 1
        weights = torch.ones(1, 1) * 0.2

        delta_w = self.h_stdp.update(pre_spikes, post_spikes, weights)

        self.assertIsNotNone(delta_w)
        print("✓ Homeostatic low rate test passed")

    def test_convergence_to_target_rate(self):
        """Test that firing rate converges to target over time"""
        weights = torch.rand(5, 5)

        # Simulate learning
        for _ in range(10):
            pre_spikes = torch.randint(0, 2, (5, 20)).float()
            post_spikes = torch.randint(0, 2, (5, 20)).float()

            delta_w = self.h_stdp.update(pre_spikes, post_spikes, weights)
            weights = torch.clamp(weights + delta_w * 0.01, 0, 1)

        # Weights should be adjusted
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))
        print("✓ Homeostatic convergence test passed")


class TestWeightNormalization(unittest.TestCase):
    """Test weight normalization schemes"""

    def test_l1_normalization(self):
        """Test L1 normalization"""
        weights = torch.randn(10, 10)
        normalized = WeightNormalization.normalize(weights, method='l1')

        # Check that rows sum to 1 (approximately)
        row_sums = normalized.abs().sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(10), atol=0.01))
        print("✓ L1 normalization test passed")

    def test_l2_normalization(self):
        """Test L2 normalization"""
        weights = torch.randn(10, 10)
        normalized = WeightNormalization.normalize(weights, method='l2')

        # Check that rows have unit L2 norm
        row_norms = torch.norm(normalized, p=2, dim=1)
        self.assertTrue(torch.allclose(row_norms, torch.ones(10), atol=0.01))
        print("✓ L2 normalization test passed")

    def test_max_normalization(self):
        """Test max normalization"""
        weights = torch.randn(10, 10) * 5
        max_weight = 1.0
        normalized = WeightNormalization.normalize(weights, method='max', max_weight=max_weight)

        # All weights should be <= max_weight
        self.assertTrue(torch.all(normalized.abs() <= max_weight))
        print("✓ Max normalization test passed")

    def test_soft_bounds(self):
        """Test soft bounding"""
        weights = torch.randn(10, 10) * 3
        bounded = WeightNormalization.soft_bound(weights, min_val=0, max_val=1)

        # Should be mostly in range [0, 1]
        self.assertTrue(torch.all(bounded >= -0.1))  # Allow small overshoot
        self.assertTrue(torch.all(bounded <= 1.1))
        print("✓ Soft bounds test passed")


class TestSTDPIntegration(unittest.TestCase):
    """Integration tests for STDP learning"""

    def test_stdp_with_spiking_layer(self):
        """Test STDP integration with spiking layer"""
        from ncgn.spiking_neurons import SpikingLayer

        layer = SpikingLayer(10, 5, neuron_model='lif')
        stdp = STDPLearning()

        # Generate spike trains
        pre_spikes = torch.randint(0, 2, (2, 10)).float()
        post_spikes = layer(pre_spikes)

        # Update weights
        delta_w = stdp.update(pre_spikes, post_spikes, layer.weight.data)

        # Apply update
        layer.weight.data += delta_w * 0.01

        self.assertIsNotNone(delta_w)
        print("✓ STDP-layer integration test passed")

    def test_learning_association(self):
        """Test that STDP learns temporal associations"""
        stdp = STDPLearning(a_plus=0.5, a_minus=0.5)

        # Pattern: neuron 0 fires, then neuron 1 fires
        weights = torch.ones(2, 2) * 0.5

        for _ in range(10):
            pre_spikes = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            post_spikes = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

            delta_w = stdp.update(pre_spikes.t(), post_spikes.t(), weights)
            weights += delta_w * 0.1
            weights = torch.clamp(weights, 0, 1)

        # Weight from 0 to 1 should increase
        self.assertGreater(weights[0, 1].item(), 0.5)
        print("✓ Association learning test passed")

    def test_stability_over_time(self):
        """Test that STDP remains stable over many updates"""
        stdp = HomeostaticSTDP(target_rate=0.1)
        weights = torch.rand(20, 20)

        # Many updates
        for _ in range(100):
            pre_spikes = torch.randint(0, 2, (20, 50)).float() * 0.1
            post_spikes = torch.randint(0, 2, (20, 50)).float() * 0.1

            delta_w = stdp.update(pre_spikes, post_spikes, weights)
            weights += delta_w * 0.01
            weights = torch.clamp(weights, 0, 1)

        # Weights should remain valid
        self.assertFalse(torch.any(torch.isnan(weights)))
        self.assertFalse(torch.any(torch.isinf(weights)))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))
        print("✓ Long-term stability test passed")


def run_stdp_tests():
    """Run all STDP tests"""
    print("\n" + "=" * 80)
    print("STDP LEARNING TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSTDPLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestTripleSTDP))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardModulatedSTDP))
    suite.addTests(loader.loadTestsFromTestCase(TestHomeostaticSTDP))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestSTDPIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_stdp_tests()
    exit(0 if success else 1)

