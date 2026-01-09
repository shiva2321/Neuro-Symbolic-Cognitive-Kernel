"""
Unit Tests for Agent 4: Analytic Learner
Tests RLS, dynamic thresholds, implicit differentiation, and continual learning.
"""

import unittest
import torch
import torch.nn as nn

# Handle DGL import error using a local stub
from tests import dgl_stub as _dgl_stub
import sys

sys.modules.setdefault("dgl", _dgl_stub)
import dgl

from agents.analytic_learner import (

    AnalyticLearner, AnalyticLearningConfig,
    RecursiveLeastSquares, DynamicThresholdNeuron, ImplicitDifferentiationLayer
)


class TestRecursiveLeastSquares(unittest.TestCase):
    """Test suite for Recursive Least Squares"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.output_dim = 3
        self.rls = RecursiveLeastSquares(self.input_dim, self.output_dim)

    def test_initialization(self):
        """Test RLS initialization"""
        self.assertEqual(self.rls.input_dim, self.input_dim)
        self.assertEqual(self.rls.output_dim, self.output_dim)
        self.assertEqual(self.rls.W.shape, (self.output_dim, self.input_dim))
        self.assertEqual(self.rls.P.shape, (self.input_dim, self.input_dim))
        print("✓ RLS initialization test passed")

    def test_single_update(self):
        """Test single RLS update"""
        x = torch.randn(1, self.input_dim)
        y = torch.randn(1, self.output_dim)

        initial_W = self.rls.W.clone()

        self.rls.update(x, y)

        # Weights should have changed
        self.assertFalse(torch.allclose(self.rls.W, initial_W))
        self.assertEqual(self.rls.num_updates, 1)
        print("✓ Single RLS update test passed")

    def test_batch_update(self):
        """Test batch RLS update"""
        batch_size = 5
        x = torch.randn(batch_size, self.input_dim)
        y = torch.randn(batch_size, self.output_dim)

        self.rls.update(x, y)

        self.assertEqual(self.rls.num_updates, batch_size)
        print("✓ Batch RLS update test passed")

    def test_prediction(self):
        """Test RLS prediction"""
        x = torch.randn(3, self.input_dim)

        predictions = self.rls.predict(x)

        self.assertEqual(predictions.shape, (3, self.output_dim))
        print("✓ RLS prediction test passed")

    def test_continual_learning(self):
        """Test RLS continual learning (no forgetting)"""
        # First batch
        x1 = torch.randn(10, self.input_dim)
        y1 = torch.randn(10, self.output_dim)
        self.rls.update(x1, y1)

        pred1_before = self.rls.predict(x1[:3])

        # Second batch
        x2 = torch.randn(10, self.input_dim)
        y2 = torch.randn(10, self.output_dim)
        self.rls.update(x2, y2)

        pred1_after = self.rls.predict(x1[:3])

        # Predictions should be somewhat similar (not catastrophically forgotten)
        diff = (pred1_before - pred1_after).abs().mean()
        self.assertLess(diff, 5.0)  # Reasonable threshold
        print("✓ RLS continual learning test passed")


class TestDynamicThresholdNeuron(unittest.TestCase):
    """Test suite for Dynamic Threshold Neurons"""

    def setUp(self):
        """Set up test fixtures"""
        self.num_neurons = 10
        self.layer = DynamicThresholdNeuron(self.num_neurons, initial_threshold=0.5)

    def test_initialization(self):
        """Test threshold neuron initialization"""
        self.assertEqual(self.layer.num_neurons, self.num_neurons)
        self.assertEqual(self.layer.threshold.shape, (self.num_neurons,))
        print("✓ Threshold neuron initialization test passed")

    def test_forward_pass(self):
        """Test forward pass with thresholding"""
        x = torch.randn(5, self.num_neurons)

        output, pre_activation = self.layer(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(pre_activation.shape, x.shape)

        # Output should be thresholded
        for i in range(self.num_neurons):
            expected = torch.where(x[:, i] > self.layer.threshold[i],
                                 x[:, i] - self.layer.threshold[i],
                                 torch.zeros_like(x[:, i]))
            self.assertTrue(torch.allclose(output[:, i], expected, atol=1e-6))

        print("✓ Threshold neuron forward pass test passed")

    def test_threshold_update(self):
        """Test threshold adaptation"""
        x = torch.randn(10, self.num_neurons) + 1.0  # Bias toward positive

        initial_thresholds = self.layer.threshold.clone()

        # Forward pass to accumulate statistics
        self.layer(x)

        # Update thresholds
        self.layer.update_thresholds(target_activation_rate=0.1)

        # Thresholds should have changed
        self.assertFalse(torch.allclose(self.layer.threshold, initial_thresholds))
        print("✓ Threshold update test passed")


class TestImplicitDifferentiationLayer(unittest.TestCase):
    """Test suite for Implicit Differentiation Layer"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 8
        self.output_dim = 4
        self.layer = ImplicitDifferentiationLayer(
            self.input_dim, self.output_dim,
            max_iterations=20, tolerance=1e-3
        )

    def test_initialization(self):
        """Test layer initialization"""
        self.assertEqual(self.layer.input_dim, self.input_dim)
        self.assertEqual(self.layer.output_dim, self.output_dim)
        self.assertIsNotNone(self.layer.W)
        self.assertIsNotNone(self.layer.U)
        print("✓ Implicit layer initialization test passed")

    def test_equilibrium_finding(self):
        """Test equilibrium state finding"""
        x = torch.randn(3, self.input_dim)

        z_star = self.layer.find_equilibrium(x)

        self.assertEqual(z_star.shape, (3, self.output_dim))
        self.assertFalse(torch.isnan(z_star).any())
        print("✓ Equilibrium finding test passed")

    def test_forward_pass(self):
        """Test forward pass"""
        x = torch.randn(3, self.input_dim)

        output = self.layer(x)

        self.assertEqual(output.shape, (3, self.output_dim))
        print("✓ Implicit layer forward pass test passed")

    def test_gradient_computation(self):
        """Test gradient computation"""
        x = torch.randn(3, self.input_dim)
        x.requires_grad = True

        output = self.layer(x)
        loss = output.sum()

        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        print("✓ Implicit layer gradient test passed")


class TestAnalyticLearner(unittest.TestCase):
    """Test suite for Analytic Learner"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = AnalyticLearningConfig(
            use_rls=True,
            use_dynamic_threshold=True,
            use_implicit_diff=False,  # Disabled for faster tests
            memory_efficient=True
        )
        self.learner = AnalyticLearner(self.config, device='cpu')

    def test_initialization(self):
        """Test learner initialization"""
        self.assertIsNotNone(self.learner)
        self.assertEqual(self.learner.samples_seen, 0)
        print("✓ Learner initialization test passed")

    def test_rls_estimator_creation(self):
        """Test RLS estimator creation"""
        self.learner.create_rls_estimator('test_layer', input_dim=10, output_dim=3)

        self.assertIn('test_layer', self.learner.rls_estimators)
        estimator = self.learner.rls_estimators['test_layer']
        self.assertEqual(estimator.input_dim, 10)
        self.assertEqual(estimator.output_dim, 3)
        print("✓ RLS estimator creation test passed")

    def test_train_step(self):
        """Test single training step"""
        # Create simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        features = torch.randn(3, 10)
        labels = torch.randint(0, 3, (3,))

        metrics = self.learner.train_step(graph, features, labels, 'test')

        self.assertIn('accuracy', metrics)
        self.assertIn('samples_seen', metrics)
        self.assertGreater(self.learner.samples_seen, 0)
        print("✓ Train step test passed")

    def test_dynamic_threshold_training(self):
        """Test training with dynamic thresholds"""
        # Create simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        features = torch.randn(3, 10)
        labels = torch.randint(0, 3, (3,))

        metrics = self.learner.train_with_dynamic_threshold(
            graph, features, labels, 'threshold_test'
        )

        self.assertIn('accuracy', metrics)
        self.assertIn('avg_threshold', metrics)
        print("✓ Dynamic threshold training test passed")

    def test_continual_learning(self):
        """Test continual learning capability"""
        # Create simple graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        graph = dgl.graph((src, dst))

        # First task
        features1 = torch.randn(3, 10)
        labels1 = torch.randint(0, 3, (3,))
        metrics1 = self.learner.train_step(graph, features1, labels1, 'continual')

        # Second task
        features2 = torch.randn(3, 10)
        labels2 = torch.randint(0, 3, (3,))
        metrics2 = self.learner.continual_learn(graph, features2, labels2, 'continual')

        self.assertGreater(len(self.learner.training_history), 0)
        print("✓ Continual learning test passed")

    def test_weight_retrieval(self):
        """Test weight retrieval"""
        self.learner.create_rls_estimator('weight_test', input_dim=10, output_dim=3)

        weights = self.learner.get_weights('weight_test')

        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (3, 10))
        print("✓ Weight retrieval test passed")

    def test_state_persistence(self):
        """Test saving and loading state"""
        import tempfile
        import os

        # Create estimator
        self.learner.create_rls_estimator('persist_test', input_dim=5, output_dim=2)

        # Save state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name

        try:
            self.learner.save_state(temp_path)

            # Create new learner and load
            new_learner = AnalyticLearner(self.config, device='cpu')
            new_learner.load_state(temp_path)

            self.assertIn('persist_test', new_learner.rls_estimators)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        print("✓ State persistence test passed")


def run_analytic_learner_tests():
    """Run all analytic learner tests"""
    print("\n" + "="*70)
    print("Running Analytic Learner Tests")
    print("="*70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestRecursiveLeastSquares))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicThresholdNeuron))
    suite.addTests(loader.loadTestsFromTestCase(TestImplicitDifferentiationLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticLearner))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_analytic_learner_tests()
    exit(0 if success else 1)