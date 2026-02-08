"""
Tests for Phase 0.1: SNN Training Pipeline
Validates SNN training, Hebbian learning, and concept activation mapping.
"""
import unittest
import torch
import torch.nn as nn
from snn_training_pipeline import (
    HebbianLayer,
    ConceptActivationMapper,
    SNNTrainingPipeline,
    TrainingMetrics,
)


class SimpleEncoder(nn.Module):
    """Minimal encoder for testing (avoids snnTorch dependency)."""

    def __init__(self, input_dim: int = 100, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)

    def forward(self, x, modality_hint=None):
        return torch.relu(self.fc(x))


class SimpleModel(nn.Module):
    """Minimal actor-critic model with encoder for testing."""

    def __init__(self, input_dim: int = 100, num_actions: int = 4):
        super().__init__()
        self.encoder = SimpleEncoder(input_dim, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, task_name="snake"):
        h = self.encoder(x)
        return self.actor(h), self.critic(h)


class TestHebbianLayer(unittest.TestCase):
    """Test Hebbian learning layer."""

    def test_forward(self):
        layer = HebbianLayer(64, 32)
        x = torch.randn(4, 64)
        out = layer(x)
        self.assertEqual(out.shape, (4, 32))

    def test_hebbian_update(self):
        layer = HebbianLayer(64, 32, learning_rate=0.1)
        pre = torch.randn(4, 64)
        post = torch.randn(4, 32)

        weight_before = layer.weight.data.clone()
        layer.hebbian_update(pre, post)
        weight_after = layer.weight.data

        # Weights should change after Hebbian update
        diff = (weight_after - weight_before).abs().sum().item()
        self.assertGreater(diff, 0.0)
        self.assertEqual(layer.update_count, 1)

    def test_multiple_updates(self):
        layer = HebbianLayer(32, 16, learning_rate=0.01)
        for _ in range(10):
            pre = torch.randn(2, 32)
            post = torch.randn(2, 16)
            layer.hebbian_update(pre, post)
        self.assertEqual(layer.update_count, 10)

    def test_empty_batch(self):
        layer = HebbianLayer(32, 16)
        pre = torch.randn(0, 32)
        post = torch.randn(0, 16)
        weight_before = layer.weight.data.clone()
        layer.hebbian_update(pre, post)
        # No change for empty batch
        self.assertTrue(torch.equal(layer.weight.data, weight_before))


class TestConceptActivationMapper(unittest.TestCase):
    """Test concept activation mapping from SNN spikes to concepts."""

    def test_register_concept(self):
        mapper = ConceptActivationMapper(num_concepts=16, spike_dim=128)
        mapper.register_concept(0, "food_ahead")
        mapper.register_concept(1, "wall_ahead")
        self.assertEqual(mapper.concept_names[0], "food_ahead")
        self.assertEqual(mapper.concept_names[1], "wall_ahead")

    def test_activate(self):
        mapper = ConceptActivationMapper(num_concepts=16, spike_dim=128)
        spike_rates = torch.randn(1, 128)
        activations = mapper.activate(spike_rates, threshold=0.3)
        self.assertIsInstance(activations, dict)
        for cid, val in activations.items():
            self.assertGreater(val, 0.3)

    def test_get_top_concepts(self):
        mapper = ConceptActivationMapper(num_concepts=16, spike_dim=128)
        mapper.register_concept(0, "danger")
        mapper.register_concept(5, "opportunity")
        spike_rates = torch.randn(1, 128)
        top = mapper.get_top_concepts(spike_rates, k=3)
        self.assertLessEqual(len(top), 3)
        for name, val in top:
            self.assertIsInstance(name, str)
            self.assertIsInstance(val, float)

    def test_activation_history(self):
        mapper = ConceptActivationMapper(num_concepts=8, spike_dim=128)
        for _ in range(5):
            mapper.activate(torch.randn(1, 128))
        self.assertEqual(len(mapper.activation_history), 5)


class TestSNNTrainingPipeline(unittest.TestCase):
    """Test the complete SNN training pipeline."""

    def setUp(self):
        self.model = SimpleModel(input_dim=100, num_actions=4)
        self.pipeline = SNNTrainingPipeline(
            model=self.model,
            learning_rate=1e-3,
            hebbian_lr=0.01,
            concept_dim=16,
        )

    def test_train_epoch(self):
        from torch.utils.data import TensorDataset, DataLoader

        states = torch.randn(64, 100)
        labels = torch.randint(0, 4, (64,))
        dataset = TensorDataset(states, labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        metrics = self.pipeline.train_epoch(loader, task_name="snake")
        self.assertIsInstance(metrics, TrainingMetrics)
        self.assertGreater(metrics.loss, 0.0)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertGreater(metrics.hebbian_updates, 0)

    def test_train_without_hebbian(self):
        from torch.utils.data import TensorDataset, DataLoader

        states = torch.randn(32, 100)
        labels = torch.randint(0, 4, (32,))
        dataset = TensorDataset(states, labels)
        loader = DataLoader(dataset, batch_size=16)

        metrics = self.pipeline.train_epoch(
            loader, task_name="snake", use_hebbian=False
        )
        self.assertEqual(metrics.hebbian_updates, 0)

    def test_bind_concepts(self):
        states = torch.randn(1, 100)
        labels = {0: "food_ahead", 1: "wall_near", 2: "safe_path"}
        activations = self.pipeline.bind_concepts_to_vsa(states, labels)
        self.assertIsInstance(activations, dict)

    def test_training_summary(self):
        # Empty summary
        summary = self.pipeline.get_training_summary()
        self.assertEqual(summary["epochs"], 0)

        # After training
        from torch.utils.data import TensorDataset, DataLoader

        states = torch.randn(32, 100)
        labels = torch.randint(0, 4, (32,))
        loader = DataLoader(TensorDataset(states, labels), batch_size=16)

        self.pipeline.train_epoch(loader)
        summary = self.pipeline.get_training_summary()
        self.assertEqual(summary["epochs"], 1)
        self.assertGreater(summary["final_loss"], 0.0)

    def test_multi_epoch_training(self):
        from torch.utils.data import TensorDataset, DataLoader

        states = torch.randn(48, 100)
        labels = torch.randint(0, 4, (48,))
        loader = DataLoader(TensorDataset(states, labels), batch_size=16)

        for epoch in range(3):
            metrics = self.pipeline.train_epoch(loader)
            self.assertIsInstance(metrics, TrainingMetrics)

        summary = self.pipeline.get_training_summary()
        self.assertEqual(summary["epochs"], 3)
        self.assertEqual(len(summary["loss_trend"]), 3)


if __name__ == "__main__":
    unittest.main()
