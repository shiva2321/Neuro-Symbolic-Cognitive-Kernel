"""
Tests for Phase 3/4: Continual Learning and Meta-Learning
Validates EWC, PackNet, MAML, and Reptile implementations.
"""
import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from continual_learning import ContinualLearner, PackNetManager
from meta_learning import MAMLLearner, ReptileLearner


class SimpleNet(nn.Module):
    """Simple net for testing continual/meta-learning."""

    def __init__(self, in_dim=8, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestContinualLearner(unittest.TestCase):
    """Test Elastic Weight Consolidation (EWC)."""

    def test_compute_importance(self):
        model = SimpleNet()
        learner = ContinualLearner(model)
        data = torch.randn(20, 8)
        labels = torch.randint(0, 4, (20,))
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)
        learner.compute_weight_importance("task_a", loader)
        self.assertIn("task_a", learner.weight_importance)
        self.assertGreater(len(learner.weight_importance["task_a"]), 0)

    def test_ewc_loss_zero_before_consolidation(self):
        model = SimpleNet()
        learner = ContinualLearner(model)
        loss = learner.ewc_loss()
        self.assertEqual(loss.item(), 0.0)

    def test_ewc_loss_nonzero_after_change(self):
        model = SimpleNet()
        learner = ContinualLearner(model)
        data = torch.randn(20, 8)
        labels = torch.randint(0, 4, (20,))
        loader = DataLoader(TensorDataset(data, labels), batch_size=4)
        learner.compute_weight_importance("task_a", loader)

        # Modify weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        loss = learner.ewc_loss()
        self.assertGreater(loss.item(), 0.0)


class TestPackNetManager(unittest.TestCase):
    """Test PackNet pruning and packing."""

    def test_prune_and_allocate(self):
        model = SimpleNet()
        manager = PackNetManager(model)
        manager.prune_and_allocate(task_id="task_a", prune_percentage=0.5)
        stats = manager.get_capacity_stats()
        self.assertIn("task_a", stats)

    def test_multiple_tasks(self):
        model = SimpleNet()
        manager = PackNetManager(model)
        manager.prune_and_allocate(task_id="task_a", prune_percentage=0.3)
        manager.prune_and_allocate(task_id="task_b", prune_percentage=0.3)
        stats = manager.get_capacity_stats()
        self.assertIn("task_a", stats)
        self.assertIn("task_b", stats)


class TestMAMLLearner(unittest.TestCase):
    """Test MAML meta-learning."""

    def test_meta_step(self):
        model = SimpleNet()
        learner = MAMLLearner(model, inner_lr=0.01, meta_lr=0.001)

        # Create task batch as list of dicts (matching existing API)
        tasks = []
        for _ in range(3):
            tasks.append({
                "support_input": torch.randn(5, 8),
                "support_target": torch.randint(0, 4, (5,)),
                "query_input": torch.randn(5, 8),
                "query_target": torch.randint(0, 4, (5,)),
            })

        loss = learner.meta_step(tasks)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_inner_loop_update(self):
        model = SimpleNet()
        learner = MAMLLearner(model, inner_lr=0.01, meta_lr=0.001)

        support_x = torch.randn(5, 8)
        support_y = torch.randint(0, 4, (5,))

        adapted = learner._inner_loop_update(support_x, support_y, steps=3)
        self.assertIsInstance(adapted, nn.Module)


class TestReptileLearner(unittest.TestCase):
    """Test Reptile meta-learning."""

    def test_meta_step(self):
        model = SimpleNet()
        learner = ReptileLearner(model, meta_lr=0.1)

        task_inputs = torch.randn(10, 8)
        task_targets = torch.randint(0, 4, (10,))

        # Reptile.meta_step takes direct inputs
        learner.meta_step(task_inputs, task_targets, inner_steps=5, inner_lr=0.01)

    def test_adapt(self):
        model = SimpleNet()
        learner = ReptileLearner(model, meta_lr=0.1)

        support_x = torch.randn(5, 8)
        support_y = torch.randint(0, 4, (5,))
        learner.adapt(support_x, support_y, steps=3)


if __name__ == "__main__":
    unittest.main()
