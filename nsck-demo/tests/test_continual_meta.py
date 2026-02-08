"""
Tests for Phase 3/4: Continual Learning and Meta-Learning
Validates EWC, PackNet, Progressive Networks, Memory Replay, MAML, and Reptile.
"""
import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from continual_learning import (
    ContinualLearner,
    PackNetManager,
    ProgressiveNetwork,
    ProgressiveColumn,
    MemoryReplayManager,
)
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


class TestProgressiveNetwork(unittest.TestCase):
    """Test Progressive Neural Networks for continual learning (Phase 3.2)."""

    def test_add_first_task(self):
        net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
        col = net.add_task("snake")
        self.assertIsInstance(col, ProgressiveColumn)
        self.assertEqual(len(net.columns), 1)

    def test_add_multiple_tasks(self):
        net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
        net.add_task("snake")
        net.add_task("pong")
        col_c = net.add_task("maze")
        self.assertEqual(len(net.columns), 3)
        # Only last column should be trainable
        for param in net.columns[0].parameters():
            self.assertFalse(param.requires_grad)
        for param in net.columns[1].parameters():
            self.assertFalse(param.requires_grad)
        # Latest column is trainable
        trainable_count = sum(
            1 for p in col_c.parameters() if p.requires_grad
        )
        self.assertGreater(trainable_count, 0)

    def test_forward_with_lateral(self):
        net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
        net.add_task("snake")
        col_b = net.add_task("pong")
        x = torch.randn(2, 8)
        out = col_b(x)
        self.assertEqual(out.shape, (2, 4))

    def test_get_column(self):
        net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
        net.add_task("snake")
        col = net.get_column("snake")
        self.assertIsNotNone(col)
        self.assertIsNone(net.get_column("nonexistent"))

    def test_stats(self):
        net = ProgressiveNetwork(input_dim=8, hidden_dim=16, output_dim=4)
        net.add_task("task_a")
        net.add_task("task_b")
        stats = net.get_stats()
        self.assertEqual(stats["num_tasks"], 2)
        self.assertIn("task_a", stats["task_names"])
        self.assertGreater(stats["total_params"], 0)


class TestMemoryReplayManager(unittest.TestCase):
    """Test Memory Replay for continual learning (Phase 3.3)."""

    def test_store_and_sample(self):
        mgr = MemoryReplayManager(capacity_per_task=100)
        inputs = torch.randn(10, 8)
        targets = torch.randint(0, 4, (10,))
        mgr.store("snake", inputs, targets)
        result = mgr.sample_mixed(batch_size=5)
        self.assertIsNotNone(result)
        sampled_inputs, sampled_targets = result
        self.assertLessEqual(sampled_inputs.shape[0], 10)

    def test_multi_task_replay(self):
        mgr = MemoryReplayManager(capacity_per_task=50)
        mgr.store("snake", torch.randn(20, 8), torch.randint(0, 4, (20,)))
        mgr.store("pong", torch.randn(20, 8), torch.randint(0, 4, (20,)))
        result = mgr.sample_mixed(batch_size=10)
        self.assertIsNotNone(result)

    def test_capacity_limit(self):
        mgr = MemoryReplayManager(capacity_per_task=5)
        mgr.store("task_a", torch.randn(20, 8), torch.randint(0, 4, (20,)))
        self.assertEqual(len(mgr.task_buffers["task_a"]), 5)

    def test_empty_sample(self):
        mgr = MemoryReplayManager()
        result = mgr.sample_mixed(batch_size=10)
        self.assertIsNone(result)

    def test_stats(self):
        mgr = MemoryReplayManager(capacity_per_task=100)
        mgr.store("snake", torch.randn(15, 8), torch.randint(0, 4, (15,)))
        mgr.store("pong", torch.randn(10, 8), torch.randint(0, 4, (10,)))
        stats = mgr.get_stats()
        self.assertEqual(stats["snake"], 15)
        self.assertEqual(stats["pong"], 10)


if __name__ == "__main__":
    unittest.main()
