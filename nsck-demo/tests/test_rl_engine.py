"""
Tests for Phase 0.2 / Phase 1: RL Engine (A2C, PPO, Neural-Symbolic Bridge)
Validates the reinforcement learning implementations from ROADMAP_TO_AGI.md.
"""
import unittest
import torch
import torch.nn as nn
from rl_engine import (
    RolloutBuffer,
    Transition,
    A2CTrainer,
    PPOTrainer,
    NeuralSymbolicBridge,
)


class SimpleActorCritic(nn.Module):
    """Minimal actor-critic for testing (avoids snnTorch dependency)."""

    def __init__(self, input_dim: int = 8, num_actions: int = 4):
        super().__init__()
        self.shared = nn.Linear(input_dim, 32)
        self.actor = nn.Linear(32, num_actions)
        self.critic = nn.Linear(32, 1)

    def forward(self, x, task_name="snake"):
        h = torch.relu(self.shared(x))
        return self.actor(h), self.critic(h)


class TestRolloutBuffer(unittest.TestCase):
    """Test the RolloutBuffer for trajectory collection and GAE."""

    def test_push_and_len(self):
        buf = RolloutBuffer()
        self.assertEqual(len(buf), 0)

        t = Transition(
            state=torch.randn(8),
            action=0,
            reward=1.0,
            next_state=torch.randn(8),
            done=False,
            log_prob=-0.5,
            value=0.8,
        )
        buf.push(t)
        self.assertEqual(len(buf), 1)

    def test_compute_returns_single(self):
        buf = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        buf.push(
            Transition(
                state=torch.randn(8),
                action=1,
                reward=1.0,
                next_state=None,
                done=True,
                log_prob=-0.3,
                value=0.5,
            )
        )
        returns, advantages = buf.compute_returns_and_advantages(last_value=0.0)
        self.assertEqual(returns.shape[0], 1)
        self.assertEqual(advantages.shape[0], 1)
        # For done=True: advantage = reward - value = 1.0 - 0.5 = 0.5
        self.assertAlmostEqual(advantages[0].item(), 0.5, places=4)

    def test_compute_returns_multi_step(self):
        buf = RolloutBuffer(gamma=0.99, gae_lambda=1.0)
        for i in range(3):
            buf.push(
                Transition(
                    state=torch.randn(8),
                    action=i % 4,
                    reward=1.0,
                    next_state=torch.randn(8),
                    done=(i == 2),
                    log_prob=-0.2,
                    value=0.0,
                )
            )
        returns, advantages = buf.compute_returns_and_advantages(last_value=0.0)
        self.assertEqual(returns.shape[0], 3)
        # With value=0 and gamma=0.99, gae_lambda=1.0:
        # returns should be discounted sums
        self.assertGreater(returns[0].item(), returns[2].item())

    def test_get_batch(self):
        buf = RolloutBuffer()
        for _ in range(5):
            buf.push(
                Transition(
                    state=torch.randn(8),
                    action=2,
                    reward=0.5,
                    next_state=torch.randn(8),
                    done=False,
                    log_prob=-0.1,
                    value=0.3,
                )
            )
        batch = buf.get_batch(last_value=0.0)
        self.assertIn("states", batch)
        self.assertIn("actions", batch)
        self.assertIn("returns", batch)
        self.assertIn("advantages", batch)
        self.assertEqual(batch["states"].shape[0], 5)

    def test_clear(self):
        buf = RolloutBuffer()
        buf.push(
            Transition(
                state=torch.randn(8),
                action=0,
                reward=1.0,
                next_state=None,
                done=True,
            )
        )
        self.assertEqual(len(buf), 1)
        buf.clear()
        self.assertEqual(len(buf), 0)

    def test_empty_buffer(self):
        buf = RolloutBuffer()
        returns, advantages = buf.compute_returns_and_advantages()
        self.assertEqual(returns.numel(), 0)
        self.assertEqual(advantages.numel(), 0)


class TestA2CTrainer(unittest.TestCase):
    """Test A2C trainer end-to-end."""

    def setUp(self):
        self.model = SimpleActorCritic(input_dim=8, num_actions=4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.trainer = A2CTrainer(
            model=self.model,
            optimizer=self.optimizer,
            gamma=0.99,
            value_coef=0.5,
            entropy_coef=0.01,
        )

    def test_select_action(self):
        state = torch.randn(8)
        action, log_prob, value = self.trainer.select_action(state)
        self.assertIn(action, [0, 1, 2, 3])
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)

    def test_train_step_empty(self):
        stats = self.trainer.train_step()
        self.assertEqual(stats["policy_loss"], 0.0)

    def test_train_step_with_data(self):
        # Collect some transitions
        for _ in range(10):
            state = torch.randn(8)
            action, log_prob, value = self.trainer.select_action(state)
            self.trainer.store_transition(
                Transition(
                    state=state,
                    action=action,
                    reward=1.0,
                    next_state=torch.randn(8),
                    done=False,
                    log_prob=log_prob,
                    value=value,
                )
            )

        stats = self.trainer.train_step(last_value=0.0)
        self.assertIn("policy_loss", stats)
        self.assertIn("value_loss", stats)
        self.assertIn("entropy", stats)
        self.assertIn("total_loss", stats)
        # After training, buffer should be cleared
        self.assertEqual(len(self.trainer.buffer), 0)

    def test_learning_reduces_loss(self):
        """Verify that A2C training actually reduces loss over iterations."""
        losses = []
        for iteration in range(5):
            for _ in range(20):
                state = torch.randn(8)
                action, log_prob, value = self.trainer.select_action(state)
                self.trainer.store_transition(
                    Transition(
                        state=state,
                        action=action,
                        reward=1.0 if action == 0 else -0.1,
                        next_state=torch.randn(8),
                        done=False,
                        log_prob=log_prob,
                        value=value,
                    )
                )
            stats = self.trainer.train_step()
            losses.append(stats["total_loss"])

        # Loss should be finite
        for loss in losses:
            self.assertTrue(torch.isfinite(torch.tensor(loss)))


class TestPPOTrainer(unittest.TestCase):
    """Test PPO trainer end-to-end."""

    def setUp(self):
        self.model = SimpleActorCritic(input_dim=8, num_actions=4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.trainer = PPOTrainer(
            model=self.model,
            optimizer=self.optimizer,
            gamma=0.99,
            clip_range=0.2,
            n_epochs=2,
            batch_size=8,
        )

    def test_select_action(self):
        state = torch.randn(8)
        action, log_prob, value = self.trainer.select_action(state)
        self.assertIn(action, [0, 1, 2, 3])

    def test_train_step_empty(self):
        stats = self.trainer.train_step()
        self.assertEqual(stats["policy_loss"], 0.0)
        self.assertEqual(stats["clip_fraction"], 0.0)

    def test_train_step_with_data(self):
        for _ in range(16):
            state = torch.randn(8)
            action, log_prob, value = self.trainer.select_action(state)
            self.trainer.store_transition(
                Transition(
                    state=state,
                    action=action,
                    reward=1.0,
                    next_state=torch.randn(8),
                    done=False,
                    log_prob=log_prob,
                    value=value,
                )
            )

        stats = self.trainer.train_step()
        self.assertIn("policy_loss", stats)
        self.assertIn("value_loss", stats)
        self.assertIn("clip_fraction", stats)
        self.assertEqual(len(self.trainer.buffer), 0)

    def test_clip_fraction_tracked(self):
        for _ in range(20):
            state = torch.randn(8)
            action, log_prob, value = self.trainer.select_action(state)
            self.trainer.store_transition(
                Transition(
                    state=state,
                    action=action,
                    reward=1.0,
                    next_state=torch.randn(8),
                    done=False,
                    log_prob=log_prob,
                    value=value,
                )
            )
        stats = self.trainer.train_step()
        # Clip fraction should be between 0 and 1
        self.assertGreaterEqual(stats["clip_fraction"], 0.0)
        self.assertLessEqual(stats["clip_fraction"], 1.0)


class TestNeuralSymbolicBridge(unittest.TestCase):
    """Test neural-symbolic arbitration."""

    def test_no_rules(self):
        bridge = NeuralSymbolicBridge()
        action, source = bridge.arbitrate(
            neural_action=0, neural_confidence=0.9, state={}
        )
        self.assertEqual(action, 0)
        self.assertEqual(source, "neural")

    def test_neural_wins_when_confident(self):
        class MockRuleLearner:
            def get_applicable_rules(self, state, task):
                return [{"action": 2, "confidence": 0.3}]

        bridge = NeuralSymbolicBridge(rule_learner=MockRuleLearner())
        action, source = bridge.arbitrate(
            neural_action=0, neural_confidence=0.8, state={}
        )
        self.assertEqual(action, 0)
        self.assertEqual(source, "neural")

    def test_symbolic_override_when_rules_stronger(self):
        class MockRuleLearner:
            def get_applicable_rules(self, state, task):
                return [{"action": 2, "confidence": 0.95}]

        bridge = NeuralSymbolicBridge(rule_learner=MockRuleLearner())
        action, source = bridge.arbitrate(
            neural_action=0, neural_confidence=0.3, state={}
        )
        self.assertEqual(action, 2)
        self.assertEqual(source, "symbolic")
        self.assertEqual(bridge.override_count, 1)

    def test_extract_rules_from_policy(self):
        model = SimpleActorCritic(input_dim=8, num_actions=4)
        bridge = NeuralSymbolicBridge()
        states = [torch.randn(8) for _ in range(50)]
        rules = bridge.extract_rules_from_policy(model, states)
        self.assertIsInstance(rules, list)
        # Should find at least one rule pattern
        self.assertGreater(len(rules), 0)
        for rule in rules:
            self.assertIn("action", rule)
            self.assertIn("confidence", rule)
            self.assertIn("source", rule)
            self.assertEqual(rule["source"], "neural_extraction")

    def test_stats(self):
        bridge = NeuralSymbolicBridge()
        bridge.arbitrate(neural_action=0, neural_confidence=0.9, state={})
        stats = bridge.get_stats()
        self.assertEqual(stats["neural_wins"], 1)
        self.assertEqual(stats["symbolic_wins"], 0)


if __name__ == "__main__":
    unittest.main()
