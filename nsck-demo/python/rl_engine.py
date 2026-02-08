"""
NSCK Reinforcement Learning Module
Phase 0.2 / Phase 1: Neural Learning Engine (ROADMAP_TO_AGI.md)

Implements:
  - A2C (Advantage Actor-Critic): Synchronous actor-critic with entropy bonus
  - PPO (Proximal Policy Optimization): Clipped surrogate objective

Design constraints (per AGENT_INSTRUCTIONS.md):
  - Efficiency first: O(n) operations, keep dims ≤ 128
  - Works on CPU-only consumer hardware (4GB+ RAM)
  - Integrates with existing TaskAwareSNN and symbolic rules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import random


@dataclass
class Transition:
    """Single transition for RL training."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: Optional[torch.Tensor]
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


class RolloutBuffer:
    """
    Collects trajectories for on-policy RL (A2C/PPO).

    Stores transitions per episode and computes GAE advantages
    at the end of each rollout.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.transitions: List[Transition] = []

    def push(self, transition: Transition):
        """Add transition to current rollout."""
        self.transitions.append(transition)

    def compute_returns_and_advantages(
        self, last_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute discounted returns and GAE advantages.

        Args:
            last_value: Bootstrap value for incomplete episodes

        Returns:
            (returns, advantages) tensors
        """
        n = len(self.transitions)
        if n == 0:
            return torch.tensor([]), torch.tensor([])

        returns = torch.zeros(n)
        advantages = torch.zeros(n)
        values = torch.tensor([t.value for t in self.transitions])
        rewards = torch.tensor([t.reward for t in self.transitions])
        dones = torch.tensor([float(t.done) for t in self.transitions])

        # GAE (Generalized Advantage Estimation)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t].item()

        returns = advantages + values
        return returns, advantages

    def get_batch(
        self, last_value: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Return all collected data as a training batch.

        Returns:
            Dictionary with states, actions, log_probs, returns, advantages
        """
        returns, advantages = self.compute_returns_and_advantages(last_value)

        states = torch.stack([t.state for t in self.transitions])
        actions = torch.tensor([t.action for t in self.transitions], dtype=torch.long)
        old_log_probs = torch.tensor(
            [t.log_prob for t in self.transitions], dtype=torch.float
        )

        # Normalize advantages for training stability
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
        }

    def clear(self):
        """Reset buffer for next rollout."""
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)


class A2CTrainer:
    """
    Advantage Actor-Critic (A2C) trainer.

    Synchronous single-worker A2C with:
      - GAE advantage estimation
      - Entropy bonus for exploration
      - Value function clipping

    References:
      - "Asynchronous Methods for Deep RL" (Mnih et al., 2016)
      - Stable-Baselines3 A2C implementation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """
        Args:
            model: Actor-Critic network (returns logits, value)
            optimizer: Optimizer for model parameters
            gamma: Discount factor
            gae_lambda: GAE lambda
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
        """
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

        # Training stats
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards: deque = deque(maxlen=100)

    def select_action(
        self, state: torch.Tensor, task_name: str = "snake"
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: Current state tensor
            task_name: Task identifier for the model

        Returns:
            (action, log_prob, value_estimate)
        """
        with torch.no_grad():
            logits, value = self.model(state.unsqueeze(0), task_name)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def store_transition(self, transition: Transition):
        """Add transition to rollout buffer."""
        self.buffer.push(transition)
        self.total_steps += 1

    def train_step(
        self, last_value: float = 0.0, task_name: str = "snake"
    ) -> Dict[str, float]:
        """
        Perform one A2C update from collected rollout.

        Args:
            last_value: Bootstrap value for the last state
            task_name: Task for forward pass

        Returns:
            Dictionary of loss components
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        batch = self.buffer.get_batch(last_value)

        # Forward pass
        logits, values = self.model(batch["states"], task_name)
        values = values.squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(batch["actions"])
        entropy = dist.entropy().mean()

        # Policy loss (negative because we maximize)
        policy_loss = -(log_probs * batch["advantages"].detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values, batch["returns"].detach())

        # Total loss
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer.

    Implements PPO-Clip with:
      - Clipped surrogate objective
      - Multiple epochs per rollout
      - Mini-batch updates
      - Value function clipping

    References:
      - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 32,
    ):
        """
        Args:
            model: Actor-Critic network (returns logits, value)
            optimizer: Optimizer for model parameters
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping range (epsilon)
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            n_epochs: PPO update epochs per rollout
            batch_size: Mini-batch size for updates
        """
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

        # Training stats
        self.total_steps = 0
        self.clip_fraction_history: deque = deque(maxlen=100)

    def select_action(
        self, state: torch.Tensor, task_name: str = "snake"
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Returns:
            (action, log_prob, value_estimate)
        """
        with torch.no_grad():
            logits, value = self.model(state.unsqueeze(0), task_name)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def store_transition(self, transition: Transition):
        """Add transition to rollout buffer."""
        self.buffer.push(transition)
        self.total_steps += 1

    def train_step(
        self, last_value: float = 0.0, task_name: str = "snake"
    ) -> Dict[str, float]:
        """
        Perform PPO update with multiple epochs and mini-batches.

        Args:
            last_value: Bootstrap value for the last state
            task_name: Task for forward pass

        Returns:
            Dictionary of loss components (averaged over epochs)
        """
        if len(self.buffer) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
            }

        batch = self.buffer.get_batch(last_value)
        n = batch["states"].shape[0]

        total_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
        }
        update_count = 0

        for _ in range(self.n_epochs):
            # Generate random mini-batch indices
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                mb_idx = indices[start:end]

                mb_states = batch["states"][mb_idx]
                mb_actions = batch["actions"][mb_idx]
                mb_old_log_probs = batch["old_log_probs"][mb_idx]
                mb_returns = batch["returns"][mb_idx]
                mb_advantages = batch["advantages"][mb_idx]

                # Forward pass
                logits, values = self.model(mb_states, task_name)
                values = values.squeeze(-1)

                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns.detach())

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Track clip fraction
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                )

                total_stats["policy_loss"] += policy_loss.item()
                total_stats["value_loss"] += value_loss.item()
                total_stats["entropy"] += entropy.item()
                total_stats["clip_fraction"] += clip_fraction
                update_count += 1

        # Average stats
        if update_count > 0:
            for k in total_stats:
                total_stats[k] /= update_count

        self.clip_fraction_history.append(total_stats["clip_fraction"])

        # Clear buffer
        self.buffer.clear()

        return total_stats


class NeuralSymbolicBridge:
    """
    Phase 1.1: Neural-Symbolic Integration (ROADMAP_TO_AGI.md)

    Bridges the neural policy (fast, generalizable) with symbolic rules
    (interpretable, safe). Implements arbitration between them.

    Architecture:
        Neural Policy → [Abstraction] → Symbolic Memory (VSA)
        Symbolic Rules → Safety Override → Final Action
    """

    def __init__(
        self,
        rule_learner=None,
        safety_threshold: float = 0.3,
    ):
        """
        Args:
            rule_learner: Symbolic rule learner (nsck rule_learner module)
            safety_threshold: Min confidence for neural override of rules
        """
        self.rule_learner = rule_learner
        self.safety_threshold = safety_threshold
        self.override_count = 0
        self.neural_wins = 0
        self.symbolic_wins = 0

    def arbitrate(
        self,
        neural_action: int,
        neural_confidence: float,
        state: Dict[str, Any],
        task_name: str = "snake",
    ) -> Tuple[int, str]:
        """
        Arbitrate between neural policy and symbolic rules.

        The symbolic system acts as a safety layer: if rules strongly
        disagree with the neural policy, the rules win.

        Args:
            neural_action: Action proposed by neural policy
            neural_confidence: Confidence of neural policy
            state: Current state dictionary
            task_name: Current task

        Returns:
            (final_action, source) where source is "neural" or "symbolic"
        """
        if self.rule_learner is None:
            self.neural_wins += 1
            return neural_action, "neural"

        # Check symbolic rules
        applicable_rules = self.rule_learner.get_applicable_rules(state, task_name)

        if not applicable_rules:
            self.neural_wins += 1
            return neural_action, "neural"

        # Find best rule
        best_rule = max(applicable_rules, key=lambda r: r.get("confidence", 0))
        rule_action = best_rule.get("action", neural_action)
        rule_confidence = best_rule.get("confidence", 0)

        # Arbitration logic:
        # Neural wins if confident enough AND no strong rule disagreement
        if neural_confidence > rule_confidence and neural_confidence > self.safety_threshold:
            self.neural_wins += 1
            return neural_action, "neural"

        # Symbolic override (safety)
        if rule_confidence > neural_confidence:
            self.override_count += 1
            self.symbolic_wins += 1
            return rule_action, "symbolic"

        self.neural_wins += 1
        return neural_action, "neural"

    def extract_rules_from_policy(
        self,
        model: nn.Module,
        states: List[torch.Tensor],
        task_name: str = "snake",
    ) -> List[Dict[str, Any]]:
        """
        Extract interpretable rules from neural policy decisions.

        Phase 1.1: Rule extraction from learned policy.
        Observes policy decisions and extracts action patterns.

        Args:
            model: Trained neural policy
            states: Representative state tensors
            task_name: Task to extract rules for

        Returns:
            List of extracted rule dictionaries
        """
        rules = []
        action_counts: Dict[int, int] = {}
        state_action_pairs: List[Tuple[Dict, int]] = []

        with torch.no_grad():
            for state in states:
                logits, _ = model(state.unsqueeze(0), task_name)
                action = logits.argmax(dim=-1).item()
                action_counts[action] = action_counts.get(action, 0) + 1
                state_action_pairs.append((state, action))

        # Extract dominant action patterns
        total = len(states)
        for action, count in action_counts.items():
            confidence = count / total if total > 0 else 0
            if confidence > 0.1:  # At least 10% frequency
                rules.append({
                    "action": action,
                    "confidence": confidence,
                    "support": count,
                    "source": "neural_extraction",
                    "task": task_name,
                })

        return rules

    def get_stats(self) -> Dict[str, Any]:
        """Return arbitration statistics."""
        total = self.neural_wins + self.symbolic_wins
        return {
            "neural_wins": self.neural_wins,
            "symbolic_wins": self.symbolic_wins,
            "override_count": self.override_count,
            "neural_ratio": self.neural_wins / total if total > 0 else 0,
        }
