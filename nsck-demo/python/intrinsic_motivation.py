"""
NSCK Intrinsic Motivation Module
================================

Implements Intrinsic Curiosity Module (ICM) for autonomous exploration.

The core idea: Give the agent internal rewards based on prediction error.
If the agent cannot predict what will happen next, the state is "novel"
and worth exploring.

Architecture:
    State → Feature_Net → φ(s)
    (φ(s), action) → Forward_Model → predicted φ(s')
    (φ(s), φ(s')) → Inverse_Model → predicted action
    
    Intrinsic_Reward = ||predicted_φ(s') - actual_φ(s')||²

This module is SEPARATE from the SNN - it provides reward signal, not actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CuriosityConfig:
    """Configuration for intrinsic motivation."""
    latent_dim: int = 256          # Feature embedding dimension
    learning_rate: float = 1e-4    # ICM training learning rate
    beta: float = 0.2              # Weight of forward loss vs inverse loss
    intrinsic_scale: float = 0.01  # Scale factor for intrinsic reward
    update_freq: int = 8           # Train ICM every N steps
    batch_size: int = 32           # Batch size for ICM training


class FeatureNetwork(nn.Module):
    """
    Extracts state features for prediction models.
    Shared between forward and inverse models.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        
        # Visual pathway (for game frames)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract features from state.
        
        Args:
            state: [B, C, H, W] or [B, H, W] tensor
            
        Returns:
            [B, latent_dim] feature tensor
        """
        # Handle different input shapes
        # Handle different input shapes
        # Input is expected to be [B, 4, 10, 10] (Frame Stack)
        if state.dim() == 3:
             # If just [4, 10, 10], add batch dim
             state = state.unsqueeze(0)
        if state.dim() == 2:
            # Assume flattened, try to reshape to 10x10 (Snake grid)
            state = state.view(-1, 1, 10, 10)
            
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ForwardModel(nn.Module):
    """
    Predicts next state features given current features and action.
    
    The prediction error is the intrinsic reward signal.
    """
    def __init__(self, latent_dim: int = 256, num_actions: int = 4):
        super().__init__()
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 32)
        
        # Dynamics prediction
        self.fc1 = nn.Linear(latent_dim + 32, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        
    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state features.
        
        Args:
            features: [B, latent_dim] current state features
            action: [B] action indices
            
        Returns:
            [B, latent_dim] predicted next state features
        """
        action_emb = self.action_embed(action)
        x = torch.cat([features, action_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class InverseModel(nn.Module):
    """
    Predicts action given current and next state features.
    
    This helps learn action-relevant features (filter out noise).
    """
    def __init__(self, latent_dim: int = 256, num_actions: int = 4):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim * 2, 256)
        self.fc2 = nn.Linear(256, num_actions)
        
    def forward(self, features: torch.Tensor, next_features: torch.Tensor) -> torch.Tensor:
        """
        Predict action from state transition.
        
        Args:
            features: [B, latent_dim] current state features
            next_features: [B, latent_dim] next state features
            
        Returns:
            [B, num_actions] action logits
        """
        x = torch.cat([features, next_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class IntrinsicCuriosityModule(nn.Module):
    """
    Complete Intrinsic Curiosity Module (ICM).
    
    Provides intrinsic rewards based on state prediction error.
    Trains itself on transitions to improve predictions.
    """
    
    def __init__(self, config: CuriosityConfig = None, num_actions: int = 4):
        super().__init__()
        self.config = config or CuriosityConfig()
        self.num_actions = num_actions
        
        # Sub-modules
        self.feature_net = FeatureNetwork(self.config.latent_dim)
        self.forward_model = ForwardModel(self.config.latent_dim, num_actions)
        self.inverse_model = InverseModel(self.config.latent_dim, num_actions)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Experience buffer for batch training
        self.buffer = deque(maxlen=10000)
        self.step_count = 0
        
        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
    def compute_intrinsic_reward(
        self, 
        state: torch.Tensor, 
        action: int, 
        next_state: torch.Tensor
    ) -> float:
        """
        Compute intrinsic reward for a single transition.
        
        Args:
            state: Current state tensor
            action: Action taken (integer)
            next_state: Next state tensor
            
        Returns:
            Intrinsic reward (float)
        """
        with torch.no_grad():
            # Ensure batch dimension
            if state.dim() < 4:
                state = state.unsqueeze(0)
            if next_state.dim() < 4:
                next_state = next_state.unsqueeze(0)
                
            action_t = torch.tensor([action], dtype=torch.long, device=state.device)
            
            # Get features
            phi_s = self.feature_net(state)
            phi_s_next = self.feature_net(next_state)
            
            # Predict next features
            phi_pred = self.forward_model(phi_s, action_t)
            
            # Prediction error = intrinsic reward
            error = F.mse_loss(phi_pred, phi_s_next).item()
            
            # Normalize reward
            reward = self._normalize_reward(error)
            
        return reward * self.config.intrinsic_scale
    
    def _normalize_reward(self, reward: float) -> float:
        """Running normalization of intrinsic rewards."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var += delta * delta2
        
        std = max(1e-6, np.sqrt(self.reward_var / max(1, self.reward_count)))
        return (reward - self.reward_mean) / std
    
    def store_transition(
        self, 
        state: torch.Tensor, 
        action: int, 
        next_state: torch.Tensor
    ):
        """Store transition for batch training."""
        self.buffer.append({
            'state': state.detach().cpu(),
            'action': action,
            'next_state': next_state.detach().cpu()
        })
        
        self.step_count += 1
        if self.step_count % self.config.update_freq == 0:
            self.train_step()
    
    def train_step(self) -> Dict[str, float]:
        """Train ICM on stored transitions."""
        if len(self.buffer) < self.config.batch_size:
            return {}
            
        # Sample batch
        indices = np.random.choice(
            len(self.buffer), 
            self.config.batch_size, 
            replace=False
        )
        batch = [self.buffer[i] for i in indices]
        
        # Collate
        states = torch.stack([t['state'] for t in batch])
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long)
        next_states = torch.stack([t['next_state'] for t in batch])
        
        # Move to device
        device = next(self.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        
        # Forward pass
        phi_s = self.feature_net(states)
        phi_s_next = self.feature_net(next_states)
        
        # Forward model loss (prediction error)
        phi_pred = self.forward_model(phi_s, actions)
        forward_loss = F.mse_loss(phi_pred, phi_s_next.detach())
        
        # Inverse model loss (action prediction for feature learning)
        action_logits = self.inverse_model(phi_s, phi_s_next)
        inverse_loss = F.cross_entropy(action_logits, actions)
        
        # Combined loss
        beta = self.config.beta
        total_loss = beta * forward_loss + (1 - beta) * inverse_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get module statistics."""
        return {
            'buffer_size': len(self.buffer),
            'reward_mean': self.reward_mean,
            'reward_std': np.sqrt(self.reward_var / max(1, self.reward_count)),
            'step_count': self.step_count
        }


class CountBasedExploration:
    """
    Simple count-based exploration bonus.
    
    States visited less frequently get higher exploration bonuses.
    Uses state hashing for tabular tracking.
    """
    
    def __init__(self, scale: float = 0.1, hash_dim: int = 32):
        self.scale = scale
        self.hash_dim = hash_dim
        self.visit_counts = {}
        
        # Random projection for state hashing
        self.projection = None
        
    def _hash_state(self, state: torch.Tensor) -> str:
        """Create hashable key from state tensor."""
        if self.projection is None:
            flat_size = state.numel()
            self.projection = torch.randn(flat_size, self.hash_dim)
            
        # Project to lower dimension and discretize
        flat = state.flatten().cpu()
        # Resize projection if needed
        if flat.shape[0] != self.projection.shape[0]:
            self.projection = torch.randn(flat.shape[0], self.hash_dim)
            
        projected = torch.matmul(flat.float(), self.projection)
        bits = (projected > 0).int()
        return ''.join(map(str, bits.tolist()))
    
    def get_bonus(self, state: torch.Tensor) -> float:
        """
        Get exploration bonus for a state.
        
        Bonus = scale / sqrt(count + 1)
        """
        key = self._hash_state(state)
        count = self.visit_counts.get(key, 0)
        return self.scale / np.sqrt(count + 1)
    
    def visit(self, state: torch.Tensor):
        """Record a state visit."""
        key = self._hash_state(state)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1
        
    def get_stats(self) -> Dict[str, any]:
        """Get exploration statistics."""
        counts = list(self.visit_counts.values())
        return {
            'unique_states': len(self.visit_counts),
            'total_visits': sum(counts),
            'max_visits': max(counts) if counts else 0,
            'avg_visits': np.mean(counts) if counts else 0
        }


class CombinedIntrinsicMotivation:
    """
    Combines multiple intrinsic motivation signals.

    - ICM: Prediction error bonus
    - Count: Visit count bonus
    - Progress: Learning progress bonus (optional)
    """

    def __init__(self, num_actions: int = 4, device: str = 'cpu'):
        self.device = device

        # ICM for prediction-based curiosity
        self.icm = IntrinsicCuriosityModule(num_actions=num_actions)
        self.icm.to(device)

        # Count-based exploration
        self.count_explorer = CountBasedExploration()

        # Weights for combining signals
        self.icm_weight = 1.0
        self.count_weight = 0.5

    def set_device(self, device: str) -> None:
        """Move ICM networks (and optimizer state) to a new device."""
        self.device = str(device)
        self.icm.to(device)
        # Move optimizer state tensors too (otherwise Adam keeps CPU tensors)
        for state in self.icm.optimizer.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def compute_reward(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        extrinsic_reward: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined intrinsic + extrinsic reward.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            extrinsic_reward: External reward from environment

        Returns:
            (total_reward, breakdown_dict)
        """
        # Ensure tensors are on the same device as the ICM (prevents CUDA/CPU mismatch)
        icm_device = next(self.icm.parameters()).device
        if state.device != icm_device:
            state = state.to(icm_device)
        if next_state.device != icm_device:
            next_state = next_state.to(icm_device)

        # ICM curiosity bonus
        icm_bonus = self.icm.compute_intrinsic_reward(state, action, next_state)

        # Count-based bonus (CPU-only hashing is fine)
        count_bonus = self.count_explorer.get_bonus(next_state)

        # Store transition for ICM training
        self.icm.store_transition(state, action, next_state)

        # Record visit
        self.count_explorer.visit(next_state)

        # Combine rewards
        intrinsic = self.icm_weight * icm_bonus + self.count_weight * count_bonus
        total = extrinsic_reward + intrinsic

        return total, {
            'extrinsic': extrinsic_reward,
            'icm_bonus': icm_bonus,
            'count_bonus': count_bonus,
            'intrinsic_total': intrinsic,
            'total': total
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get combined statistics."""
        return {
            'icm': self.icm.get_stats(),
            'count': self.count_explorer.get_stats()
        }


# -----------------------------------------------------------------------------
# Test / Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Intrinsic Curiosity Module Test ===\n")
    
    # Create module
    motivation = CombinedIntrinsicMotivation(num_actions=4, device='cpu')
    
    # Simulate some transitions
    print("Simulating 100 random transitions...")
    for i in range(100):
        state = torch.randn(1, 10, 10)
        action = np.random.randint(0, 4)
        next_state = state + torch.randn_like(state) * 0.1
        
        total_reward, breakdown = motivation.compute_reward(
            state, action, next_state, extrinsic_reward=0.0
        )
        
        if i % 20 == 0:
            print(f"Step {i}: total={total_reward:.4f}, "
                  f"icm={breakdown['icm_bonus']:.4f}, "
                  f"count={breakdown['count_bonus']:.4f}")
    
    print("\n=== Statistics ===")
    stats = motivation.get_stats()
    print(f"ICM buffer size: {stats['icm']['buffer_size']}")
    print(f"Unique states visited: {stats['count']['unique_states']}")
    print(f"ICM reward mean: {stats['icm']['reward_mean']:.4f}")
    
    print("\n[OK] Intrinsic Motivation Module Ready!")
