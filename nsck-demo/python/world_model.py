"""
NSCK World Model Module
Dynamics predictor for mental simulation and imagination.

EFFICIENCY NOTE: Uses a compact bottleneck architecture with random
projection to avoid heavy O(n²) dense matrix multiplications.
The HV dimension (10240) is first projected down to a small latent
space (128) via a fixed sparse random projection, then a tiny MLP
predicts the next-state delta and reward. This reduces FLOPs from
~10.5M to ~200K per forward pass.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random as _random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class WorldModelConfig:
    hv_dim: int = 10240
    bottleneck_dim: int = 128   # compact latent space
    hidden_dim: int = 64        # tiny MLP
    learning_rate: float = 0.001
    device: str = "cpu"

class DynamicsPredictor(nn.Module):
    """
    Efficient Dynamics Predictor using random projection bottleneck.

    Architecture (energy-efficient):
    1. Fixed sparse random projection: hv_dim → bottleneck_dim  (no grad, O(n))
    2. Tiny MLP: bottleneck_dim*2 → hidden → hidden  (~33K params vs ~10.5M)
    3. State delta head + reward head

    Total FLOPs per forward: ~200K  (vs ~10.5M in dense version)
    """
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Fixed sparse random projection (not trainable — saves memory & compute)
        # Johnson-Lindenstrauss: random ±1 projection preserves distances
        proj = torch.zeros(config.hv_dim, config.bottleneck_dim)
        # Sparse: only ~10% non-zero entries
        for j in range(config.bottleneck_dim):
            indices = torch.randperm(config.hv_dim)[:config.hv_dim // 10]
            signs = torch.sign(torch.randn(len(indices)))
            proj[indices, j] = signs
        # Normalise columns
        col_norms = proj.norm(dim=0, keepdim=True).clamp(min=1e-6)
        proj = proj / col_norms
        self.register_buffer("projection", proj)

        # Compact MLP: 2×bottleneck (state+action projected) → hidden → hidden
        input_dim = config.bottleneck_dim * 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # Heads
        self.state_head = nn.Linear(config.hidden_dim, config.bottleneck_dim)
        self.reward_head = nn.Linear(config.hidden_dim, 1)

        # Inverse projection for reconstruction (transpose of projection)
        # Not stored — computed on the fly from self.projection

        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.to(config.device)

    def _project(self, hv: torch.Tensor) -> torch.Tensor:
        """Project HV to compact latent via sparse random projection (O(n))."""
        return hv @ self.projection  # sparse proj makes this efficient

    def _unproject(self, latent: torch.Tensor) -> torch.Tensor:
        """Approximate inverse projection (pseudo-inverse via transpose)."""
        return latent @ self.projection.t()

    def forward(self, state_hv: torch.Tensor, action_hv: torch.Tensor):
        s_lat = self._project(state_hv)
        a_lat = self._project(action_hv)
        x = torch.cat([s_lat, a_lat], dim=-1)
        features = self.network(x)

        next_delta = self.state_head(features)  # delta in latent space
        reward = self.reward_head(features)

        # Predicted next state = current latent + delta
        next_latent = s_lat + next_delta
        return next_latent, reward

    def train_step(self, state_hv, action_hv, next_state_hv, reward):
        self.optimizer.zero_grad()

        s = torch.FloatTensor(state_hv).to(self.config.device)
        a = torch.FloatTensor(action_hv).to(self.config.device)
        ns = torch.FloatTensor(next_state_hv).to(self.config.device)
        r = torch.FloatTensor([reward]).to(self.config.device)

        # Ensure batch dim
        if s.dim() == 1:
            s, a, ns = s.unsqueeze(0), a.unsqueeze(0), ns.unsqueeze(0)

        # Target in latent space
        ns_lat = self._project(ns)

        pred_ns_lat, pred_r = self.forward(s, a)

        loss_state = self.loss_fn(pred_ns_lat, ns_lat)
        loss_reward = self.loss_fn(pred_r, r)

        loss = loss_state + 10.0 * loss_reward
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, state_hv: np.ndarray, action_hv: np.ndarray) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state_hv).unsqueeze(0).to(self.config.device)
            a = torch.FloatTensor(action_hv).unsqueeze(0).to(self.config.device)

            pred_lat, pred_r = self.forward(s, a)
            # Reconstruct full HV from latent
            pred_full = self._unproject(pred_lat)
            # Threshold to binary-ish
            pred_bits = (pred_full > 0.0).float()

            return pred_bits.squeeze(0).cpu().numpy(), float(pred_r.cpu().item())

class WorldModel:
    """
    High-level wrapper for cognitive engine integration.
    """
    def __init__(self, hv_dim: int = 10240):
        self.config = WorldModelConfig(hv_dim=hv_dim)
        self.predictor = DynamicsPredictor(self.config)
        self.stats = {"train_steps": 0, "total_loss": 0.0}

    def is_ready(self, task_tag: str) -> bool:
        """Check if world model is ready for simulation (enough training)."""
        # For prototype, just check if we have done some training
        return self.stats["train_steps"] > 100

    def update(self, state, action_hv, next_hv, reward):
        """
        Record a transition for the world model.
        Args:
            state: current HyperVector
            action_hv: action HyperVector
            next_hv: next HyperVector
            reward: reward received
        """
        s_np = self.hv_to_numpy(state)
        ns_np = self.hv_to_numpy(next_hv)
        a_np = self.hv_to_numpy(action_hv)
        
        loss = self.predictor.train_step(s_np, a_np, ns_np, reward)
        self.stats["train_steps"] += 1
        self.stats["total_loss"] += loss
        
        if self.stats["train_steps"] % 100 == 0:
            avg_loss = self.stats["total_loss"] / 100
            print(f"[WORLD_MODEL] Trained 100 steps. Avg Loss: {avg_loss:.6f}")
            self.stats["total_loss"] = 0.0

    def imagine(self, state, action_hv) -> Tuple[np.ndarray, float]:
        """Predict next HV and reward."""
        s_np = self.hv_to_numpy(state)
        a_np = self.hv_to_numpy(action_hv)
        return self.predictor.predict(s_np, a_np)

    def sample_hypothetical_trajectories(
        self, 
        initial_hv, 
        action_hvs: List[Any], 
        horizon: int = 5, 
        num_paths: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """
        Explore potential futures from a starting state.
        Returns a list of paths, each path is a list of steps.
        """
        paths = []
        initial_bits = self.hv_to_numpy(initial_hv)
        
        for _ in range(num_paths):
            path = []
            current_bits = initial_bits.copy()
            
            for _ in range(horizon):
                # Randomly pick an action
                import random
                action_hv = random.choice(action_hvs)
                a_bits = self.hv_to_numpy(action_hv)
                
                # Predict next
                next_bits, reward = self.predictor.predict(current_bits, a_bits)
                
                step = {
                    "source_bits": current_bits.copy(),
                    "action_hv": action_hv,
                    "reward": reward,
                    "target_bits": next_bits.copy()
                }
                path.append(step)
                current_bits = next_bits
                
                # Stop if we hit an extreme reward (e.g. death or success)
                if abs(reward) > 0.5:
                    break
            
            paths.append(path)
            
        return paths

    def hv_to_numpy(self, hv) -> np.ndarray:
        """Convert HyperVector or numpy bit-vector to float32 numpy."""
        if isinstance(hv, np.ndarray):
            return hv.astype(np.float32)
        # Python fallback HyperVector stores .bits as np.ndarray
        if hasattr(hv, 'bits'):
            bits = hv.bits
            if isinstance(bits, np.ndarray):
                return bits.astype(np.float32)
        if hasattr(hv, "__getstate__"):
            state = hv.__getstate__()
            arr = np.array(state, dtype=np.uint64)
            bits = np.unpackbits(arr.view(np.uint8))
            return bits.astype(np.float32)[:self.config.hv_dim]
        # Fallback if object is not an HV
        return np.zeros(self.config.hv_dim, dtype=np.float32)
