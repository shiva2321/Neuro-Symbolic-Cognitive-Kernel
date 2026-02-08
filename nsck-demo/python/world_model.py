"""
NSCK World Model Module
Dynamics predictor for mental simulation and imagination.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class WorldModelConfig:
    hv_dim: int = 10240
    hidden_dim: int = 512
    learning_rate: float = 0.001
    device: str = "cpu"

class DynamicsPredictor(nn.Module):
    """
    Neural-Symbolic Dynamics Predictor.
    Learns SITUATION_t + ACTION_t -> SITUATION_t+1 + REWARD_t+1
    """
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # State + Action (concatenated or bound?)
        # For simplicity, we'll concatenate the 10,240-bit state and action
        # Note: In a real VSA system, we might BIND them, but for neural learning, 
        # concatenation often works better if the vectors are sparse/normalized.
        
        input_dim = config.hv_dim * 2 # State HV + Action HV
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        
        # Predictor heads
        self.state_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hv_dim),
            nn.Sigmoid() # Bits are 0 or 1
        )
        self.reward_head = nn.Linear(config.hidden_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.to(config.device)

    def forward(self, state_hv: torch.Tensor, action_hv: torch.Tensor):
        x = torch.cat([state_hv, action_hv], dim=-1)
        features = self.network(x)
        
        next_state = self.state_head(features)
        reward = self.reward_head(features)
        
        return next_state, reward

    def train_step(self, state_hv, action_hv, next_state_hv, reward):
        self.optimizer.zero_grad()
        
        # Convert to torch if needed
        s = torch.FloatTensor(state_hv).to(self.config.device)
        a = torch.FloatTensor(action_hv).to(self.config.device)
        ns = torch.FloatTensor(next_state_hv).to(self.config.device)
        r = torch.FloatTensor([reward]).to(self.config.device)
        
        pred_ns, pred_r = self.forward(s, a)
        
        loss_state = self.loss_fn(pred_ns, ns)
        loss_reward = self.loss_fn(pred_r, r)
        
        # Boost reward loss and use plain state loss (already mean-scaled by MSELoss)
        loss = loss_state + 100.0 * loss_reward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def predict(self, state_hv: np.ndarray, action_hv: np.ndarray) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state_hv).to(self.config.device)
            a = torch.FloatTensor(action_hv).to(self.config.device)
            
            pred_ns, pred_r = self.forward(s, a)
            
            return pred_ns.cpu().numpy(), float(pred_r.cpu().item())

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
