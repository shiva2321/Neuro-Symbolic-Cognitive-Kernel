"""
NSCK World Model Module
Dynamics predictor for mental simulation and imagination.

Architecture (hybrid symbolic + lightweight numeric):
─────────────────────────────────────────────────────
1. **VSA Transition Memory** (primary, symbolic):
   Stores (state_HV ⊗ action_HV) → (next_state_HV, reward) as bound
   pairs.  Retrieval via Hamming similarity — O(N·D) lookup, no matrix
   multiplication.  Generalises to novel situations by finding the most
   similar stored transition.

2. **Random-Projection Linear Model** (secondary, lightweight):
   Projects 10 240-bit HVs down to a 128-dim bottleneck via a *fixed*
   sparse random matrix (not learned), then a tiny linear layer predicts
   state-delta + reward.  Total FLOPs ~200K per forward.  Provides
   gradient-based interpolation when the symbolic memory has no close match.

3. **Ensemble** (optional): Multiple predictors for uncertainty estimation
   via disagreement.

FALLBACK: When PyTorch is not available, a pure-numpy linear model
provides the same API surface with simpler gradient-free learning.
"""
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import numpy as np
import random as _random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# VSA Transition Memory (fully symbolic, no neural nets)
# ---------------------------------------------------------------------------
class VSATransitionMemory:
    """Symbolic world model using VSA hypervector operations.

    Each transition (s, a) → (s', r) is stored as:
      key_HV   = s_bits XOR a_bits           (binding = context HV)
      value    = (next_state_bits, reward)

    Retrieval: given (s, a), compute key = s ⊕ a, find the stored key
    with highest Hamming similarity, return the associated (s', r).

    Generalisation: because XOR-binding preserves component similarity,
    a novel state similar to a previously seen state will produce a key
    that is close to the stored key, enabling analogical prediction.

    Capacity: uses a ring buffer of fixed size (default 2000 transitions)
    to bound memory.  For large histories, an LSH index could replace
    the linear scan.
    """

    def __init__(self, hv_dim: int = 10240, capacity: int = 2000):
        self.dim = hv_dim
        self.capacity = capacity
        # Ring buffer storage
        self._keys: List[np.ndarray] = []       # bound (state ⊕ action) bit arrays
        self._next_states: List[np.ndarray] = [] # next-state bit arrays
        self._rewards: List[float] = []
        self._cursor = 0
        self._full = False

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._cursor

    def store(self, state_bits: np.ndarray, action_bits: np.ndarray,
              next_state_bits: np.ndarray, reward: float):
        """Record a single (s, a) → (s', r) transition."""
        key = np.bitwise_xor(
            state_bits.astype(np.int8), action_bits.astype(np.int8)
        )
        if self._full:
            idx = self._cursor % self.capacity
            self._keys[idx] = key
            self._next_states[idx] = next_state_bits.astype(np.int8)
            self._rewards[idx] = reward
        else:
            self._keys.append(key)
            self._next_states.append(next_state_bits.astype(np.int8))
            self._rewards.append(reward)
        self._cursor += 1
        if self._cursor >= self.capacity and not self._full:
            self._full = True
            self._cursor = 0

    def predict(self, state_bits: np.ndarray, action_bits: np.ndarray,
                k: int = 3) -> Optional[Tuple[np.ndarray, float, float]]:
        """Predict next state and reward for (s, a).

        Uses k-nearest-neighbour voting over stored transitions.
        Returns (predicted_next_bits, predicted_reward, similarity) or
        None if memory is empty.
        """
        if not self._keys:
            return None

        query = np.bitwise_xor(
            state_bits.astype(np.int8), action_bits.astype(np.int8)
        )

        # Compute Hamming similarity to all stored keys
        n = self.size
        sims = np.zeros(n, dtype=np.float64)
        for i in range(n):
            diff = np.bitwise_xor(query, self._keys[i])
            sims[i] = 1.0 - np.sum(diff) / self.dim

        # Top-k neighbours
        k = min(k, n)
        top_indices = np.argpartition(-sims, k)[:k]
        top_sims = sims[top_indices]

        best_sim = float(np.max(top_sims))

        # Weighted average reward
        weights = top_sims - 0.45  # shift so ~0.5 baseline → small weight
        weights = np.clip(weights, 0.01, None)
        w_sum = weights.sum()

        pred_reward = float(np.dot(weights, [self._rewards[i] for i in top_indices]) / w_sum)

        # Majority-vote bits from top-k neighbours
        stacked = np.stack([self._next_states[i].astype(np.float64) for i in top_indices])
        weighted = (stacked * weights[:, None]).sum(axis=0) / w_sum
        pred_bits = (weighted > 0.5).astype(np.float32)

        return pred_bits, pred_reward, best_sim

@dataclass
class WorldModelConfig:
    hv_dim: int = 10240
    bottleneck_dim: int = 128   # compact latent space
    hidden_dim: int = 128       # doubled from 64 for better capacity
    learning_rate: float = 0.001
    device: str = "cpu"
    n_ensemble: int = 3         # ensemble members for uncertainty
    dropout: float = 0.1        # dropout for regularisation

if _HAS_TORCH:
    class DynamicsPredictor(nn.Module):
        """
        Efficient Dynamics Predictor using random projection bottleneck.

        Architecture (energy-efficient):
        1. Fixed sparse random projection: hv_dim -> bottleneck_dim  (no grad, O(n))
        2. Tiny MLP: bottleneck_dim*2 -> hidden -> hidden  (~33K params vs ~10.5M)
        3. State delta head + reward head

        Total FLOPs per forward: ~200K  (vs ~10.5M in dense version)
        """

        def __init__(self, config: WorldModelConfig):
            super().__init__()
            self.config = config

            # Fixed sparse random projection (not trainable)
            proj = torch.zeros(config.hv_dim, config.bottleneck_dim)
            for j in range(config.bottleneck_dim):
                indices = torch.randperm(config.hv_dim)[:config.hv_dim // 10]
                signs = torch.sign(torch.randn(len(indices)))
                proj[indices, j] = signs
            col_norms = proj.norm(dim=0, keepdim=True).clamp(min=1e-6)
            proj = proj / col_norms
            self.register_buffer("projection", proj)

            input_dim = config.bottleneck_dim * 2
            self.network = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
            )

            self.state_head = nn.Linear(config.hidden_dim // 2, config.bottleneck_dim)
            self.reward_head = nn.Linear(config.hidden_dim // 2, 1)

            self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
            self.loss_fn = nn.MSELoss()
            self.to(config.device)

        def _project(self, hv: torch.Tensor) -> torch.Tensor:
            return hv @ self.projection

        def _unproject(self, latent: torch.Tensor) -> torch.Tensor:
            return latent @ self.projection.t()

        def forward(self, state_hv: torch.Tensor, action_hv: torch.Tensor):
            s_lat = self._project(state_hv)
            a_lat = self._project(action_hv)
            x = torch.cat([s_lat, a_lat], dim=-1)
            features = self.network(x)
            next_delta = self.state_head(features)
            reward = self.reward_head(features)
            next_latent = s_lat + next_delta
            return next_latent, reward

        def train_step(self, state_hv, action_hv, next_state_hv, reward):
            self.optimizer.zero_grad()
            s = torch.FloatTensor(state_hv).to(self.config.device)
            a = torch.FloatTensor(action_hv).to(self.config.device)
            ns = torch.FloatTensor(next_state_hv).to(self.config.device)
            r = torch.FloatTensor([reward]).to(self.config.device)
            if s.dim() == 1:
                s, a, ns = s.unsqueeze(0), a.unsqueeze(0), ns.unsqueeze(0)
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
                pred_full = self._unproject(pred_lat)
                pred_bits = (pred_full > 0.0).float()
                return pred_bits.squeeze(0).cpu().numpy(), float(pred_r.cpu().item())


# ---------------------------------------------------------------------------
# NUMPY FALLBACK: Used when PyTorch is not installed
# ---------------------------------------------------------------------------
class NumpyDynamicsPredictor:
    """
    Pure-numpy dynamics predictor using random projection + linear model.
    Provides the same predict/train_step API as the torch version.
    Uses online SGD over a compact latent space.
    """

    def __init__(self, config: WorldModelConfig):
        self.config = config
        rng = np.random.RandomState(42)

        # Sparse random projection matrix (same design as torch version)
        proj = np.zeros((config.hv_dim, config.bottleneck_dim), dtype=np.float32)
        for j in range(config.bottleneck_dim):
            indices = rng.choice(config.hv_dim, size=config.hv_dim // 10, replace=False)
            signs = rng.choice([-1.0, 1.0], size=len(indices)).astype(np.float32)
            proj[indices, j] = signs
        col_norms = np.linalg.norm(proj, axis=0, keepdims=True).clip(min=1e-6)
        self.projection = proj / col_norms

        # Linear weights: input (2*bottleneck) -> state delta (bottleneck) + reward (1)
        input_dim = config.bottleneck_dim * 2
        output_dim = config.bottleneck_dim + 1  # delta + reward
        scale = 0.01
        self.W = rng.randn(input_dim, output_dim).astype(np.float32) * scale
        self.b = np.zeros(output_dim, dtype=np.float32)
        self.lr = config.learning_rate

    def _project(self, hv: np.ndarray) -> np.ndarray:
        return hv @ self.projection

    def _unproject(self, latent: np.ndarray) -> np.ndarray:
        return latent @ self.projection.T

    def predict(self, state_hv: np.ndarray, action_hv: np.ndarray) -> Tuple[np.ndarray, float]:
        s_lat = self._project(state_hv.reshape(1, -1))
        a_lat = self._project(action_hv.reshape(1, -1))
        x = np.concatenate([s_lat, a_lat], axis=-1)
        out = x @ self.W + self.b

        delta = out[0, :-1]
        reward = float(out[0, -1])
        next_lat = s_lat[0] + delta
        pred_full = self._unproject(next_lat.reshape(1, -1))
        pred_bits = (pred_full > 0.0).astype(np.float32)
        return pred_bits.squeeze(0), reward

    def train_step(self, state_hv, action_hv, next_state_hv, reward) -> float:
        s = np.asarray(state_hv, dtype=np.float32).reshape(1, -1)
        a = np.asarray(action_hv, dtype=np.float32).reshape(1, -1)
        ns = np.asarray(next_state_hv, dtype=np.float32).reshape(1, -1)

        s_lat = self._project(s)
        a_lat = self._project(a)
        ns_lat = self._project(ns)
        x = np.concatenate([s_lat, a_lat], axis=-1)  # (1, 2*bottleneck)

        # Forward
        out = x @ self.W + self.b  # (1, bottleneck+1)
        delta = out[:, :-1]
        pred_reward = out[:, -1:]
        pred_ns_lat = s_lat + delta

        # Loss = MSE(state) + 10*MSE(reward)
        err_state = pred_ns_lat - ns_lat
        err_reward = pred_reward - np.array([[reward]], dtype=np.float32)
        loss = float(np.mean(err_state ** 2) + 10.0 * np.mean(err_reward ** 2))

        # Backward (analytical gradient for linear model)
        d_out = np.zeros_like(out)
        d_out[:, :-1] = 2.0 * err_state / err_state.size
        d_out[:, -1:] = 20.0 * err_reward
        grad_W = x.T @ d_out
        grad_b = d_out.squeeze(0)

        # SGD update
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return loss


class WorldModel:
    """
    High-level wrapper for cognitive engine integration.

    Hybrid architecture:
      1. VSATransitionMemory (primary) — symbolic, no matrix ops
      2. NumpyDynamicsPredictor or torch DynamicsPredictor (secondary) — lightweight
      3. Ensemble mode for uncertainty estimation

    Prediction strategy:
      - If VSA memory has a close match (sim ≥ 0.55), use it.
      - Otherwise, fall back to the numeric predictor.
      - If both available, blend with similarity-weighted average.
    """
    # Threshold for trusting VSA memory over numeric predictor
    VSA_TRUST_THRESHOLD = 0.55

    def __init__(self, hv_dim: int = 10240):
        self.config = WorldModelConfig(hv_dim=hv_dim)
        self._backend = "torch" if _HAS_TORCH else "numpy"

        # Primary: symbolic VSA transition memory
        self.vsa_memory = VSATransitionMemory(hv_dim=hv_dim)

        # Secondary: numeric ensemble
        n = self.config.n_ensemble
        if _HAS_TORCH:
            self.predictors = [DynamicsPredictor(self.config) for _ in range(n)]
            self.predictor = self.predictors[0]  # backward compat
        else:
            self.predictors = [NumpyDynamicsPredictor(self.config) for _ in range(n)]
            self.predictor = self.predictors[0]
        self.stats = {"train_steps": 0, "total_loss": 0.0, "vsa_hits": 0, "vsa_misses": 0}

    def is_ready(self, task_tag: str) -> bool:
        """Check if world model is ready for simulation (enough training)."""
        # For prototype, just check if we have done some training
        return self.stats["train_steps"] > 100

    def update(self, state, action_hv, next_hv, reward):
        """
        Record a transition for both the symbolic VSA memory and
        the numeric predictor ensemble.
        """
        s_np = self.hv_to_numpy(state)
        ns_np = self.hv_to_numpy(next_hv)
        a_np = self.hv_to_numpy(action_hv)

        # Store in symbolic VSA memory (primary)
        s_int = (s_np > 0.5).astype(np.int8)
        a_int = (a_np > 0.5).astype(np.int8)
        ns_int = (ns_np > 0.5).astype(np.int8)
        self.vsa_memory.store(s_int, a_int, ns_int, reward)

        # Train numeric ensemble (secondary)
        total_loss = 0.0
        for pred in self.predictors:
            loss = pred.train_step(s_np, a_np, ns_np, reward)
            total_loss += loss
        avg_loss = total_loss / len(self.predictors)
        
        self.stats["train_steps"] += 1
        self.stats["total_loss"] += avg_loss
        
        if self.stats["train_steps"] % 100 == 0:
            avg = self.stats["total_loss"] / 100
            vsa_sz = self.vsa_memory.size
            print(f"[WORLD_MODEL] 100 steps. Avg Loss: {avg:.6f}  VSA mem: {vsa_sz}  "
                  f"hits: {self.stats['vsa_hits']}  misses: {self.stats['vsa_misses']}")
            self.stats["total_loss"] = 0.0

    def imagine(self, state, action_hv) -> Tuple[np.ndarray, float]:
        """Predict next HV and reward using hybrid VSA + numeric strategy.

        1. Query VSA transition memory for nearest match.
        2. If confidence ≥ threshold, return symbolic prediction.
        3. Otherwise, fall back to numeric ensemble average.
        4. If both available, blend weighted by VSA confidence.
        """
        s_np = self.hv_to_numpy(state)
        a_np = self.hv_to_numpy(action_hv)

        s_int = (s_np > 0.5).astype(np.int8)
        a_int = (a_np > 0.5).astype(np.int8)

        # Try symbolic memory first
        vsa_result = self.vsa_memory.predict(s_int, a_int, k=3)

        # Numeric ensemble prediction
        all_bits = []
        all_rewards = []
        for pred in self.predictors:
            bits, reward = pred.predict(s_np, a_np)
            all_bits.append(bits)
            all_rewards.append(reward)
        stacked = np.stack(all_bits, axis=0)
        num_bits = (np.mean(stacked, axis=0) > 0.5).astype(np.float32)
        num_reward = float(np.mean(all_rewards))

        if vsa_result is not None:
            vsa_bits, vsa_reward, vsa_sim = vsa_result
            if vsa_sim >= self.VSA_TRUST_THRESHOLD:
                # VSA memory has a close enough match — trust it
                self.stats["vsa_hits"] = self.stats.get("vsa_hits", 0) + 1
                # Weighted blend: higher sim → more VSA weight
                alpha = min(1.0, (vsa_sim - 0.45) / 0.3)  # 0.45→0, 0.75→1
                blended_bits = (alpha * vsa_bits + (1 - alpha) * num_bits)
                blended_bits = (blended_bits > 0.5).astype(np.float32)
                blended_reward = alpha * vsa_reward + (1 - alpha) * num_reward
                return blended_bits, blended_reward

        # No close match in VSA → pure numeric
        self.stats["vsa_misses"] = self.stats.get("vsa_misses", 0) + 1
        return num_bits, num_reward

    def get_uncertainty(self, state, action_hv) -> float:
        """Estimate prediction uncertainty via ensemble disagreement."""
        s_np = self.hv_to_numpy(state)
        a_np = self.hv_to_numpy(action_hv)
        rewards = []
        for pred in self.predictors:
            _, r = pred.predict(s_np, a_np)
            rewards.append(r)
        return float(np.std(rewards))

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
