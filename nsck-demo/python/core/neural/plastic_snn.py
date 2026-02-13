"""
NSCK Structural Plasticity: Deep Rewiring and Neurogenesis
====================================
Implements Deep Rewiring (Bellec et al. 2018) and Neurogenesis.

Key Components:
- SparseLinear: A layer with latent connectivity (Theta) and fixed sparsity.
- PlasticSNN: A network that grows when learning stalls.

Mechanisms:
- Forward: W = s * relu(theta)
- Rewire: Regrow dormant connections to maintain density.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity # Fraction of connections to keep ACTIVE
        
        # Latent Parameters (Theta)
        # We init theta so that exactly 'sparsity' fraction are positive
        self.theta = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Fixed Sign Matrix (Buffer = not optimized)
        self.register_buffer("sign", torch.empty(out_features, in_features))
        
        # Init
        self._reset_parameters()

    def _reset_parameters(self):
        # 1. Init Theta with mixed +/- values near zero
        # Standard Kaiming init but scaled
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        
        # Apply Sparsity Mask immediately to Theta init
        # Set (1-sparsity)% of weights to -epsilon (Dormant)
        total = self.theta.numel()
        n_active = int(total * self.sparsity)
        n_dormant = total - n_active
        
        # Randomly choose indices to kill
        flat_theta = self.theta.data.view(-1)
        indices = torch.randperm(total)
        dormant_idx = indices[:n_dormant]
        active_idx = indices[n_dormant:]
        
        # Set active to positive small, dormant to negative small
        flat_theta[active_idx] = flat_theta[active_idx].abs().clamp(min=0.01)
        flat_theta[dormant_idx] = -0.01
        
        # 2. Init Signs (Fixed random +/- 1)
        self.sign.data.uniform_(-1, 1)
        self.sign.data.sign_() # {-1, 1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Effective Weight: s * relu(theta)
        # ReLU acts as the Mask M = (theta > 0)
        w_eff = self.sign * torch.relu(self.theta)
        return torch.nn.functional.linear(x, w_eff, self.bias)

    def rewire(self):
        """
        Deep Rewiring Step.
        1. Check current sparsity.
        2. Regrow connections if below target.
        """
        with torch.no_grad():
            total = self.theta.numel()
            target_active = int(total * self.sparsity)
            
            # Current active count
            active_mask = (self.theta > 0)
            current_active = active_mask.sum().item()
            
            n_regrow = target_active - current_active
            
            if n_regrow > 0:
                # Regrowth (as before)
                dormant_indices = (~active_mask).nonzero(as_tuple=True)
                n_dormant = dormant_indices[0].size(0)
                n_regrow = min(n_regrow, n_dormant)
                
                if n_regrow > 0:
                    perm = torch.randperm(n_dormant)[:n_regrow]
                    idx0 = dormant_indices[0][perm]
                    idx1 = dormant_indices[1][perm]
                    
                    # Init: Epsilon + Noise
                    epsilon = 1e-3
                    noise = torch.randn(n_regrow, device=self.theta.device) * 1e-4
                    
                    self.theta.data[idx0, idx1] = epsilon + noise
                    
            elif n_regrow < 0:
                # Forced Pruning (if L1 is too slow or density drifts up)
                # Select n_prune ACTIVE items with smallest magnitude and kill them
                n_prune = -n_regrow
                
                # Get indices of active items
                active_indices = active_mask.nonzero(as_tuple=True)
                active_values = self.theta.data[active_indices] # Should be positive
                
                # Sort by magnitude (ascending)
                _, sorted_idx = torch.sort(active_values)
                params_to_kill = sorted_idx[:n_prune]
                
                # Kill them (Set to -epsilon)
                idx0 = active_indices[0][params_to_kill]
                idx1 = active_indices[1][params_to_kill]
                
                self.theta.data[idx0, idx1] = -1e-3

    def get_mask(self) -> torch.Tensor:
        return (self.theta > 0).float()


class PlasticSNN(nn.Module):
    """
    Simple MLP-like SNN substrate to test plasticity.
    (Not the full TaskAwareSNN yet)
    """
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = SparseLinear(input_dim, hidden_dim, sparsity=0.3)
        self.fc2 = SparseLinear(hidden_dim, output_dim, sparsity=0.5)
        
        # Plateau Detection
        self.loss_history = []
        self.patience = 5
        self.window = 20
        self.cooldown = 0
        
    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Simple Activation
        x = self.fc2(x)
        return x
        
    def check_plateau(self, current_loss: float) -> bool:
        """Returns True if growth triggered."""
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
            
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.window + self.patience:
            # Compare current avg to past avg
            recent = np.mean(self.loss_history[-self.patience:])
            past = np.mean(self.loss_history[-(self.window+self.patience):-self.patience])
            
            # If improvement is negligible (< 1%)
            if (past - recent) / (past + 1e-6) < 0.01:
                self.cooldown = 20 # Wait before growing again
                return True
                
        return False

    def expand_capacity(self, new_neurons: int = 5):
        """Neurogenesis: Resizes fc1."""
        print(f"[Neurogenesis] Adding {new_neurons} neurons to Hidden Layer.")
        old_layer = self.fc1
        
        # New dimensions
        new_out = old_layer.out_features + new_neurons
        new_layer = SparseLinear(old_layer.in_features, new_out, sparsity=old_layer.sparsity)
        
        # Copy Existing Params
        with torch.no_grad():
            # Copy active rows
            n_old = old_layer.out_features
            new_layer.theta.data[:n_old, :] = old_layer.theta.data
            new_layer.sign.data[:n_old, :] = old_layer.sign.data
            new_layer.bias.data[:n_old] = old_layer.bias.data
            
        # Replace
        self.fc1 = new_layer
        
        # Also need to expand fc2 (Input side)
        old_fc2 = self.fc2
        new_fc2 = SparseLinear(new_out, old_fc2.out_features, sparsity=old_fc2.sparsity)
        
        with torch.no_grad():
            # Copy cols
            new_fc2.theta.data[:, :n_old] = old_fc2.theta.data
            new_fc2.sign.data[:, :n_old] = old_fc2.sign.data
            
        self.fc2 = new_fc2

import math
