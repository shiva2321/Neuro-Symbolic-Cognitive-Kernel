"""
NSCK Multi-Task Learning Module (Phase 1.2)
==========================================
Implements shared encoder with task-specific heads for multi-task learning.
Enables knowledge transfer across different tasks (Snake, Pong, Maze).

Design Principles (per AGENT_INSTRUCTIONS.md):
  - Efficiency first: O(n) operations, keep dims ≤ 128
  - Shared convolutional/linear encoder for common representations
  - Task-specific actor-critic heads
  - Gradient surgery to avoid negative transfer
  - Task balancing for equal updates per task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MultiTaskMetrics:
    """Metrics for multi-task training."""
    epoch: int = 0
    task_losses: Dict[str, float] = field(default_factory=dict)
    task_accuracies: Dict[str, float] = field(default_factory=dict)
    total_loss: float = 0.0
    gradient_conflicts: int = 0


class SharedEncoder(nn.Module):
    """
    Shared encoder for multi-task learning.
    
    Extracts common features across tasks (Snake, Pong, Maze).
    Uses lightweight architecture (≤128 dims) for efficiency.
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, latent_dim: int = 64):
        """
        Args:
            input_dim: Input feature dimension (e.g., flattened game state)
            hidden_dim: Hidden layer dimension (≤128 for efficiency)
            latent_dim: Shared representation dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Shared encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to shared latent space.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Shared latent representation [batch_size, latent_dim]
        """
        return self.encoder(x)


class TaskHead(nn.Module):
    """
    Task-specific actor-critic head.
    
    Takes shared representation and produces task-specific:
    - Action logits (policy)
    - Value estimate (critic)
    """
    
    def __init__(self, latent_dim: int = 64, num_actions: int = 4, hidden_dim: int = 64):
        """
        Args:
            latent_dim: Shared representation dimension
            num_actions: Number of actions for this task
            hidden_dim: Hidden layer dimension for task head
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action logits and value estimate.
        
        Args:
            latent: Shared representation [batch_size, latent_dim]
            
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        action_logits = self.actor(latent)
        value = self.critic(latent)
        return action_logits, value


class MultiTaskNetwork(nn.Module):
    """
    Multi-task neural network with shared encoder and task-specific heads.
    
    Architecture:
        Input → SharedEncoder → [TaskHead_1, TaskHead_2, ..., TaskHead_N]
        
    Enables transfer learning across tasks while maintaining task-specific policies.
    """
    
    def __init__(self, input_dim: int = 512, latent_dim: int = 64, 
                 task_configs: Optional[Dict[str, int]] = None):
        """
        Args:
            input_dim: Input dimension (flattened state)
            latent_dim: Shared representation dimension
            task_configs: Dict mapping task_name -> num_actions
                         e.g., {"snake": 4, "pong": 3, "maze": 4}
        """
        super().__init__()
        
        # Default task configurations
        if task_configs is None:
            task_configs = {
                "snake": 4,  # UP, DOWN, LEFT, RIGHT
                "pong": 3,   # UP, DOWN, STAY
                "maze": 4    # UP, DOWN, LEFT, RIGHT
            }
        
        self.task_configs = task_configs
        self.latent_dim = latent_dim
        
        # Shared encoder
        self.shared_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dim=128,
            latent_dim=latent_dim
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task_name: TaskHead(latent_dim, num_actions)
            for task_name, num_actions in task_configs.items()
        })
        
    def forward(self, x: torch.Tensor, task_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a specific task.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            task_name: Name of the task (e.g., "snake", "pong", "maze")
            
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        # Encode to shared representation
        latent = self.shared_encoder(x)
        
        # Task-specific forward
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.task_heads.keys())}")
        
        action_logits, value = self.task_heads[task_name](latent)
        return action_logits, value
    
    def get_shared_params(self) -> List[torch.nn.Parameter]:
        """Get parameters of the shared encoder."""
        return list(self.shared_encoder.parameters())
    
    def get_task_params(self, task_name: str) -> List[torch.nn.Parameter]:
        """Get parameters of a specific task head."""
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")
        return list(self.task_heads[task_name].parameters())


class GradientSurgery:
    """
    Implements gradient surgery to avoid negative transfer.
    
    Based on "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020).
    Projects conflicting gradients to reduce interference.
    """
    
    @staticmethod
    def project_conflicting_gradients(gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Project gradients to avoid conflicts.
        
        Args:
            gradients: Dict mapping task_name -> gradient tensor
            
        Returns:
            Dict of projected gradients
        """
        if len(gradients) <= 1:
            return gradients
        
        task_names = list(gradients.keys())
        grad_list = [gradients[name] for name in task_names]
        
        # Compute pairwise cosine similarities
        # If negative (conflicting), project out the conflicting component
        projected = {}
        for i, name1 in enumerate(task_names):
            grad1 = grad_list[i]
            projected_grad = grad1.clone()
            
            for j, name2 in enumerate(task_names):
                if i == j:
                    continue
                    
                grad2 = grad_list[j]
                
                # Cosine similarity
                dot_product = torch.dot(grad1.flatten(), grad2.flatten())
                norm1 = torch.norm(grad1)
                norm2 = torch.norm(grad2)
                
                if norm1 > 1e-8 and norm2 > 1e-8:
                    cos_sim = dot_product / (norm1 * norm2)
                    
                    # If conflicting (negative cosine), project out
                    if cos_sim < -0.01:  # Small threshold to avoid numerical issues
                        # Project grad1 onto the orthogonal complement of grad2
                        projection = (dot_product / (norm2 ** 2)) * grad2
                        projected_grad = projected_grad - projection
                        
                        # Ensure projected gradient is not zero
                        if torch.norm(projected_grad) < 1e-8:
                            # Use average of gradients as fallback
                            projected_grad = (grad1 + grad2) / 2.0
            
            projected[name1] = projected_grad
        
        return projected


class MultiTaskTrainer:
    """
    Multi-task trainer with gradient surgery and task balancing.
    
    Features:
    - Shared encoder training
    - Task-specific head training
    - Gradient surgery for conflict resolution
    - Task balancing for equal updates
    """
    
    def __init__(self, model: MultiTaskNetwork, lr: float = 0.001, 
                 task_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            model: Multi-task network
            lr: Learning rate
            task_weights: Optional weights for task balancing
        """
        self.model = model
        self.lr = lr
        
        # Task weights for balancing (default: equal)
        if task_weights is None:
            task_weights = {name: 1.0 for name in model.task_configs.keys()}
        self.task_weights = task_weights
        
        # Optimizers
        self.shared_optimizer = torch.optim.Adam(
            model.get_shared_params(), lr=lr
        )
        self.task_optimizers = {
            name: torch.optim.Adam(model.get_task_params(name), lr=lr)
            for name in model.task_configs.keys()
        }
        
        self.metrics = MultiTaskMetrics()
        
    def train_step(self, batch_data: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                   use_gradient_surgery: bool = True) -> MultiTaskMetrics:
        """
        Perform one training step across multiple tasks.
        
        Args:
            batch_data: Dict mapping task_name -> (states, actions, returns)
            use_gradient_surgery: Whether to use gradient surgery
            
        Returns:
            Training metrics
        """
        # Zero gradients
        self.shared_optimizer.zero_grad()
        for optimizer in self.task_optimizers.values():
            optimizer.zero_grad()
        
        # Compute losses and gradients for each task
        task_losses = {}
        shared_gradients = {}
        
        for task_name, (states, actions, returns) in batch_data.items():
            if task_name not in self.model.task_configs:
                continue
            
            # Forward pass
            action_logits, values = self.model(states, task_name)
            
            # Policy loss (negative log likelihood)
            log_probs = F.log_softmax(action_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Advantage estimate (simplified)
            advantages = returns - values.squeeze(1)
            policy_loss = -(selected_log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(1), returns)
            
            # Combined loss
            task_loss = policy_loss + 0.5 * value_loss
            task_losses[task_name] = task_loss.item()
            
            # Compute gradients for shared encoder
            weighted_loss = self.task_weights[task_name] * task_loss
            weighted_loss.backward(retain_graph=True)
            
            # Collect shared gradients for surgery
            if use_gradient_surgery:
                shared_grad = torch.cat([
                    p.grad.flatten() for p in self.model.get_shared_params()
                    if p.grad is not None
                ])
                shared_gradients[task_name] = shared_grad
        
        # Apply gradient surgery if enabled
        if use_gradient_surgery and len(shared_gradients) > 1:
            projected = GradientSurgery.project_conflicting_gradients(shared_gradients)
            
            # Apply projected gradients
            idx = 0
            for p in self.model.get_shared_params():
                if p.grad is not None:
                    numel = p.grad.numel()
                    # Average projected gradients from all tasks
                    avg_grad = torch.stack([
                        projected[name][idx:idx+numel].view(p.grad.shape)
                        for name in projected.keys()
                    ]).mean(dim=0)
                    p.grad.data = avg_grad
                    idx += numel
        
        # Update shared encoder
        self.shared_optimizer.step()
        
        # Update task-specific heads
        for optimizer in self.task_optimizers.values():
            optimizer.step()
        
        # Update metrics
        self.metrics.task_losses = task_losses
        self.metrics.total_loss = sum(task_losses.values()) / len(task_losses)
        
        return self.metrics


# Example usage and integration helpers
def create_snake_input_tensor(state_dict: Dict[str, Any], device: str = "cpu") -> torch.Tensor:
    """Convert Snake game state to input tensor."""
    # Simple flattening of state (customize based on actual state structure)
    # Assuming state contains position, direction, food location, etc.
    features = []
    
    # Extract relevant features (this is a simplified example)
    if "head_pos" in state_dict:
        features.extend(state_dict["head_pos"])
    if "direction" in state_dict:
        features.append(state_dict["direction"])
    if "food_pos" in state_dict:
        features.extend(state_dict["food_pos"])
    
    # Pad to input_dim (512) if needed
    while len(features) < 512:
        features.append(0.0)
    
    return torch.tensor(features[:512], dtype=torch.float32, device=device).unsqueeze(0)


def create_multitask_network(device: str = "cpu") -> MultiTaskNetwork:
    """Create a multi-task network for Snake, Pong, and Maze."""
    task_configs = {
        "snake": 4,  # UP, DOWN, LEFT, RIGHT
        "pong": 3,   # UP, DOWN, STAY
        "maze": 4    # UP, DOWN, LEFT, RIGHT
    }
    
    model = MultiTaskNetwork(
        input_dim=512,
        latent_dim=64,
        task_configs=task_configs
    )
    
    return model.to(device)
