"""
NSCK Continual Learning Module (Phase 4.1)
=========================================
Implements Elastic Weight Consolidation (EWC) and PackNet.
Prevents catastrophic forgetting when learning multiple games/tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

class ContinualLearner:
    """
    Lifelong learning with catastrophic forgetting prevention.
    Uses EWC to protect important weights for previous tasks.
    """
    def __init__(self, model: nn.Module, lambda_ewc: float = 5000.0):
        self.model = model
        
        # Track importance of each weight: task_id -> {param_name -> importance_tensor}
        self.weight_importance = {}
        
        # Store optimal weights for each task: task_id -> {param_name -> weights_tensor}
        self.task_optimal_weights = {}
        
        self.lambda_ewc = lambda_ewc

    def compute_weight_importance(self, task_id: str, data_loader, criterion=nn.CrossEntropyLoss()):
        """
        Compute Fisher Information Matrix (FIM) for the current task.
        FIM approximates which weights are critical for the task's performance.
        """
        print(f"[CONTINUAL] Computing weight importance for task: {task_id}")
        self.model.eval()
        importance = {}
        
        # Initialize importance tensors
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = torch.zeros_like(param.data)

        # Approximate FIM using squared gradients
        # We iterate through the data to see which parameters affect the loss most
        count = 0
        for batch in data_loader:
            inputs, targets = batch
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Handle SNN output (if temporal, sum spikes)
            if hasattr(outputs, 'sum') and outputs.dim() == 3:
                outputs = outputs.sum(dim=0)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Accumulate squared gradients (diagonal FIM approximation)
                    importance[name].add_(param.grad.data ** 2)
            
            count += 1
            if count >= 100: # Limit samples for performance
                break
                
        # Average and store
        for name in importance:
            importance[name] /= count
            
        self.weight_importance[task_id] = importance
        
        # Store current weights as optimal for this task
        self.task_optimal_weights[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute total EWC regularization loss across all previous tasks.
        L = L_current + sum( lambda/2 * F_i * (theta - theta_i)^2 )
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        # Guard if no previous tasks
        if not self.weight_importance:
            return loss
            
        for task_id in self.weight_importance:
            importance = self.weight_importance[task_id]
            optimal_weights = self.task_optimal_weights[task_id]
            
            for name, param in self.model.named_parameters():
                if name in importance:
                    # Penalty for moving away from previously learned optimal weights
                    diff = param - optimal_weights[name]
                    loss += (importance[name] * (diff ** 2)).sum()
        
        return (self.lambda_ewc / 2.0) * loss

class PackNetManager:
    """
    Pack multiple tasks into a single network using iterative pruning (Phase 4.1).
    Allows dedicated parameters for specific tasks.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.task_masks = {} # task_id -> {param_name -> binary_mask}
        self.current_free_mask = {} # param_name -> binary_mask (1 means free)
        
        # Initialize free mask as all ones
        for name, param in self.model.named_parameters():
            if "weight" in name:
                self.current_free_mask[name] = torch.ones_like(param.data)

    def prune_and_allocate(self, task_id: str, prune_percentage: float = 0.5):
        """
        Prune low-magnitude weights and allocate them to the task_id.
        Reserved weights are masked out for future tasks.
        """
        task_mask = {}
        
        for name, param in self.model.named_parameters():
            if name not in self.current_free_mask:
                continue
                
            # Only consider weights that are CURRENTLY FREE
            free_weights = param.data * self.current_free_mask[name]
            
            # Find threshold for magnitude-based pruning
            abs_weights = torch.abs(free_weights)
            # We want to keep (1 - prune_percentage) of currently free weights
            # But the roadmap says "Prune task subnet to free up capacity"
            # Standard PackNet: Train -> Prune -> Fix -> Train new...
            
            # Simplified version: Allocate a fixed percentage of remaining capacity
            # In a real system, we'd sort and pick the top K.
            t = torch.quantile(abs_weights[abs_weights > 0], prune_percentage)
            
            # Mask for THIS task
            this_task_mask = (abs_weights >= t).float() * self.current_free_mask[name]
            task_mask[name] = this_task_mask
            
            # Update global free mask: these weights are no longer free
            self.current_free_mask[name] -= this_task_mask
            
        self.task_masks[task_id] = task_mask

    def apply_task_mask(self, task_id: str):
        """Apply the mask for a specific task during inference."""
        if task_id not in self.task_masks:
            return
            
        mask = self.task_masks[task_id]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask:
                    param.data *= mask[name]

    def get_capacity_stats(self) -> Dict[str, float]:
        """Return percentage of weights allocated vs free."""
        stats = {"free": 0.0}
        total_params = 0
        total_free = 0
        
        for name, mask in self.current_free_mask.items():
            total_params += mask.numel()
            total_free += mask.sum().item()
            
        stats["free"] = total_free / total_params if total_params > 0 else 0
        for task_id in self.task_masks:
            allocated = 0
            for name, mask in self.task_masks[task_id].items():
                allocated += mask.sum().item()
            stats[task_id] = allocated / total_params
            
        return stats
