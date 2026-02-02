"""
NSCK Meta-Learning Module (Phase 4.2)
=====================================
Implements MAML (Model-Agnostic Meta-Learning) and Reptile.
Enables rapid few-shot adaptation to new tasks and environments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import List, Dict, Any, Tuple, Callable

class MAMLLearner:
    """
    Meta-algorithm for fast adaptation.
    Optimizes for a set of weights that are easy to fine-tune.
    """
    def __init__(self, model: nn.Module, meta_lr: float = 0.001, inner_lr: float = 0.01):
        self.model = model
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr

    def _inner_loop_update(self, support_inputs, support_targets, steps: int = 5):
        """
        Adapt the model to a specific task using a support set.
        Returns a 'fast' model (functional or cloned).
        """
        # Create a copy for task-specific weights
        # For true MAML, we'd use higher-order gradients/functional API.
        # Here we implement a first-order approximation for stability.
        fast_model = copy.deepcopy(self.model)
        fast_optimizer = optim.SGD(fast_model.parameters(), lr=self.inner_lr)
        criterion = nn.CrossEntropyLoss()
        
        fast_model.train()
        for _ in range(steps):
            fast_optimizer.zero_grad()
            outputs = fast_model(support_inputs)
            
            # Aggregate SNN output
            if outputs.dim() == 3: outputs = outputs.sum(dim=0)
            
            loss = criterion(outputs, support_targets)
            loss.backward()
            fast_optimizer.step()
            
        return fast_model

    def meta_step(self, task_batch: List[Dict[str, torch.Tensor]]):
        """
        Perform a single meta-update across a batch of tasks.
        Each task has a 'support' set for adaptation and a 'query' set for evaluation.
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0
        
        for task in task_batch:
            # 1. Adapt to task using support set (Inner Loop)
            adapted_model = self._inner_loop_update(
                task['support_input'], 
                task['support_target']
            )
            
            # 2. Evaluate adapted model on query set (Outer Loop)
            # This loss tells us how well the adaptation worked.
            query_outputs = adapted_model(task['query_input'])
            if query_outputs.dim() == 3: query_outputs = query_outputs.sum(dim=0)
            
            criterion = nn.CrossEntropyLoss()
            task_loss = criterion(query_outputs, task['query_target'])
            
            # 3. Accumulated loss
            task_loss.backward() # Gradients w.r.t. meta-parameters (original model)
            meta_loss += task_loss.item()
            
        # 4. Meta-update
        # We average the gradients across tasks
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.div_(len(task_batch))
                
        self.meta_optimizer.step()
        return meta_loss / len(task_batch)

class ReptileLearner:
    """
    Simpler Meta-Learning (Nichol et al., 2018).
    Weights = Weights + epsilon * (AdaptedWeights - Weights)
    """
    def __init__(self, model: nn.Module, meta_lr: float = 0.1):
        self.model = model
        self.meta_lr = meta_lr

    def meta_step(self, task_inputs, task_targets, inner_steps: int = 10, inner_lr: float = 0.01):
        # 1. Store initial weights
        weights_before = {name: p.data.clone() for name, p in self.model.named_parameters()}
        
        # 2. Perform several steps of SGD on the task (Inner Loop)
        optimizer = optim.SGD(self.model.parameters(), lr=inner_lr)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(inner_steps):
            optimizer.zero_grad()
            outputs = self.model(task_inputs)
            if outputs.dim() == 3: outputs = outputs.sum(dim=0)
            loss = criterion(outputs, task_targets)
            loss.backward()
            optimizer.step()
            
        # 3. Update Meta-Weights (Outer Loop)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                new_val = weights_before[name] + self.meta_lr * (p.data - weights_before[name])
                p.data.copy_(new_val)

    def adapt(self, support_inputs, support_targets, steps: int = 5):
        """Few-shot adaptation for a new task."""
        # Just standard fine-tuning starting from the meta-learned initialization
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(steps):
            optimizer.zero_grad()
            outputs = self.model(support_inputs)
            if outputs.dim() == 3: outputs = outputs.sum(dim=0)
            loss = criterion(outputs, support_targets)
            loss.backward()
            optimizer.step()
