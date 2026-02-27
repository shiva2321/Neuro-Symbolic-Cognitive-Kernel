"""
NSCK Continual Learning Module
================================
Implements mechanisms to prevent catastrophic forgetting in neural components.

Features:
1. Elastic Weight Consolidation (EWC) for neural bridges
2. VSA-based task boundaries (task-specific HV subspaces)
3. Progressive neural-symbolic consolidation
4. Task-specific knowledge isolation

Theory:
- Each task gets its own HV subspace (via task_tag binding)
- Important weights are protected via Fisher Information Matrix
- Symbolic knowledge (rules, causal graphs) never suffers catastrophic forgetting
- Neural components (embeddings, bridges) apply EWC regularization
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import python.core.vsa.hypervec_shim as hypervec_rs


@dataclass
class TaskMemory:
    """Represents knowledge specific to a single task."""
    task_tag: str
    task_hv: hypervec_rs.HyperVector
    importance_weights: Dict[str, np.ndarray] = field(default_factory=dict)  # Fisher diagonal
    optimal_params: Dict[str, np.ndarray] = field(default_factory=dict)      # θ*
    num_samples: int = 0
    active: bool = True


class ContinualLearner:
    """
    Manages continual learning across multiple tasks.
    
    Prevents catastrophic forgetting through:
    1. VSA task isolation - each task operates in its own HV subspace
    2. EWC regularization - protects important neural weights
    3. Selective consolidation - only important knowledge persists
    """
    
    def __init__(self, ewc_lambda: float = 1000.0, task_capacity: int = 100):
        """
        Initialize continual learner.
        
        Args:
            ewc_lambda: EWC regularization strength (higher = stronger protection)
            task_capacity: Maximum number of tasks to remember
        """
        self.ewc_lambda = ewc_lambda
        self.task_capacity = task_capacity
        
        # Task-specific memories
        self.tasks: Dict[str, TaskMemory] = {}
        
        # Task HV registry for quick lookup
        self.task_hvs: Dict[str, hypervec_rs.HyperVector] = {}
        
        # Global importance accumulator (across all tasks)
        self.global_importance: Dict[str, np.ndarray] = {}
        
        print(f"ContinualLearner initialized (EWC λ={ewc_lambda})")
    
    def register_task(self, task_tag: str) -> hypervec_rs.HyperVector:
        """
        Register a new task and create its dedicated HV subspace.
        
        Args:
            task_tag: Unique task identifier
            
        Returns:
            Task-specific hypervector for knowledge binding
        """
        if task_tag in self.tasks:
            return self.tasks[task_tag].task_hv
        
        # Create task-specific HV
        task_hv = hypervec_rs.HyperVector(hash(task_tag) % (2**32))
        
        # Store task memory
        self.tasks[task_tag] = TaskMemory(
            task_tag=task_tag,
            task_hv=task_hv
        )
        self.task_hvs[task_tag] = task_hv
        
        print(f"[CONTINUAL] Registered task: {task_tag}")
        return task_hv
    
    def compute_importance(
        self,
        task_tag: str,
        params: Dict[str, np.ndarray],
        gradients: Dict[str, np.ndarray]
    ):
        """
        Compute Fisher Information Matrix (diagonal approximation).
        
        This estimates which parameters are important for the current task
        by computing the squared gradients at convergence.
        
        Args:
            task_tag: Task identifier
            params: Current network parameters
            gradients: Gradients at optimal point
        """
        if task_tag not in self.tasks:
            self.register_task(task_tag)
        
        task = self.tasks[task_tag]
        
        # Fisher Information ≈ E[∇log p(y|x,θ)]²
        # We approximate using squared gradients
        for param_name, grad in gradients.items():
            if param_name not in task.importance_weights:
                task.importance_weights[param_name] = np.zeros_like(grad)
            
            # Accumulate squared gradients (Fisher diagonal)
            task.importance_weights[param_name] += grad ** 2
            task.num_samples += 1
        
        # Store optimal parameters
        task.optimal_params = {k: v.copy() for k, v in params.items()}
        
        print(f"[CONTINUAL] Computed importance for {task_tag} (n={task.num_samples})")
    
    def ewc_loss(
        self,
        current_params: Dict[str, np.ndarray],
        exclude_task: Optional[str] = None
    ) -> float:
        """
        Compute EWC regularization loss.
        
        Penalizes deviation from important parameters of previous tasks:
        L_EWC = (λ/2) Σ_i F_i (θ_i - θ*_i)²
        
        Args:
            current_params: Current network parameters
            exclude_task: Task to exclude from penalty (current task)
            
        Returns:
            EWC regularization loss
        """
        total_loss = 0.0
        
        for task_tag, task in self.tasks.items():
            if task_tag == exclude_task or not task.active:
                continue
            
            if not task.optimal_params or not task.importance_weights:
                continue
            
            # EWC penalty for this task
            task_loss = 0.0
            for param_name, optimal in task.optimal_params.items():
                if param_name not in current_params:
                    continue
                
                current = current_params[param_name]
                importance = task.importance_weights[param_name]
                
                # Average the Fisher diagonal if we have multiple samples
                if task.num_samples > 1:
                    importance = importance / task.num_samples
                
                # Quadratic penalty weighted by importance
                diff = current - optimal
                task_loss += np.sum(importance * (diff ** 2))
            
            total_loss += task_loss
        
        return (self.ewc_lambda / 2.0) * total_loss
    
    def bind_to_task(
        self,
        concept_hv: hypervec_rs.HyperVector,
        task_tag: str
    ) -> hypervec_rs.HyperVector:
        """
        Bind a concept to a task-specific subspace.
        
        This creates task-isolated knowledge: the same concept in different
        tasks has different representations, preventing interference.
        
        Args:
            concept_hv: Base concept hypervector
            task_tag: Task to bind to
            
        Returns:
            Task-bound hypervector
        """
        if task_tag not in self.task_hvs:
            task_hv = self.register_task(task_tag)
        else:
            task_hv = self.task_hvs[task_tag]
        
        # XOR binding creates orthogonal subspaces
        return concept_hv.xor(task_hv)
    
    def consolidate_task(self, task_tag: str):
        """
        Mark a task as consolidated (knowledge is frozen).
        
        Future learning won't modify this task's optimal parameters.
        
        Args:
            task_tag: Task to consolidate
        """
        if task_tag not in self.tasks:
            return
        
        task = self.tasks[task_tag]
        
        # Normalize importance weights
        for param_name in task.importance_weights:
            if task.num_samples > 1:
                task.importance_weights[param_name] /= task.num_samples
        
        # Add to global importance (for cross-task protection)
        for param_name, importance in task.importance_weights.items():
            if param_name not in self.global_importance:
                self.global_importance[param_name] = np.zeros_like(importance)
            self.global_importance[param_name] += importance
        
        print(f"[CONTINUAL] Consolidated task: {task_tag}")
    
    def retrieve_task(self, query_hv: hypervec_rs.HyperVector, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Identify which task(s) a query belongs to via HV similarity.
        
        Args:
            query_hv: Query hypervector
            top_k: Return top K most similar tasks
            
        Returns:
            List of (task_tag, similarity) tuples
        """
        similarities = []
        
        for task_tag, task_hv in self.task_hvs.items():
            if not self.tasks[task_tag].active:
                continue
            
            sim = query_hv.similarity(task_hv)
            similarities.append((task_tag, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def measure_forgetting(
        self,
        task_tag: str,
        current_params: Dict[str, np.ndarray]
    ) -> float:
        """
        Measure how much a task's knowledge has been forgotten.
        
        Computes normalized L2 distance from optimal parameters.
        
        Args:
            task_tag: Task to measure
            current_params: Current network parameters
            
        Returns:
            Forgetting score (0 = no forgetting, 1 = complete forgetting)
        """
        if task_tag not in self.tasks:
            return 0.0
        
        task = self.tasks[task_tag]
        if not task.optimal_params:
            return 0.0
        
        total_diff = 0.0
        total_norm = 0.0
        
        for param_name, optimal in task.optimal_params.items():
            if param_name not in current_params:
                continue
            
            current = current_params[param_name]
            diff = current - optimal
            
            total_diff += np.sum(diff ** 2)
            total_norm += np.sum(optimal ** 2)
        
        if total_norm == 0:
            return 0.0
        
        # Normalized forgetting score
        return np.sqrt(total_diff / total_norm)
    
    def get_protected_concepts(self, task_tag: str) -> List[str]:
        """Return the top-10 most important concept names for a task."""
        if task_tag not in self.tasks:
            return []
        task = self.tasks[task_tag]
        if not task.importance_weights:
            return []
        sorted_items = sorted(
            task.importance_weights.items(),
            key=lambda x: float(np.sum(x[1])),
            reverse=True,
        )
        return [name for name, _ in sorted_items[:10]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get continual learning statistics."""
        active_tasks = sum(1 for t in self.tasks.values() if t.active)
        consolidated_tasks = sum(1 for t in self.tasks.values() if not t.active)
        
        total_params = 0
        for task in self.tasks.values():
            if task.optimal_params:
                total_params += sum(p.size for p in task.optimal_params.values())
        
        return {
            "num_tasks": len(self.tasks),
            "active_tasks": active_tasks,
            "consolidated_tasks": consolidated_tasks,
            "ewc_lambda": self.ewc_lambda,
            "total_protected_params": total_params,
            "avg_samples_per_task": np.mean([t.num_samples for t in self.tasks.values()]) if self.tasks else 0
        }


# Example usage
if __name__ == "__main__":
    learner = ContinualLearner(ewc_lambda=1000.0)
    
    # Register tasks
    task1_hv = learner.register_task("robot_navigation")
    task2_hv = learner.register_task("object_manipulation")
    
    # Bind concepts to tasks
    concept_hv = hypervec_rs.HyperVector(12345)
    task1_concept = learner.bind_to_task(concept_hv, "robot_navigation")
    task2_concept = learner.bind_to_task(concept_hv, "object_manipulation")
    
    print(f"Same concept, different tasks are orthogonal: "
          f"sim={task1_concept.similarity(task2_concept):.3f}")
    
    # Simulate learning
    params = {"weights": np.random.randn(10, 10)}
    gradients = {"weights": np.random.randn(10, 10)}
    
    learner.compute_importance("robot_navigation", params, gradients)
    learner.consolidate_task("robot_navigation")
    
    print(learner.get_statistics())
