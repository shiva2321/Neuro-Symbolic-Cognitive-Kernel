"""
NSCK Meta-Learning Module
==========================
Implements meta-learning to "learn to learn faster" across tasks.

Features:
1. MAML-style (Model-Agnostic Meta-Learning) for VSA encoders
2. Task-agnostic initiative functions (curiosity, exploration)
3. Self-modifying cognitive strategies
4. Fast adaptation to new tasks with few examples

Theory:
- Meta-learn initialization parameters θ that enable fast adaptation
- Inner loop: task-specific learning (few gradient steps)
- Outer loop: meta-optimization across tasks
- VSA-specific: learn optimal binding patterns, bundle weights
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import python.core.vsa.hypervec_shim as hypervec_rs


@dataclass
class MetaTask:
    """Represents a meta-learning task."""
    task_id: str
    support_set: List[Tuple[Any, Any]]  # (input, output) pairs for adaptation
    query_set: List[Tuple[Any, Any]]     # (input, output) pairs for evaluation
    task_hv: hypervec_rs.HyperVector
    
    def __post_init__(self):
        if not self.task_hv:
            self.task_hv = hypervec_rs.HyperVector(hash(self.task_id) % (2**32))


@dataclass
class StrategyPerformance:
    """Tracks performance of a cognitive strategy."""
    strategy_name: str
    successes: int = 0
    failures: int = 0
    avg_reward: float = 0.0
    adaptation_steps: List[int] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0


class MetaLearner:
    """
    Implements meta-learning for fast task adaptation.
    
    Key capabilities:
    1. Few-shot learning - learn from 1-5 examples
    2. Strategy selection - choose optimal cognitive strategy per task
    3. Meta-VSA - learn optimal HV encoding schemes
    4. Self-modification - adjust own learning parameters
    """
    
    def __init__(
        self,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        adaptation_steps: int = 5,
        meta_batch_size: int = 4
    ):
        """
        Initialize meta-learner.
        
        Args:
            inner_lr: Learning rate for task-specific adaptation
            meta_lr: Learning rate for meta-optimization
            adaptation_steps: Number of gradient steps for inner loop
            meta_batch_size: Number of tasks per meta-update
        """
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.meta_batch_size = meta_batch_size
        
        # Meta-parameters (initialization for fast adaptation)
        self.meta_params: Dict[str, np.ndarray] = {}
        
        # Strategy library
        self.strategies: Dict[str, StrategyPerformance] = {}
        self._init_strategies()
        
        # Task memory for meta-learning
        self.task_history: List[MetaTask] = []
        
        # Learned binding patterns
        self.optimal_bindings: Dict[str, Tuple[str, str]] = {}  # concept_type -> (role, filler)
        
        print(f"MetaLearner initialized (inner_lr={inner_lr}, meta_lr={meta_lr})")
    
    def _init_strategies(self):
        """Initialize cognitive strategy library."""
        strategies = [
            "exploration",      # Curiosity-driven exploration
            "exploitation",     # Use known good actions
            "analogy",          # Transfer from similar tasks
            "planning",         # STRIPS planning
            "causal_reasoning", # Causal inference
            "episodic_recall",  # Memory-based decisions
            "rule_following",   # Apply learned rules
        ]
        
        for strategy in strategies:
            self.strategies[strategy] = StrategyPerformance(strategy_name=strategy)
    
    def add_task(
        self,
        task_id: str,
        support_set: List[Tuple[Any, Any]],
        query_set: List[Tuple[Any, Any]]
    ):
        """
        Add a task to the meta-learning dataset.
        
        Args:
            task_id: Unique task identifier
            support_set: Training examples for adaptation
            query_set: Test examples for evaluation
        """
        task = MetaTask(
            task_id=task_id,
            support_set=support_set,
            query_set=query_set,
            task_hv=hypervec_rs.HyperVector(hash(task_id) % (2**32))
        )
        self.task_history.append(task)
        print(f"[META] Added task: {task_id} (support={len(support_set)}, query={len(query_set)})")
    
    def adapt(
        self,
        task: MetaTask,
        base_params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Rapid adaptation to a new task (inner loop).
        
        Takes base parameters and adapts them to the task using support set.
        
        Args:
            task: Task to adapt to
            base_params: Starting parameters (uses meta_params if None)
            
        Returns:
            Adapted parameters
        """
        if base_params is None:
            base_params = self.meta_params.copy()
        
        adapted_params = {k: v.copy() for k, v in base_params.items()}
        
        # Inner loop: few-shot adaptation
        for step in range(self.adaptation_steps):
            # Compute gradients on support set
            gradients = self._compute_gradients(adapted_params, task.support_set)
            
            # Update parameters
            for param_name, grad in gradients.items():
                adapted_params[param_name] -= self.inner_lr * grad
        
        return adapted_params
    
    def meta_update(self, task_batch: List[MetaTask]):
        """
        Meta-optimization step (outer loop).
        
        Updates meta-parameters to improve adaptation across tasks.
        
        Args:
            task_batch: Batch of tasks for meta-update
        """
        meta_gradients = defaultdict(lambda: 0.0)
        
        for task in task_batch:
            # 1. Adapt to task using support set
            adapted_params = self.adapt(task, self.meta_params)
            
            # 2. Evaluate on query set
            query_loss = self._evaluate_loss(adapted_params, task.query_set)
            
            # 3. Compute meta-gradient (gradient of query loss w.r.t. meta-params)
            task_meta_grad = self._compute_meta_gradients(
                self.meta_params,
                adapted_params,
                task.support_set,
                task.query_set
            )
            
            # 4. Accumulate meta-gradients
            for param_name, grad in task_meta_grad.items():
                meta_gradients[param_name] += grad
        
        # 5. Meta-parameter update
        for param_name, grad in meta_gradients.items():
            if param_name in self.meta_params:
                self.meta_params[param_name] -= (self.meta_lr / len(task_batch)) * grad
        
        print(f"[META] Updated meta-parameters (batch_size={len(task_batch)})")
    
    def select_strategy(
        self,
        situation_hv: hypervec_rs.HyperVector,
        context: Dict[str, Any]
    ) -> str:
        """
        Meta-learned strategy selection.
        
        Choose optimal cognitive strategy based on situation and past performance.
        
        Args:
            situation_hv: Current situation hypervector
            context: Additional context information
            
        Returns:
            Selected strategy name
        """
        # Score each strategy based on historical performance
        scores = {}
        
        for strategy_name, perf in self.strategies.items():
            # Base score: success rate
            base_score = perf.success_rate
            
            # Bonus for consistent performance
            if len(perf.adaptation_steps) > 5:
                consistency = 1.0 / (1.0 + np.std(perf.adaptation_steps))
                base_score *= (1.0 + 0.2 * consistency)
            
            # Context-based modulation
            if strategy_name == "exploration" and context.get("novelty", 0) > 0.7:
                base_score *= 1.5  # Favor exploration in novel situations
            elif strategy_name == "exploitation" and context.get("confidence", 0) > 0.8:
                base_score *= 1.3  # Favor exploitation when confident
            elif strategy_name == "analogy" and context.get("similar_tasks", 0) > 2:
                base_score *= 1.4  # Favor analogy when similar tasks exist
            
            scores[strategy_name] = base_score
        
        # Epsilon-greedy for exploration
        if np.random.rand() < 0.1:
            return np.random.choice(list(self.strategies.keys()))
        
        # Select best strategy
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        return best_strategy
    
    def update_strategy_performance(
        self,
        strategy_name: str,
        success: bool,
        reward: float,
        steps_taken: int
    ):
        """
        Update strategy performance statistics.
        
        Args:
            strategy_name: Name of strategy used
            success: Whether strategy succeeded
            reward: Reward achieved
            steps_taken: Number of steps for adaptation
        """
        if strategy_name not in self.strategies:
            self.strategies[strategy_name] = StrategyPerformance(strategy_name=strategy_name)
        
        perf = self.strategies[strategy_name]
        
        if success:
            perf.successes += 1
        else:
            perf.failures += 1
        
        # Update running average reward
        total = perf.successes + perf.failures
        perf.avg_reward = (perf.avg_reward * (total - 1) + reward) / total
        
        perf.adaptation_steps.append(steps_taken)
        
        # Keep only recent history
        if len(perf.adaptation_steps) > 100:
            perf.adaptation_steps = perf.adaptation_steps[-100:]
    
    def learn_binding_pattern(
        self,
        concept_type: str,
        role: str,
        filler: str,
        performance: float
    ):
        """
        Learn optimal VSA binding pattern for a concept type.
        
        Args:
            concept_type: Type of concept (e.g., "action", "object")
            role: Role hypervector name
            filler: Filler hypervector name
            performance: Performance metric for this binding
        """
        key = concept_type
        
        if key not in self.optimal_bindings:
            self.optimal_bindings[key] = (role, filler)
        else:
            # Keep if performance is better
            # (would need to track performances in a more complete implementation)
            pass
        
        print(f"[META] Learned binding: {concept_type} -> ({role}, {filler})")
    
    def _compute_gradients(
        self,
        params: Dict[str, np.ndarray],
        dataset: List[Tuple[Any, Any]]
    ) -> Dict[str, np.ndarray]:
        """Compute gradients on dataset (placeholder)."""
        # This is a simplified placeholder
        # Real implementation would depend on the specific neural architecture
        gradients = {}
        for param_name, param_value in params.items():
            # Synthetic gradient for demonstration
            gradients[param_name] = np.random.randn(*param_value.shape) * 0.01
        return gradients
    
    def _compute_meta_gradients(
        self,
        meta_params: Dict[str, np.ndarray],
        adapted_params: Dict[str, np.ndarray],
        support_set: List[Tuple[Any, Any]],
        query_set: List[Tuple[Any, Any]]
    ) -> Dict[str, np.ndarray]:
        """Compute meta-gradients (placeholder)."""
        # Simplified placeholder
        meta_gradients = {}
        for param_name in meta_params:
            meta_gradients[param_name] = np.random.randn(*meta_params[param_name].shape) * 0.001
        return meta_gradients
    
    def _evaluate_loss(
        self,
        params: Dict[str, np.ndarray],
        dataset: List[Tuple[Any, Any]]
    ) -> float:
        """Evaluate loss on dataset (placeholder)."""
        # Simplified placeholder
        return np.random.rand()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        strategy_stats = {}
        for name, perf in self.strategies.items():
            strategy_stats[name] = {
                "success_rate": perf.success_rate,
                "avg_reward": perf.avg_reward,
                "total_uses": perf.successes + perf.failures
            }
        
        return {
            "num_tasks": len(self.task_history),
            "num_strategies": len(self.strategies),
            "strategy_performance": strategy_stats,
            "optimal_bindings": len(self.optimal_bindings),
            "meta_params": len(self.meta_params)
        }


# Example usage
if __name__ == "__main__":
    meta_learner = MetaLearner(inner_lr=0.01, meta_lr=0.001)
    
    # Initialize meta-parameters
    meta_learner.meta_params = {
        "encoder": np.random.randn(100, 10240),
        "decoder": np.random.randn(10240, 100)
    }
    
    # Add some tasks
    for i in range(5):
        support = [(np.random.randn(100), np.random.randn(100)) for _ in range(5)]
        query = [(np.random.randn(100), np.random.randn(100)) for _ in range(10)]
        meta_learner.add_task(f"task_{i}", support, query)
    
    # Meta-training
    task_batch = meta_learner.task_history[:meta_learner.meta_batch_size]
    meta_learner.meta_update(task_batch)
    
    # Strategy selection
    situation = hypervec_rs.HyperVector(42)
    context = {"novelty": 0.8, "confidence": 0.3}
    selected = meta_learner.select_strategy(situation, context)
    print(f"Selected strategy: {selected}")
    
    # Update performance
    meta_learner.update_strategy_performance(selected, success=True, reward=1.0, steps_taken=3)
    
    print(meta_learner.get_statistics())
