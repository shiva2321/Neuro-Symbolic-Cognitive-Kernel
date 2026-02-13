"""
NSCK Curiosity Module
Intrinsic motivation for exploration through novelty detection.

NO NEURAL NETWORKS for curiosity - uses VSA similarity and counting.
"""
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import python.core.vsa.hypervec_shim as hypervec_rs


@dataclass
class ExplorationDecision:
    """Result of curiosity evaluation."""
    should_explore: bool
    novelty_score: float
    learning_progress: float
    reason: str
    testable_hypotheses: List[Tuple[str, str]] = field(default_factory=list)


class CuriosityModule:
    """
    Curiosity-driven exploration using novelty and learning progress.
    
    NO RND (Random Network Distillation) - uses VSA similarity instead.
    
    Novelty = 1 - max_similarity(situation, known_concepts)
    Learning Progress = improvement in success rate over time window
    
    Explore when:
    - High novelty (never seen this before)
    - AND low confidence (don't know what to do)
    - OR stagnant learning (not improving)
    """
    
    def __init__(
        self,
        novelty_threshold: float = 0.7,
        progress_window: int = 100,
        stagnation_threshold: float = 0.01
    ):
        """
        Initialize curiosity module.
        
        Args:
            novelty_threshold: Similarity below this = novel
            progress_window: Number of episodes to measure progress
            stagnation_threshold: Improvement below this = stagnant
        """
        self.novelty_threshold = novelty_threshold
        self.progress_window = progress_window
        self.stagnation_threshold = stagnation_threshold
        
        # Known concept prototypes per task
        self.prototypes: Dict[str, List[hypervec_rs.HyperVector]] = defaultdict(list)
        self.max_prototypes = 100
        
        # Learning progress tracking
        self.success_history: Dict[str, deque] = {}
        
        # Visit counts for state hashing
        self.visit_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # [AGI 6.2] Sub-goal tracking for Planner-Guided ICM
        self.active_subgoals: Dict[str, List[hypervec_rs.HyperVector]] = defaultdict(list)
    
    def compute_novelty(
        self,
        situation_hv: hypervec_rs.HyperVector,
        task_tag: str
    ) -> float:
        """
        Compute novelty of a situation.
        
        Novelty = 1 - max_similarity to known prototypes.
        High novelty = never seen before.
        
        Args:
            situation_hv: Hypervector of current situation
            task_tag: Which task
            
        Returns:
            Novelty score (0.0 = familiar, 1.0 = completely novel)
        """
        prototypes = self.prototypes.get(task_tag, [])
        
        if not prototypes:
            return 1.0  # Everything is novel at first
        
        # Find maximum similarity to any prototype
        max_sim = 0.0
        for proto in prototypes:
            sim = situation_hv.similarity(proto)
            max_sim = max(max_sim, sim)
        
        # Novelty = inverse of familiarity
        return 1.0 - max_sim
    
    def compute_learning_progress(self, task_tag: str) -> float:
        """
        Compute learning progress over recent window.
        
        Progress = success_rate(recent) - success_rate(past)
        Positive = improving, Negative = degrading, Zero = stagnant.
        
        Args:
            task_tag: Which task
            
        Returns:
            Learning progress (-1.0 to 1.0)
        """
        if task_tag not in self.success_history:
            return 0.0
        
        history = self.success_history[task_tag]
        
        if len(history) < self.progress_window:
            return 0.0  # Not enough data
        
        # Split into recent and past halves
        half = len(history) // 2
        recent = list(history)[-half:]
        past = list(history)[-2*half:-half]
        
        recent_rate = sum(recent) / len(recent) if recent else 0
        past_rate = sum(past) / len(past) if past else 0
        
        return recent_rate - past_rate
    
    def update_prototype(
        self,
        situation_hv: hypervec_rs.HyperVector,
        task_tag: str
    ):
        """
        Add situation to prototype memory if novel enough.
        
        Args:
            situation_hv: Hypervector to potentially add
            task_tag: Which task
        """
        protos = self.prototypes[task_tag]
        
        # Check novelty
        novelty = self.compute_novelty(situation_hv, task_tag)
        
        if novelty > self.novelty_threshold:
            # Novel enough to add
            protos.append(situation_hv)
            
            # Prune if too many prototypes
            if len(protos) > self.max_prototypes:
                # Remove oldest prototype
                protos.pop(0)
    
    def record_outcome(self, task_tag: str, success: bool):
        """
        Record success/failure for learning progress tracking.
        
        Args:
            task_tag: Which task
            success: Whether the action led to positive outcome
        """
        if task_tag not in self.success_history:
            self.success_history[task_tag] = deque(maxlen=self.progress_window * 2)
        
        self.success_history[task_tag].append(1.0 if success else 0.0)
    
    def record_visit(self, situation_hv: hypervec_rs.HyperVector, task_tag: str):
        """Record a visit to a state (for count-based exploration bonus)."""
        # Use LSH hash for state identification
        if hasattr(situation_hv, 'lsh_hash'):
            state_hash = situation_hv.lsh_hash(123, 32)
        else:
            state_hash = hash(str(situation_hv)) % (2**32)
        
        self.visit_counts[task_tag][state_hash] += 1
    
    def get_visit_count(self, situation_hv: hypervec_rs.HyperVector, task_tag: str) -> int:
        """Get how many times this state has been visited."""
        if hasattr(situation_hv, 'lsh_hash'):
            state_hash = situation_hv.lsh_hash(123, 32)
        else:
            state_hash = hash(str(situation_hv)) % (2**32)
        
        return self.visit_counts[task_tag].get(state_hash, 0)
    
    def set_subgoals(self, task_tag: str, subgoals: List[hypervec_rs.HyperVector]):
        """[AGI 6.2] Set active sub-goals to bias exploration."""
        self.active_subgoals[task_tag] = subgoals

    def _get_subgoal_bonus(self, situation_hv: hypervec_rs.HyperVector, task_tag: str) -> float:
        """[AGI 6.2] Calculate bonus based on proximity to active sub-goals."""
        subgoals = self.active_subgoals.get(task_tag, [])
        if not subgoals:
            return 0.0
            
        max_sim = 0.0
        for goal in subgoals:
            sim = situation_hv.similarity(goal)
            max_sim = max(max_sim, sim)
            
        # Bonus is significant if we are close to a sub-goal
        return max_sim * 0.5 

    def should_explore(
        self,
        situation_hv: hypervec_rs.HyperVector,
        task_tag: str,
        confidence: float,
        hypotheses: Optional[List[Tuple[str, str]]] = None,
        active_predicates: Optional[List[str]] = None
    ) -> ExplorationDecision:
        """
        Decide whether to explore or exploit.
        
        Now includes 'Causal Curiosity': if a hypothesis (cause->effect) can be tested
        in the current state, increase exploration drive.
        """
        novelty = self.compute_novelty(situation_hv, task_tag)
        
        # [AGI 6.2] Planner-Guided Bonus
        subgoal_bonus = self._get_subgoal_bonus(situation_hv, task_tag)
        novelty = min(1.0, novelty + subgoal_bonus)
        
        # Causal Hypothesis Boost
        causal_boost = 0.0
        testable_hypotheses = []
        if hypotheses and active_predicates:
            preds_set = set(active_predicates)
            for cause, effect in hypotheses:
                # If the cause (predicate or action) is 'testable'
                # For predicates, we check if they are currently TRUE.
                # For actions, they are ALWAYS testable (but we bias selection later).
                if cause.startswith("PRED_"):
                    pure_pred = cause.replace("PRED_", "")
                    if pure_pred in preds_set:
                        causal_boost += 0.2
                        testable_hypotheses.append((cause, effect))
                else:
                    # Action hypothesis - always testable if we reach here
                    testable_hypotheses.append((cause, effect))
        
        novelty = min(1.0, novelty + causal_boost)
        
        progress = self.compute_learning_progress(task_tag)
        visit_count = self.get_visit_count(situation_hv, task_tag)
        
        # Count-based exploration bonus (decays with visits)
        count_bonus = 1.0 / (1.0 + visit_count)
        
        # Decision logic
        explore = False
        reason = "exploit"
        
        # High novelty + low confidence = explore
        if novelty > self.novelty_threshold and confidence < 0.5:
            explore = True
            reason = f"novel_uncertain (nov={novelty:.2f}, conf={confidence:.2f})"
        
        # Stagnant learning = explore to break out
        elif abs(progress) < self.stagnation_threshold and len(self.success_history.get(task_tag, [])) > self.progress_window:
            explore = True
            reason = f"stagnant_learning (progress={progress:.3f})"
        
        # Rarely visited = worth exploring
        elif visit_count < 3 and novelty > 0.5:
            explore = True
            reason = f"rarely_visited (count={visit_count})"
        
        return ExplorationDecision(
            should_explore=explore,
            novelty_score=novelty,
            learning_progress=progress,
            reason=reason,
            testable_hypotheses=testable_hypotheses
        )
    
    def get_exploration_action(
        self,
        available_actions: List[str],
        action_probs: List[float],
        explore_rate: float = 0.3
    ) -> str:
        """
        Select an action with exploration bias.
        
        Uses softmax temperature to encourage trying less probable actions.
        
        Args:
            available_actions: List of possible actions
            action_probs: Current probability for each action
            explore_rate: How much to boost exploration (0=exploit, 1=random)
            
        Returns:
            Selected action
        """
        import random
        import math
        
        if not available_actions:
            return "ACTION_STAY"
        
        # Apply temperature to soften probabilities
        temperature = 1.0 + explore_rate * 2
        
        # Compute softmax with temperature
        max_p = max(action_probs) if action_probs else 1.0
        exp_probs = [math.exp((p - max_p) / temperature) for p in action_probs]
        sum_exp = sum(exp_probs)
        
        soft_probs = [e / sum_exp for e in exp_probs] if sum_exp > 0 else [1/len(available_actions)] * len(available_actions)
        
        # Sample from distribution
        r = random.random()
        cumulative = 0.0
        
        for action, prob in zip(available_actions, soft_probs):
            cumulative += prob
            if r <= cumulative:
                return action
        
        return available_actions[-1]
    
    def get_statistics(self, task_tag: str) -> Dict[str, Any]:
        """Get curiosity statistics for a task."""
        return {
            "prototype_count": len(self.prototypes.get(task_tag, [])),
            "learning_progress": self.compute_learning_progress(task_tag),
            "unique_states_visited": len(self.visit_counts.get(task_tag, {})),
            "success_rate": sum(self.success_history.get(task_tag, [])) / max(1, len(self.success_history.get(task_tag, []))),
        }
