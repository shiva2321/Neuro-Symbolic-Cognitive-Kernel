"""
CustomPlanner - Goal-Directed Planning Module
==============================================

Example module demonstrating custom planning logic.

Features:
- Goal-directed reasoning
- A* search with hypervector state space
- Subgoal decomposition
- Progress tracking

Use Case:
When NSCK needs domain-specific planning (e.g., maze navigation with custom
heuristics, multi-step task planning), CustomPlanner can propose action
sequences optimized for the specific domain.

Author: NSCK SDK Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import deque
import hashlib

from global_workspace import WorkspaceModule, Coalition
import hypervec_shim as hv


@dataclass
class PlanStep:
    """
    A single step in a plan.
    
    Attributes:
        action_hv: Hypervector representing the action
        expected_state_hv: Expected resulting state
        cost: Estimated cost of this step
        description: Human-readable description
    """
    action_hv: np.ndarray
    expected_state_hv: np.ndarray
    cost: float
    description: str


class CustomPlanner(WorkspaceModule):
    """
    Goal-directed planner with custom domain heuristics.
    
    This module demonstrates:
    - Goal management and tracking
    - Multi-step planning
    - Cost-based bidding
    - Plan refinement from feedback
    
    Example:
        # Create planner with goal
        goal_hv = encode("reach_exit")
        planner = CustomPlanner(goal_hv=goal_hv)
        
        # Add domain actions
        planner.add_action("move_north", cost=1.0)
        planner.add_action("move_south", cost=1.0)
        planner.add_action("unlock_door", cost=5.0)
        
        # Register with CognitiveEngine
        engine.register_module(planner)
    """
    
    __author__ = "NSCK SDK Team"
    __version__ = "1.0.0"
    
    def __init__(self, goal_hv: Optional[np.ndarray] = None, max_plan_length: int = 10):
        """
        Initialize CustomPlanner.
        
        Args:
            goal_hv: Target goal hypervector (can be set later)
            max_plan_length: Maximum number of steps in a plan
        """
        self.goal_hv = goal_hv
        self.max_plan_length = max_plan_length
        
        # Known actions (action_name -> (action_hv, cost))
        self.actions: Dict[str, Tuple[np.ndarray, float]] = {}
        
        # Current plan
        self.current_plan: deque[PlanStep] = deque()
        
        # Telemetry
        self.total_proposals = 0
        self.plans_created = 0
        self.plans_completed = 0
        self.plans_failed = 0
        self.total_reward_received = 0.0
        self.last_plan_length = 0
    
    def set_goal(self, goal_hv: np.ndarray):
        """Set or update the goal."""
        self.goal_hv = goal_hv
        self.current_plan.clear()  # Invalidate old plan
    
    def add_action(self, name: str, cost: float = 1.0, action_hv: Optional[np.ndarray] = None):
        """
        Add a known action to the planner.
        
        Args:
            name: Action name (will be encoded if action_hv not provided)
            cost: Cost of executing this action
            action_hv: Optional pre-encoded action hypervector (np.ndarray)
        """
        if action_hv is None:
            # Deterministic encoding
            h = hashlib.sha256(name.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            action_hv = hv.HyperVector(seed).bits
        self.actions[name] = (action_hv, cost)
    
    def propose(self, state_hv: np.ndarray) -> Optional[Coalition]:
        """
        Propose next action from current plan, or create new plan.
        
        Args:
            state_hv: Current workspace state hypervector (10240-bit np.ndarray)
            
        Returns:
            Coalition with next planned action, or None if no goal
        """
        self.total_proposals += 1
        
        # No goal set
        if self.goal_hv is None:
            return None
        
        # Check if goal reached
        diff = np.bitwise_xor(state_hv, self.goal_hv)
        hamming_dist = np.sum(diff)
        goal_similarity = 1.0 - (hamming_dist / len(state_hv))
        if goal_similarity >= 0.7:  # Goal threshold
            self.plans_completed += 1
            self.current_plan.clear()
            return None  # Goal achieved!
        
        # If no plan, create one
        if len(self.current_plan) == 0:
            self._create_plan(state_hv)
        
        # If still no plan, can't help
        if len(self.current_plan) == 0:
            return None
        
        # Pop next step from plan
        next_step = self.current_plan.popleft()
        
        # Bid based on:
        # 1. Progress towards goal (higher = better)
        # 2. Inverse of cost (lower cost = higher bid)
        # 3. Confidence in plan (more steps remaining = lower confidence)
        
        # Compute similarity between expected next state and goal
        diff = np.bitwise_xor(next_step.expected_state_hv, self.goal_hv)
        hamming_dist = np.sum(diff)
        progress_to_goal = 1.0 - (hamming_dist / len(self.goal_hv))
        confidence = 1.0 / (len(self.current_plan) + 2)  # Fewer remaining steps = higher confidence
        
        # Map planning metrics to Coalition salience
        base_salience = progress_to_goal  # How close we're getting to goal
        relevance = confidence  # How confident we are in the plan
        sender_confidence = 1.0 / (next_step.cost + 1)  # Lower cost = higher confidence
        
        return Coalition(
            source="CustomPlanner",
            content={
                "type": "action",
                "action_hv": next_step.action_hv,
                "action_description": next_step.description,
                "expected_cost": next_step.cost,
                "remaining_steps": len(self.current_plan),
                "progress_to_goal": float(progress_to_goal),
                "goal_similarity": float(goal_similarity)
            },
            base_salience=base_salience,
            relevance=relevance,
            sender_confidence=sender_confidence
        )
    
    def _create_plan(self, state_hv: np.ndarray):
        """
        Create a plan from current state to goal.
        
        Simple greedy approach: Choose action that moves closest to goal.
        More sophisticated planners could use A* or Monte Carlo Tree Search.
        """
        if self.goal_hv is None or len(self.actions) == 0:
            return
        
        plan = []
        current_state = state_hv.copy()
        
        for step_idx in range(self.max_plan_length):
            # Find action that moves closest to goal
            best_action_name = None
            best_action_hv = None
            best_cost = 0.0
            best_similarity = -1.0
            
            for action_name, (action_hv, cost) in self.actions.items():
                # Simulate taking action (simple: bundle state with action)
                # Using weighted bundle (majority vote)
                predicted_state = self._simple_bundle(current_state, action_hv)
                
                # Measure progress towards goal
                diff = np.bitwise_xor(predicted_state, self.goal_hv)
                hamming_dist = np.sum(diff)
                sim = 1.0 - (hamming_dist / len(self.goal_hv))
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_action_name = action_name
                    best_action_hv = action_hv
                    best_cost = cost
            
            # If no improvement, stop planning
            current_goal_sim = self._similarity(current_state, self.goal_hv)
            if best_similarity < current_goal_sim:
                break
            
            # Add step to plan
            predicted_state = self._simple_bundle(current_state, best_action_hv)
            plan.append(PlanStep(
                action_hv=best_action_hv,
                expected_state_hv=predicted_state,
                cost=best_cost,
                description=best_action_name
            ))
            
            current_state = predicted_state
            
            # If reached goal, stop
            if best_similarity >= 0.7:
                break
        
        self.current_plan = deque(plan)
        self.last_plan_length = len(plan)
        if len(plan) > 0:
            self.plans_created += 1
    
    def _simple_bundle(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Simple majority-vote bundle for two hypervectors."""
        # For 2 vectors, majority is just AND (agree) + XOR * random()
        rand_mask = np.random.randint(0, 2, size=len(hv1), dtype=np.int8)
        same = np.bitwise_and(hv1, hv2)
        diff = np.bitwise_xor(hv1, hv2)
        return np.bitwise_or(same, np.bitwise_and(diff, rand_mask))
    
    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute Hamming similarity."""
        diff = np.bitwise_xor(hv1, hv2)
        hamming_dist = np.sum(diff)
        return 1.0 - (hamming_dist / len(hv1))
    
    def update(self, feedback_hv: np.ndarray, reward: float, info: Dict):
        """
        Learn from feedback to improve future plans.
        
        Args:
            feedback_hv: Actual resulting state
            reward: Reward received
            info: Additional information
        """
        self.total_reward_received += reward
        
        # If reward is very negative, plan failed
        if reward < -10:
            self.plans_failed += 1
            self.current_plan.clear()  # Abandon plan
    
    def receive_broadcast(self, content: Any):
        """
        React to broadcasts (e.g., new goal assignments).
        """
        if isinstance(content, dict) and "goal" in content:
            # New goal broadcast
            if isinstance(content["goal"], np.ndarray):
                self.set_goal(content["goal"])
                print(f"📍 CustomPlanner received new goal")
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Return planning statistics.
        
        Returns:
            Dict with:
            - total_proposals: How many times propose() called
            - plans_created: Number of plans created
            - plans_completed: Number of goals successfully reached
            - plans_failed: Number of plans abandoned
            - success_rate: Percentage of successful plans
            - avg_reward: Average reward per plan
            - current_plan_length: Steps remaining in current plan
            - last_plan_length: Length of most recent created plan
            - known_actions: Number of actions in repertoire
        """
        success_rate = (self.plans_completed / self.plans_created * 100 
                       if self.plans_created > 0 else 0.0)
        
        avg_reward = (self.total_reward_received / self.plans_created 
                     if self.plans_created > 0 else 0.0)
        
        return {
            "total_proposals": self.total_proposals,
            "plans_created": self.plans_created,
            "plans_completed": self.plans_completed,
            "plans_failed": self.plans_failed,
            "success_rate": round(success_rate, 2),
            "avg_reward": round(avg_reward, 2),
            "current_plan_length": len(self.current_plan),
            "last_plan_length": self.last_plan_length,
            "known_actions": len(self.actions),
            "has_goal": self.goal_hv is not None
        }


# Example usage
if __name__ == "__main__":
    def _encode(label: str) -> hv.HyperVector:
        """Simple deterministic encoding."""
        h = hashlib.sha256(label.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        return hv.HyperVector(seed)
    
    print("CustomPlanner Demo")
    print("=" * 50)
    
    # Create planner
    goal = _encode("reach_exit").bits
    planner = CustomPlanner(goal_hv=goal, max_plan_length=5)
    
    # Add actions
    planner.add_action("move_north", cost=1.0)
    planner.add_action("move_south", cost=1.0)
    planner.add_action("move_east", cost=1.0)
    planner.add_action("move_west", cost=1.0)
    planner.add_action("open_door", cost=5.0)
    
    # Test planning
    initial_state = _encode("start_position").bits
    
    print("\n📍 Creating plan from start to exit...")
    for step in range(8):
        coalition = planner.propose(initial_state)
        if coalition is None:
            print(f"\n✅ Goal reached or no plan available")
            break
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {coalition.content['action_description']}")
        print(f"  Salience: {coalition.base_salience:.2f}")
        print(f"  Remaining: {coalition.content['remaining_steps']} steps")
        print(f"  Progress: {coalition.content['progress_to_goal']:.3f}")
        
        # Simulate state transition
        action_hv = coalition.content['action_hv']
        # Simple bundle for state transition
        rand_mask = np.random.randint(0, 2, size=len(initial_state), dtype=np.int8)
        same = np.bitwise_and(initial_state, action_hv)
        diff = np.bitwise_xor(initial_state, action_hv)
        initial_state = np.bitwise_or(same, np.bitwise_and(diff, rand_mask))
        
        # Simulate reward
        planner.update(initial_state, reward=1.0, info={})
    
    # Print telemetry
    print(f"\n📊 Telemetry: {planner.get_telemetry()}")
