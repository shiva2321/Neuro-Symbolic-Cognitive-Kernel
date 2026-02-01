"""
NSCK Planning Module (Phase 2)
Implements STRIPS-style planning using the Forward Chaining engine or explicit operators.
"""
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, FrozenSet
import heapq

@dataclass
class PlanStep:
    action: str
    state_snapshot: FrozenSet[str]

@dataclass(order=True)
class PlanNode:
    cost: int
    state: FrozenSet[str] = field(compare=False)
    history: List[str] = field(compare=False)

class STRIPSPlanner:
    """
    GOAL-DIRECTED REASONING
    
    Uses Breadth-First Search (or A*) to find a sequence of actions 
    that transforms start_state -> goal_state.
    
    Unlike standard STRIPS which has fixed operators, this planner 
    uses the LEARNED RULES from RuleLearner/BrainFusion as dynamic operators.
    """
    
    def __init__(self, operators: List[any] = None):
        """
        operators: List of Rule objects (condition, consequence)
        """
        self.operators = operators or []
    
    def set_operators(self, rules: List[any]):
        """Update available operators from learned rules."""
        self.operators = rules
        
    def plan(self, initial_state: Set[str], goal: Set[str], max_depth: int = 10) -> Optional[List[str]]:
        """
        Find a sequence of actions to reach goal state.
        
        Args:
            initial_state: Set of active predicates (e.g., {"AT_5_5", "FOOD_AT_5_6"})
            goal: Set of target predicates (e.g., {"AT_5_6"})
            max_depth: limit search depth
            
        Returns:
            List of action strings or None
        """
        # Priority Queue for A* (Cost = depth + heuristic)
        # For now, just BFS (Cost = depth)
        start_node = PlanNode(0, frozenset(initial_state), [])
        queue = [start_node]
        visited = {frozenset(initial_state)}
        
        while queue:
            current = heapq.heappop(queue)
            
            # GOAL CHECK: Is goal a subset of current state?
            if goal.issubset(current.state):
                return current.history
            
            if len(current.history) >= max_depth:
                continue
                
            # EXPAND: Apply all applicable rules
            # In STRIPS, Action A has Preconditions P, Add Effects A, Del Effects D.
            # Our Rules are: Condition C -> Consequence E.
            # We assume Rule: C -> E implies "If C, Action 'E' leads to effect E?"
            # Wait, our rules are: Condition -> Action OR Condition -> Fact.
            #
            # If Rule is FACT->FACT (Inference), it runs automatically (Forward Chaining).
            # If Rule is FACT->ACTION (Policy), it tells us "In this state, do Action".
            # But Planning requires: "If I do Action A, State S becomes S'".
            #
            # The current RuleLearner learns Policy (State -> Action). 
            # It does NOT explicitly learn Transition Dynamics (State + Action -> Effect).
            #
            # CausalGraph (causal_reasoning.py) DOES learn transitions!
            # It has: Cause (Action) -> Effect (State Change).
            #
            # So the Planner should use the CausalGraph (Transition Model), 
            # NOT the Policy Rules.
            
            # REFACTORING ON THE FLY:
            possible_actions = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]
            
            for action in possible_actions:
                # 1. Predict Next State (Add and Delete effects)
                add_effects, del_effects = self._predict(current.state, action)
                
                # Apply effects
                next_state_set = set(current.state)
                
                # Apply Deletes first
                next_state_set.difference_update(del_effects)
                
                # Apply Adds
                next_state_set.update(add_effects)
                
                next_frozen = frozenset(next_state_set)
                
                if next_frozen not in visited:
                    visited.add(next_frozen)
                    new_node = PlanNode(
                        cost=current.cost + 1,
                        state=next_frozen,
                        history=current.history + [action]
                    )
                    heapq.heappush(queue, new_node)
                    
        return None

    def _predict(self, state: FrozenSet[str], action: str) -> Tuple[Set[str], Set[str]]:
        """
        Predict (add_effects, del_effects) of action given state.
        """
        add_effects = set()
        del_effects = set()
        
        if hasattr(self, 'causal_reasoner') and self.causal_reasoner:
             # Standard CausalGraph only gives additions
             add_effects = set(self.causal_reasoner.graph.get_immediate_effects(action))
             # TODO: Extract 'prevents' from graph for del_effects
             
        return add_effects, del_effects
    
    def set_reasoner(self, reasoner):
        self.causal_reasoner = reasoner

    def simulate_sequence(self, state: FrozenSet[str], actions: List[str]) -> FrozenSet[str]:
        """Apply a sequence of actions to state to get resulting state."""
        current = set(state)
        for action in actions:
            add, delete = self._predict(frozenset(current), action)
            current.difference_update(delete)
            current.update(add)
        return frozenset(current)

    def decompose_goal(self, state: FrozenSet[str], goal: Set[str]) -> List[Set[str]]:
        """
        Divide goal into subgoals if possible.
        Base implementation returns empty list.
        Override in subclasses (e.g. GridPlanner).
        """
        return []

    def plan_hierarchical(self, initial_state: Set[str], goal: Set[str], max_depth: int = 50) -> Optional[List[str]]:
        """
        Hierarchical planning with goal decomposition.
        """
        # 1. Try direct plan first (quick check)
        # Use a smaller depth for the direct check to avoid deep BFS if unnecessary
        direct_limit = min(10, max_depth)
        direct_plan = self.plan(initial_state, goal, max_depth=direct_limit)
        if direct_plan:
            return direct_plan

        # 2. Decompose
        # Note: decompose_goal usually requires knowing 'state' to calculate midpoints
        subgoals = self.decompose_goal(frozenset(initial_state), goal)
        if not subgoals:
            return None

        # 3. Plan for each subgoal sequentially
        full_plan = []
        current_state = frozenset(initial_state)
        
        for subgoal_set in subgoals:
            # Recursive call allows multi-level decomposition
            # We treat subgoal_set as the target
            # Note: We decrease max_depth to prevent infinite recursion
            plan_segment = self.plan_hierarchical(current_state, subgoal_set, max_depth=max_depth)
            
            if not plan_segment:
                return None  # Failed to satisfy a subgoal
            
            full_plan.extend(plan_segment)
            current_state = self.simulate_sequence(current_state, plan_segment)
            
        return full_plan
