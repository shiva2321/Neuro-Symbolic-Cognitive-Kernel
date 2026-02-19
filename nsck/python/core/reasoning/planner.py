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
    
    Uses A* search to find a sequence of actions 
    that transforms start_state -> goal_state.
    
    Supports:
    1. Hardcoded operators (bootstrap)
    2. CausalGraph-derived operators (learned from experience)
    3. Rule-derived operators (from RuleLearner)
    
    Operators are automatically extracted from the CausalGraph when
    available, so the planner can use *learned* transition dynamics
    rather than only hardcoded actions.
    """
    
    def __init__(self, operators: List[any] = None):
        """
        operators: List of Rule objects (condition, consequence)
        """
        self.operators = operators or []
        self.causal_reasoner = None
        self._learned_ops: Dict[str, Dict[str, Set[str]]] = {}  # action -> {"add": set, "del": set, "pre": set}
    
    def set_operators(self, rules: List[any]):
        """Update available operators from learned rules."""
        self.operators = rules
        
    def learn_operators_from_graph(self, graph, context: str = None):
        """Extract STRIPS-style operators from a CausalGraph.
        
        For each action node in the graph, compute:
        - preconditions: states that must be true for the action to have its effects
        - add effects: nodes reachable via forward chaining (causes/enables)
        - delete effects: nodes that are 'prevented' by the action
        
        This eliminates the need for hardcoded action operators.
        """
        self._learned_ops.clear()
        action_nodes = [n for n in graph.forward if n.startswith("ACTION_")]
        
        for action in action_nodes:
            add_effects = set()
            del_effects = set()
            preconditions = set()
            
            # Forward: direct effects
            for effect, link in graph.forward.get(action, []):
                if context and link.context and link.context != context:
                    continue
                if link.relation.value == "prevents":
                    del_effects.add(effect)
                else:
                    add_effects.add(effect)
            
            # Backward: extract preconditions
            # Nodes that ENABLE or are REQUIRED by this action are preconditions
            for preventer, plink in graph.backward.get(action, []):
                if context and plink.context and plink.context != context:
                    continue
                if plink.relation.value in ("enables", "requires"):
                    preconditions.add(preventer)
                elif plink.relation.value == "prevents":
                    # The preventer prevents this action — it must be absent
                    # We don't model negative preconditions in simple STRIPS,
                    # but we can add the prevented state as a delete effect
                    pass
            
            if add_effects or del_effects:
                self._learned_ops[action] = {
                    "add": add_effects,
                    "del": del_effects,
                    "pre": preconditions,
                }
        
        if self._learned_ops:
            print(f"[PLANNER] Learned {len(self._learned_ops)} operators from CausalGraph")
        
    def _heuristic(self, state: FrozenSet[str], goal: Set[str]) -> int:
        """A* heuristic: number of unsatisfied goal predicates."""
        return len(goal - state)

    def plan(self, initial_state: Set[str], goal: Set[str], max_depth: int = 10) -> Optional[List[str]]:
        """
        Find a sequence of actions to reach goal state using A* search.
        
        Args:
            initial_state: Set of active predicates (e.g., {"AT_5_5", "FOOD_AT_5_6"})
            goal: Set of target predicates (e.g., {"AT_5_6"})
            max_depth: limit search depth
            
        Returns:
            List of action strings or None
        """
        start_frozen = frozenset(initial_state)
        h = self._heuristic(start_frozen, goal)
        start_node = PlanNode(0 + h, start_frozen, [])
        queue = [start_node]
        # Track best g-cost per visited state
        best_cost: Dict[FrozenSet[str], int] = {start_frozen: 0}
        
        while queue:
            current = heapq.heappop(queue)
            g_cost = len(current.history)
            
            # GOAL CHECK: Is goal a subset of current state?
            if goal.issubset(current.state):
                return current.history
            
            if g_cost >= max_depth:
                continue
                
            # EXPAND: Apply all applicable operators
            if self._learned_ops:
                possible_actions = list(self._learned_ops.keys())
            else:
                possible_actions = ["ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"]
            
            for action in possible_actions:
                # Check preconditions
                if not self._preconditions_met(current.state, action):
                    continue
                
                # Predict effects
                add_effects, del_effects = self._predict(current.state, action)
                
                # Apply effects
                next_state_set = set(current.state)
                next_state_set.difference_update(del_effects)
                next_state_set.update(add_effects)
                
                next_frozen = frozenset(next_state_set)
                new_g = g_cost + 1
                
                # Only expand if this path is cheaper than any we've found
                if next_frozen not in best_cost or new_g < best_cost[next_frozen]:
                    best_cost[next_frozen] = new_g
                    h = self._heuristic(next_frozen, goal)
                    new_node = PlanNode(
                        cost=new_g + h,
                        state=next_frozen,
                        history=current.history + [action]
                    )
                    heapq.heappush(queue, new_node)
                    
        return None

    def _preconditions_met(self, state: FrozenSet[str], action: str) -> bool:
        """Check if all learned preconditions for action are satisfied."""
        if action in self._learned_ops:
            preconditions = self._learned_ops[action].get("pre", set())
            return preconditions.issubset(state)
        return True  # No known preconditions — assume applicable

    def _predict(self, state: FrozenSet[str], action: str) -> Tuple[Set[str], Set[str]]:
        """
        Predict (add_effects, del_effects) of action given state.
        
        Priority:
        1. Learned operators from CausalGraph (most accurate)
        2. CausalReasoner direct query (backward compat)
        3. Empty (no prediction)
        """
        add_effects = set()
        del_effects = set()
        
        # 1. Check learned operators first
        if action in self._learned_ops:
            add_effects = set(self._learned_ops[action]["add"])
            del_effects = set(self._learned_ops[action]["del"])
            return add_effects, del_effects
        
        # 2. Fallback to CausalReasoner
        if hasattr(self, 'causal_reasoner') and self.causal_reasoner:
             add_effects = set(self.causal_reasoner.graph.get_immediate_effects(action))
             
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
        
        Default decomposition: separate each unsatisfied goal predicate
        into its own subgoal, ordered by causal dependency if the causal
        graph is available. Falls back to arbitrary ordering.
        
        Args:
            state: Current state
            goal: Full goal set
            
        Returns:
            List of subgoal sets to achieve sequentially, or empty list
            if goal has only one predicate (not decomposable).
        """
        unsatisfied = goal - state
        if len(unsatisfied) <= 1:
            return []  # Not decomposable
        
        # If we have a causal graph, order subgoals so prerequisites come first
        if hasattr(self, 'causal_reasoner') and self.causal_reasoner:
            ordered = []
            remaining = set(unsatisfied)
            
            # Topological-ish ordering: put predicates that enable others first
            while remaining:
                # Find predicates that don't depend on any other remaining predicate
                independent = set()
                for pred in remaining:
                    deps = set()
                    for other in remaining:
                        if other == pred:
                            continue
                        # Check if pred requires other (via causal backward links)
                        backward = self.causal_reasoner.graph.backward.get(pred, [])
                        for cause, link in backward:
                            if cause == other and link.relation.value in ("enables", "requires"):
                                deps.add(other)
                    if not deps:
                        independent.add(pred)
                
                if not independent:
                    # Circular or no dependency info — just take any
                    independent = {remaining.pop()}
                
                ordered.append(independent)
                remaining -= independent
            
            return ordered
        
        # No causal graph — just treat each predicate as a sequential subgoal
        return [{pred} for pred in unsatisfied]

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
            # Recursive call with decreased depth to prevent infinite recursion
            plan_segment = self.plan_hierarchical(current_state, subgoal_set, max_depth=max_depth // 2)
            
            if not plan_segment:
                return None  # Failed to satisfy a subgoal
            
            full_plan.extend(plan_segment)
            current_state = self.simulate_sequence(current_state, plan_segment)
            
        return full_plan
