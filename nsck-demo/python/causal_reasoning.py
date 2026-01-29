"""
NSCK Causal Reasoning Module
Implements causal chains, counterfactual simulation, and "why" inference.

NO NEURAL NETWORKS - uses symbolic graph traversal and simulation.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
from collections import defaultdict


class CausalRelation(Enum):
    """Types of causal relationships."""
    CAUSES = "causes"          # X causes Y
    PREVENTS = "prevents"      # X prevents Y
    ENABLES = "enables"        # X enables Y (necessary but not sufficient)
    REQUIRES = "requires"      # Y requires X


@dataclass
class CausalLink:
    """A single causal relationship."""
    cause: str
    effect: str
    relation: CausalRelation
    strength: float = 1.0  # 0.0 to 1.0
    context: Optional[str] = None  # Task-specific context
    
    def __hash__(self):
        return hash((self.cause, self.effect, self.relation.value))


@dataclass
class CausalChain:
    """A sequence of causal links forming a reasoning chain."""
    links: List[CausalLink]
    start: str
    end: str
    total_strength: float = 1.0
    
    def __len__(self):
        return len(self.links)
    
    def describe(self) -> str:
        """Human-readable description of the chain."""
        if not self.links:
            return f"{self.start} → {self.end} (direct)"
        
        parts = [self.start]
        for link in self.links:
            rel = link.relation.value
            parts.append(f"--[{rel}]→ {link.effect}")
        
        return " ".join(parts)


@dataclass
class CounterfactualResult:
    """Result of counterfactual simulation."""
    query: str  # "What if X?"
    original_outcome: str
    counterfactual_outcome: str
    affected_states: List[str]
    confidence: float
    explanation: str


class CausalGraph:
    """
    Directed graph of causal relationships.
    
    Supports:
    - Forward chaining: "X happened, what follows?"
    - Backward chaining: "Y happened, what caused it?"
    - Counterfactual: "What if X didn't happen?"
    """
    
    def __init__(self):
        """Initialize empty causal graph."""
        # Forward links: cause -> [(effect, link), ...]
        self.forward: Dict[str, List[Tuple[str, CausalLink]]] = defaultdict(list)
        
        # Backward links: effect -> [(cause, link), ...]
        self.backward: Dict[str, List[Tuple[str, CausalLink]]] = defaultdict(list)
        
        # All links for enumeration
        self.all_links: Set[CausalLink] = set()
        
        # Context-specific subgraphs
        self.context_links: Dict[str, Set[CausalLink]] = defaultdict(set)
    
    def add_link(self, link: CausalLink):
        """Add a causal relationship to the graph."""
        self.forward[link.cause].append((link.effect, link))
        self.backward[link.effect].append((link.cause, link))
        self.all_links.add(link)
        
        if link.context:
            self.context_links[link.context].add(link)
    
    def add_causes(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        context: Optional[str] = None
    ):
        """Shorthand for adding a CAUSES relationship."""
        self.add_link(CausalLink(
            cause=cause,
            effect=effect,
            relation=CausalRelation.CAUSES,
            strength=strength,
            context=context
        ))
    
    def add_prevents(
        self,
        preventer: str,
        prevented: str,
        strength: float = 1.0,
        context: Optional[str] = None
    ):
        """Shorthand for adding a PREVENTS relationship."""
        self.add_link(CausalLink(
            cause=preventer,
            effect=prevented,
            relation=CausalRelation.PREVENTS,
            strength=strength,
            context=context
        ))
    
    def forward_chain(
        self,
        start: str,
        max_depth: int = 5,
        context: Optional[str] = None
    ) -> List[CausalChain]:
        """
        Find all effects reachable from a cause.
        
        Args:
            start: Starting cause
            max_depth: Maximum chain length
            context: Optional context filter
            
        Returns:
            List of causal chains
        """
        chains = []
        visited = set()
        
        def dfs(current: str, path: List[CausalLink], strength: float, depth: int):
            if depth > max_depth:
                return
            
            if current in visited:
                return
            visited.add(current)
            
            # Record chain if we've moved from start
            if path:
                chains.append(CausalChain(
                    links=path.copy(),
                    start=start,
                    end=current,
                    total_strength=strength
                ))
            
            # Explore neighbors
            for effect, link in self.forward.get(current, []):
                # Filter by context if specified
                if context and link.context and link.context != context:
                    continue
                
                new_strength = strength * link.strength
                dfs(effect, path + [link], new_strength, depth + 1)
            
            visited.remove(current)
        
        dfs(start, [], 1.0, 0)
        return chains
    
    def backward_chain(
        self,
        end: str,
        max_depth: int = 5,
        context: Optional[str] = None
    ) -> List[CausalChain]:
        """
        Find all causes that lead to an effect.
        
        Args:
            end: Target effect
            max_depth: Maximum chain length
            context: Optional context filter
            
        Returns:
            List of causal chains (reversed)
        """
        chains = []
        visited = set()
        
        def dfs(current: str, path: List[CausalLink], strength: float, depth: int):
            if depth > max_depth:
                return
            
            if current in visited:
                return
            visited.add(current)
            
            if path:
                # Reverse path since we're going backwards
                reversed_path = list(reversed(path))
                chains.append(CausalChain(
                    links=reversed_path,
                    start=current,
                    end=end,
                    total_strength=strength
                ))
            
            for cause, link in self.backward.get(current, []):
                if context and link.context and link.context != context:
                    continue
                
                new_strength = strength * link.strength
                dfs(cause, path + [link], new_strength, depth + 1)
            
            visited.remove(current)
        
        dfs(end, [], 1.0, 0)
        return chains
    
    def find_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
        context: Optional[str] = None
    ) -> Optional[CausalChain]:
        """Find a causal path between two nodes."""
        chains = self.forward_chain(start, max_depth, context)
        for chain in chains:
            if chain.end == end:
                return chain
        return None
    
    def get_immediate_effects(self, cause: str) -> List[str]:
        """Get direct effects of a cause."""
        return [effect for effect, _ in self.forward.get(cause, [])]
    
    def get_immediate_causes(self, effect: str) -> List[str]:
        """Get direct causes of an effect."""
        return [cause for cause, _ in self.backward.get(effect, [])]


class CausalReasoner:
    """
    High-level causal reasoning engine.
    
    Combines causal graph with state simulation for:
    - Forward prediction
    - Backward explanation
    - Counterfactual analysis
    """
    
    def __init__(
        self,
        graph: CausalGraph,
        simulator: Optional[Callable[[Dict, str], Tuple[Dict, bool]]] = None
    ):
        """
        Initialize causal reasoner.
        
        Args:
            graph: Causal knowledge graph
            simulator: Optional (state, action) -> (next_state, terminal) function
        """
        self.graph = graph
        self.simulator = simulator
    
    def predict_effects(
        self,
        state: Dict[str, Any],
        action: str,
        task_tag: str
    ) -> List[str]:
        """
        Predict effects of an action in current state.
        
        Args:
            state: Current game state
            action: Proposed action
            task_tag: Task context
            
        Returns:
            List of predicted effects
        """
        effects = []
        
        # Get direct effects from causal graph
        chain_effects = self.graph.forward_chain(action, max_depth=3, context=task_tag)
        
        for chain in chain_effects:
            effects.append(chain.end)
        
        # If simulator available, use it for ground truth
        if self.simulator:
            try:
                next_state, terminal = self.simulator(state, action.replace("ACTION_", ""))
                
                if terminal:
                    effects.append("GAME_OVER")
                
                # Detect state changes
                for key in next_state:
                    if key in state and next_state[key] != state[key]:
                        effects.append(f"CHANGED_{key.upper()}")
            except Exception:
                pass
        
        return list(set(effects))
    
    def explain_why(
        self,
        effect: str,
        state: Dict[str, Any],
        task_tag: str,
        recent_actions: Optional[List[str]] = None
    ) -> List[CausalChain]:
        """
        Explain why an effect occurred.
        
        Args:
            effect: The effect to explain
            state: Current state
            task_tag: Task context
            recent_actions: Optional list of recent actions
            
        Returns:
            List of possible causal chains explaining the effect
        """
        chains = self.graph.backward_chain(effect, max_depth=4, context=task_tag)
        
        # Filter chains to those consistent with recent actions
        if recent_actions:
            action_set = set(recent_actions)
            relevant = []
            for chain in chains:
                if chain.start in action_set or any(
                    link.cause in action_set for link in chain.links
                ):
                    relevant.append(chain)
            return relevant
        
        return chains
    
    def counterfactual(
        self,
        actual_action: str,
        alternative_action: str,
        state: Dict[str, Any],
        task_tag: str
    ) -> CounterfactualResult:
        """
        Answer "What if I did Y instead of X?"
        
        Args:
            actual_action: What was actually done
            alternative_action: What could have been done
            state: State when action was taken
            task_tag: Task context
            
        Returns:
            CounterfactualResult with comparison
        """
        original_outcome = "unknown"
        counterfactual_outcome = "unknown"
        affected = []
        
        if self.simulator:
            try:
                # Simulate actual action
                actual_next, actual_terminal = self.simulator(
                    state, actual_action.replace("ACTION_", "")
                )
                original_outcome = "TERMINAL" if actual_terminal else "CONTINUE"
                
                # Simulate alternative action
                alt_next, alt_terminal = self.simulator(
                    state, alternative_action.replace("ACTION_", "")
                )
                counterfactual_outcome = "TERMINAL" if alt_terminal else "CONTINUE"
                
                # Find differences
                for key in set(actual_next.keys()) | set(alt_next.keys()):
                    if actual_next.get(key) != alt_next.get(key):
                        affected.append(key)
                        
            except Exception as e:
                pass
        
        # Generate explanation
        if original_outcome == counterfactual_outcome:
            explanation = f"No difference: both {actual_action} and {alternative_action} lead to {original_outcome}"
        else:
            explanation = f"Different outcomes: {actual_action}→{original_outcome}, {alternative_action}→{counterfactual_outcome}"
        
        return CounterfactualResult(
            query=f"What if {alternative_action} instead of {actual_action}?",
            original_outcome=original_outcome,
            counterfactual_outcome=counterfactual_outcome,
            affected_states=affected,
            confidence=0.8 if self.simulator else 0.3,
            explanation=explanation
        )
    
    def why_not(
        self,
        rejected_action: str,
        chosen_action: str,
        state: Dict[str, Any],
        task_tag: str
    ) -> str:
        """
        Explain why an action was rejected.
        
        Args:
            rejected_action: Action that wasn't taken
            chosen_action: Action that was taken
            state: Current state
            task_tag: Task context
            
        Returns:
            Human-readable explanation
        """
        # Check if rejected action has negative consequences
        rejected_effects = self.predict_effects(state, rejected_action, task_tag)
        chosen_effects = self.predict_effects(state, chosen_action, task_tag)
        
        reasons = []
        
        if "GAME_OVER" in rejected_effects:
            reasons.append(f"{rejected_action} would cause GAME_OVER")
        
        if "GAME_OVER" not in chosen_effects and "GAME_OVER" in rejected_effects:
            reasons.append(f"{chosen_action} is safer")
        
        # Check causal chains
        bad_chains = self.graph.forward_chain(rejected_action, max_depth=2, context=task_tag)
        for chain in bad_chains:
            if "DEATH" in chain.end or "FAIL" in chain.end or "COLLISION" in chain.end:
                reasons.append(f"{rejected_action} leads to {chain.end}")
        
        if not reasons:
            return f"No specific reason found for preferring {chosen_action} over {rejected_action}"
        
        return "; ".join(reasons)


def create_snake_causal_graph() -> CausalGraph:
    """Create a pre-populated causal graph for Snake."""
    graph = CausalGraph()
    
    # Movement causes position changes
    graph.add_causes("ACTION_UP", "HEAD_MOVES_UP", context="snake")
    graph.add_causes("ACTION_DOWN", "HEAD_MOVES_DOWN", context="snake")
    graph.add_causes("ACTION_LEFT", "HEAD_MOVES_LEFT", context="snake")
    graph.add_causes("ACTION_RIGHT", "HEAD_MOVES_RIGHT", context="snake")
    
    # Position changes can cause collisions
    graph.add_causes("HEAD_MOVES_UP", "WALL_COLLISION", strength=0.3, context="snake")
    graph.add_causes("HEAD_MOVES_DOWN", "WALL_COLLISION", strength=0.3, context="snake")
    graph.add_causes("HEAD_MOVES_LEFT", "WALL_COLLISION", strength=0.3, context="snake")
    graph.add_causes("HEAD_MOVES_RIGHT", "WALL_COLLISION", strength=0.3, context="snake")
    
    # Body collision
    graph.add_causes("HEAD_MEETS_BODY", "BODY_COLLISION", context="snake")
    graph.add_causes("BODY_COLLISION", "DEATH", context="snake")
    graph.add_causes("WALL_COLLISION", "DEATH", strength=0.9, context="snake")
    
    # Food
    graph.add_causes("HEAD_MEETS_FOOD", "EAT_FOOD", context="snake")
    graph.add_causes("EAT_FOOD", "GROW", context="snake")
    graph.add_causes("EAT_FOOD", "SCORE_UP", context="snake")
    
    # Prevention relationships
    graph.add_prevents("DANGER_UP", "ACTION_UP", strength=0.9, context="snake")
    graph.add_prevents("DANGER_DOWN", "ACTION_DOWN", strength=0.9, context="snake")
    graph.add_prevents("DANGER_LEFT", "ACTION_LEFT", strength=0.9, context="snake")
    graph.add_prevents("DANGER_RIGHT", "ACTION_RIGHT", strength=0.9, context="snake")
    
    return graph


def create_pong_causal_graph() -> CausalGraph:
    """Create a pre-populated causal graph for Pong."""
    graph = CausalGraph()
    
    # Paddle movement
    graph.add_causes("ACTION_UP", "PADDLE_MOVES_UP", context="pong")
    graph.add_causes("ACTION_DOWN", "PADDLE_MOVES_DOWN", context="pong")
    
    # Ball interaction
    graph.add_causes("PADDLE_ALIGNED", "BALL_HIT", context="pong")
    graph.add_causes("BALL_HIT", "BALL_RETURNS", context="pong")
    graph.add_causes("BALL_RETURNS", "SCORE_UP", strength=0.5, context="pong")
    
    # Miss
    graph.add_causes("PADDLE_MISALIGNED", "BALL_MISS", context="pong")
    graph.add_causes("BALL_MISS", "POINT_LOST", context="pong")
    
    return graph
