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
    strength: float = 1.0  # 0.0 to 1.0 (Probability of Effect given Cause)
    context: Optional[str] = None  # Task-specific context
    evidence_count: int = 0  # [AGI] Phase 4.1: Number of observations
    
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


class CausalDiscovery:
    """
    [AGI] Phase 4.1: Causal Discovery Module
    
    Learns causal structure from experience streams using
    statistical contingency (Delta-P) and temporal precedence.
    """
    
    def __init__(self):
        # Counts for contingency tables:
        # N(Cause, Effect), N(Cause, ~Effect), N(~Cause, Effect), N(~Cause, ~Effect)
        # Stored as: self.counts[context][(cause, effect)] = { 'c_e': 0, 'c_ne': 0, 'nc_e': 0 }
        self.stats = defaultdict(lambda: defaultdict(lambda: {
            'c_e': 0,   # Cause and Effect
            'c_ne': 0,  # Cause and No Effect
            'nc_e': 0,  # No Cause and Effect (Spontaneous?)
            'nc_ne': 0  # No Cause and No Effect
        }))
        
        # Track all observed events per context to calculate "Not Cause"
        self.observed_causes = defaultdict(set)
        self.observed_effects = defaultdict(set)

    def observe(self, context: str, causes: List[str], effects: List[str]):
        """
        Record a single time-step (transition).
        
        Args:
            context: Task tag (e.g., 'snake')
            causes: List of events at T (Action, State Conditions)
            effects: List of events at T+1 (State Changes, Rewards)
        """
        self.observed_causes[context].update(causes)
        self.observed_effects[context].update(effects)
        
        cause_set = set(causes)
        effect_set = set(effects)
        
        # 1. Update pairings for ALL known candidates (slow but thorough)
        all_known_causes = self.observed_causes[context]
        all_known_effects = self.observed_effects[context]
        
        for c in all_known_causes:
            is_c = c in cause_set
            
            for e in all_known_effects:
                is_e = e in effect_set
                
                # Retrieve stats dict
                s = self.stats[context][(c, e)]
                
                if is_c and is_e:
                    s['c_e'] += 1
                elif is_c and not is_e:
                    s['c_ne'] += 1
                elif not is_c and is_e:
                    s['nc_e'] += 1
                else:
                    s['nc_ne'] += 1

    def induce_graph(self, context: str, min_confidence: float = 0.5, min_evidence: int = 5) -> "CausalGraph":
        """
        Generate a CausalGraph from collected statistics.
        """
        graph = CausalGraph()
        
        for (c, e), s in self.stats[context].items():
            total_c = s['c_e'] + s['c_ne']
            total_nc = s['nc_e'] + s['nc_ne']
            
            if total_c < min_evidence:
                continue
                
            # Calclate P(E|C)
            p_e_given_c = s['c_e'] / total_c
            
            # Calculate P(E|~C)
            p_e_given_nc = 0.0
            if total_nc > 0:
                p_e_given_nc = s['nc_e'] / total_nc
            
            # Delta-P: Causal Power
            delta_p = p_e_given_c - p_e_given_nc
            
            if delta_p > min_confidence:
                graph.add_causes(
                    cause=c,
                    effect=e,
                    strength=delta_p,
                    context=context
                )
                
        return graph

    def get_hypotheses(self, context: str, max_confidence: float = 0.5, min_evidence: int = 2) -> List[Tuple[str, str]]:
        """
        Return a list of (cause, effect) pairs that have some evidence but low confidence.
        These are 'hypotheses' that interventional learning should target.
        """
        hypotheses = []
        for (c, e), s in self.stats[context].items():
            total_c = s['c_e'] + s['c_ne']
            if min_evidence <= total_c:
                # Calculate current Delta-P
                p_e_given_c = s['c_e'] / total_c
                total_nc = s['nc_e'] + s['nc_ne']
                p_e_given_nc = s['nc_e'] / total_nc if total_nc > 0 else 0.0
                delta_p = p_e_given_c - p_e_given_nc
                # If Delta-P is positive, it's a hypothesis
                if delta_p > 0.1:
                    hypotheses.append((c, e))
        return hypotheses


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

    def simulate_counterfactual(
        self,
        state: Dict[str, Any],
        action_taken: str,
        hypothetical_action: str,
        task_tag: str
    ) -> Dict[str, Any]:
        """
        Reason about 'What if' - what would have happened if a different action were taken?
        
        Returns a dict describing the difference in predicted outcome.
        """
        outcome_actual = self.predict_effects(state, action_taken, task_tag)
        outcome_hypothetical = self.predict_effects(state, hypothetical_action, task_tag)
        
        # Find differences
        added = [e for e in outcome_hypothetical if e not in outcome_actual]
        removed = [e for e in outcome_actual if e not in outcome_hypothetical]
        
        return {
            "action_taken": action_taken,
            "hypothetical_action": hypothetical_action,
            "predicted_outcome": outcome_hypothetical,
            "diff_added": added,
            "diff_removed": removed
        }
    
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


def create_maze_causal_graph() -> CausalGraph:
    """Create a pre-populated causal graph for Maze."""
    graph = CausalGraph()
    
    # Movement causes position changes
    graph.add_causes("ACTION_UP", "MOVED_UP", context="maze")
    graph.add_causes("ACTION_DOWN", "MOVED_DOWN", context="maze")
    graph.add_causes("ACTION_LEFT", "MOVED_LEFT", context="maze")
    graph.add_causes("ACTION_RIGHT", "MOVED_RIGHT", context="maze")
    
    # Position changes can cause wall hits
    graph.add_causes("MOVED_UP", "WALL_HIT", strength=0.2, context="maze")
    graph.add_causes("MOVED_DOWN", "WALL_HIT", strength=0.2, context="maze")
    graph.add_causes("MOVED_LEFT", "WALL_HIT", strength=0.2, context="maze")
    graph.add_causes("MOVED_RIGHT", "WALL_HIT", strength=0.2, context="maze")
    
    # Targeting
    graph.add_causes("HEAD_AT_TARGET", "GOAL_REACHED", context="maze")
    graph.add_causes("GOAL_REACHED", "SCORE_UP", context="maze")
    
    # Prevention
    graph.add_prevents("WALL_AT_UP", "ACTION_UP", strength=0.9, context="maze")
    graph.add_prevents("WALL_AT_DOWN", "ACTION_DOWN", strength=0.9, context="maze")
    graph.add_prevents("WALL_AT_LEFT", "ACTION_LEFT", strength=0.9, context="maze")
    graph.add_prevents("WALL_AT_RIGHT", "ACTION_RIGHT", strength=0.9, context="maze")

    return graph


@dataclass
class CausalSchema:
    """An abstracted causal relationship (Theory)."""
    cause_type: str  # e.g., 'MOVEMENT', 'ACTION'
    effect_type: str # e.g., 'COLLISION', 'REWARD'
    template: str    # e.g., "{cause} leads to {effect}"
    confidence: float = 0.0
    examples: List[Tuple[str, str]] = field(default_factory=list)

class TheoryModule:
    """
    Generalizes context-specific causal links into universal theories.
    """
    def __init__(self):
        self.theories: List[CausalSchema] = []
        
        # Primitive abstractions
        self.abstractions = {
            "ACTION_UP": "MOVEMENT",
            "ACTION_DOWN": "MOVEMENT",
            "ACTION_LEFT": "MOVEMENT",
            "ACTION_RIGHT": "MOVEMENT",
            "WALL_COLLISION": "COLLIDER",
            "BODY_COLLISION": "COLLIDER",
            "DEATH": "FAILURE",
            "GAME_OVER": "FAILURE",
            "REWARD_POS": "SUCCESS",
            "REWARD_NEG": "FAILURE"
        }

    def abstract_term(self, term: str) -> str:
        """Map a specific term to an abstract type."""
        for key, val in self.abstractions.items():
            if key in term:
                return val
        return "UNKNOWN"

    def form_theories(self, links: List[CausalLink]) -> List[CausalSchema]:
        """
        Identify recurring patterns across links and form schemas.
        """
        from collections import defaultdict
        patterns = defaultdict(list)
        for link in links:
            c_type = self.abstract_term(link.cause)
            e_type = self.abstract_term(link.effect)
            
            if c_type != "UNKNOWN" and e_type != "UNKNOWN":
                patterns[(c_type, e_type)].append((link.cause, link.effect))

        new_theories = []
        for (c_type, e_type), examples in patterns.items():
            # If we have multiple examples of the same pattern, it's a theory
            if len(set(examples)) >= 2:
                # Check if exists
                existing = next((t for t in self.theories if t.cause_type == c_type and t.effect_type == e_type), None)
                if not existing:
                    theory = CausalSchema(
                        cause_type=c_type,
                        effect_type=e_type,
                        template=f"{c_type} leads to {e_type}",
                        confidence=0.8,
                        examples=list(set(examples))
                    )
                    self.theories.append(theory)
                    new_theories.append(theory)
                    print(f"[THEORY] Formed Theory: {theory.template} (from {len(theory.examples)} examples)")
                else:
                    existing.examples = list(set(existing.examples + examples))
        
        return new_theories

    def predict_from_theory(self, cause: str) -> List[str]:
        """Predict abstract effects based on theories."""
        c_type = self.abstract_term(cause)
        predictions = []
        for theory in self.theories:
            if theory.cause_type == c_type:
                predictions.append(theory.effect_type)
        return predictions
