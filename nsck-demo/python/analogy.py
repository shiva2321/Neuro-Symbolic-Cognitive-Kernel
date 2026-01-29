"""
NSCK Analogy Module
Cross-task transfer via structural alignment of concepts.

NO NEURAL EMBEDDINGS - uses VSA binding and symbolic mapping.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import hypervec_rs


@dataclass
class ConceptMapping:
    """A mapping between concepts in two domains."""
    source_domain: str
    target_domain: str
    source_concept: str
    target_concept: str
    similarity: float
    relation_type: str  # "structural", "functional", "perceptual"


@dataclass
class Analogy:
    """A complete analogical mapping between domains."""
    source_domain: str
    target_domain: str
    mappings: List[ConceptMapping]
    overall_similarity: float
    reasoning_chain: List[str]


@dataclass
class AbstractConcept:
    """A domain-independent abstract concept."""
    name: str
    description: str
    grounding_predicates: Dict[str, str]  # domain -> predicate
    transfer_rules: List[str]


class AnalogyEngine:
    """
    Enables cross-task transfer through analogical reasoning.
    
    Key idea: Snake and Pong share abstract structure:
    - AGENT that moves
    - TARGET it moves toward
    - DANGER it avoids
    
    By mapping concepts, rules transfer automatically.
    """
    
    def __init__(self):
        """Initialize with default abstract concepts."""
        # Abstract concept library (domain-independent)
        self.abstract_concepts: Dict[str, AbstractConcept] = {}
        
        # Domain-specific groundings: domain -> {local_concept: abstract_concept}
        self.groundings: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Cached analogies
        self.analogies: Dict[Tuple[str, str], Analogy] = {}
        
        # Initialize default abstractions
        self._init_default_abstractions()
    
    def _init_default_abstractions(self):
        """Set up default abstract concepts and groundings."""
        # Abstract AGENT concept
        self.register_abstract("AGENT", 
            description="The entity the player controls",
            groundings={
                "snake": "SNAKE_HEAD",
                "pong": "PLAYER_PADDLE",
                "maze": "MAZE_PLAYER",
            }
        )
        
        # Abstract TARGET concept
        self.register_abstract("TARGET",
            description="What the agent moves toward",
            groundings={
                "snake": "SNAKE_FOOD",
                "pong": "PONG_BALL",
                "maze": "MAZE_EXIT",
            }
        )
        
        # Abstract DANGER concept
        self.register_abstract("DANGER",
            description="What the agent must avoid",
            groundings={
                "snake": "SNAKE_BODY",
                "pong": "PONG_MISS",
                "maze": "MAZE_WALL",
            }
        )
        
        # Abstract TARGET_ABOVE relation
        self.register_abstract("TARGET_ABOVE",
            description="Target is above agent",
            groundings={
                "snake": "REL_ABOVE",
                "pong": "BALL_ABOVE",
                "maze": "EXIT_ABOVE",
            }
        )
        
        # Abstract TARGET_BELOW relation
        self.register_abstract("TARGET_BELOW",
            description="Target is below agent",
            groundings={
                "snake": "REL_BELOW",
                "pong": "BALL_BELOW",
                "maze": "EXIT_BELOW",
            }
        )
        
        # Abstract actions
        self.register_abstract("MOVE_UP",
            description="Move toward top of screen",
            groundings={
                "snake": "ACTION_UP",
                "pong": "ACTION_UP",
                "maze": "ACTION_UP",
            }
        )
        
        self.register_abstract("MOVE_DOWN",
            description="Move toward bottom of screen",
            groundings={
                "snake": "ACTION_DOWN",
                "pong": "ACTION_DOWN",
                "maze": "ACTION_DOWN",
            }
        )
    
    def register_abstract(
        self,
        name: str,
        description: str,
        groundings: Dict[str, str]
    ):
        """
        Register an abstract concept with domain groundings.
        
        Args:
            name: Abstract concept name
            description: Human-readable description
            groundings: Domain -> local concept mappings
        """
        self.abstract_concepts[name] = AbstractConcept(
            name=name,
            description=description,
            grounding_predicates=groundings,
            transfer_rules=[]
        )
        
        # Update reverse mappings
        for domain, local_concept in groundings.items():
            self.groundings[domain][local_concept] = name
    
    def lift_to_abstract(
        self,
        local_concept: str,
        domain: str
    ) -> Optional[str]:
        """
        Lift a domain-specific concept to abstract level.
        
        Args:
            local_concept: Domain-specific concept
            domain: Which domain
            
        Returns:
            Abstract concept name or None
        """
        return self.groundings.get(domain, {}).get(local_concept)
    
    def ground_to_domain(
        self,
        abstract_concept: str,
        target_domain: str
    ) -> Optional[str]:
        """
        Ground an abstract concept to a specific domain.
        
        Args:
            abstract_concept: Abstract concept name
            target_domain: Target domain
            
        Returns:
            Domain-specific concept or None
        """
        ac = self.abstract_concepts.get(abstract_concept)
        if ac:
            return ac.grounding_predicates.get(target_domain)
        return None
    
    def find_analogy(
        self,
        source_domain: str,
        target_domain: str
    ) -> Analogy:
        """
        Build analogical mapping between two domains.
        
        Args:
            source_domain: Domain with known knowledge
            target_domain: Domain to transfer to
            
        Returns:
            Analogy with all concept mappings
        """
        cache_key = (source_domain, target_domain)
        if cache_key in self.analogies:
            return self.analogies[cache_key]
        
        mappings = []
        source_groundings = self.groundings.get(source_domain, {})
        target_groundings = self.groundings.get(target_domain, {})
        
        # Map concepts through shared abstractions
        for source_local, abstract in source_groundings.items():
            target_local = self.ground_to_domain(abstract, target_domain)
            if target_local:
                mappings.append(ConceptMapping(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_concept=source_local,
                    target_concept=target_local,
                    similarity=1.0,  # Exact structural match
                    relation_type="structural"
                ))
        
        # Compute overall similarity
        source_count = len(source_groundings)
        matched_count = len(mappings)
        similarity = matched_count / max(source_count, 1)
        
        # Generate reasoning chain
        reasoning = []
        for m in mappings:
            abstract = self.lift_to_abstract(m.source_concept, source_domain)
            reasoning.append(
                f"{m.source_concept} ({source_domain}) ≈ {m.target_concept} ({target_domain}) via {abstract}"
            )
        
        analogy = Analogy(
            source_domain=source_domain,
            target_domain=target_domain,
            mappings=mappings,
            overall_similarity=similarity,
            reasoning_chain=reasoning
        )
        
        self.analogies[cache_key] = analogy
        return analogy
    
    def transfer_rule(
        self,
        rule_condition: Set[str],
        rule_action: str,
        source_domain: str,
        target_domain: str
    ) -> Tuple[Set[str], str]:
        """
        Transfer a rule from source to target domain.
        
        Args:
            rule_condition: Set of predicates in rule condition
            rule_action: Action consequence
            source_domain: Where rule was learned
            target_domain: Where to apply
            
        Returns:
            Tuple of (new_condition, new_action)
        """
        analogy = self.find_analogy(source_domain, target_domain)
        
        # Build mapping dict
        mapping = {}
        for m in analogy.mappings:
            mapping[m.source_concept] = m.target_concept
        
        # Map condition predicates
        new_condition = set()
        for pred in rule_condition:
            if pred in mapping:
                new_condition.add(mapping[pred])
            else:
                # Try partial match (e.g., REL_ABOVE -> BALL_ABOVE)
                abstract = self.lift_to_abstract(pred, source_domain)
                if abstract:
                    grounded = self.ground_to_domain(abstract, target_domain)
                    if grounded:
                        new_condition.add(grounded)
                    else:
                        new_condition.add(pred)  # Keep as-is
                else:
                    new_condition.add(pred)  # Keep as-is
        
        # Map action
        new_action = mapping.get(rule_action, rule_action)
        
        # If action is abstract-grounded
        abstract_action = self.lift_to_abstract(rule_action, source_domain)
        if abstract_action:
            grounded_action = self.ground_to_domain(abstract_action, target_domain)
            if grounded_action:
                new_action = grounded_action
        
        return (new_condition, new_action)
    
    def adapt_state(
        self,
        state: Dict[str, Any],
        source_domain: str,
        target_domain: str
    ) -> Dict[str, Any]:
        """
        Adapt state representation from source to target domain.
        
        Args:
            state: Source domain state
            source_domain: Where state is from
            target_domain: Target domain
            
        Returns:
            Adapted state dict
        """
        analogy = self.find_analogy(source_domain, target_domain)
        
        # Standard field mappings
        field_mapping = {
            # Snake -> Pong
            ("snake", "pong"): {
                "head": "p1_y",  # Agent position (y-only for Pong)
                "food": "ball_y",  # Target position (y-only)
            },
            # Snake -> Maze
            ("snake", "maze"): {
                "head": "player_pos",
                "food": "exit_pos",
                "body": "visited",
            },
        }
        
        mapping = field_mapping.get((source_domain, target_domain), {})
        
        adapted = {}
        for src_key, tgt_key in mapping.items():
            if src_key in state:
                val = state[src_key]
                # Handle tuple to scalar (e.g., (x,y) -> y)
                if isinstance(val, tuple) and tgt_key.endswith("_y"):
                    adapted[tgt_key] = val[1]
                else:
                    adapted[tgt_key] = val
        
        return adapted
    
    def zero_shot_action(
        self,
        state: Dict[str, Any],
        known_domain: str,
        new_domain: str,
        learned_rules: List[Tuple[Set[str], str]],
        active_predicates: Set[str]
    ) -> Optional[str]:
        """
        Get action for new domain using transferred knowledge.
        
        Args:
            state: Current state in new domain
            known_domain: Domain with learned rules
            new_domain: Domain to act in
            learned_rules: Rules from known domain
            active_predicates: Active predicates in new domain
            
        Returns:
            Recommended action or None
        """
        # Transfer each rule
        for condition, action in learned_rules:
            new_condition, new_action = self.transfer_rule(
                condition, action, known_domain, new_domain
            )
            
            # Check if transferred rule fires
            if new_condition.issubset(active_predicates):
                return new_action
        
        return None
    
    def get_transfer_explanation(
        self,
        source_domain: str,
        target_domain: str
    ) -> str:
        """Get human-readable explanation of transfer."""
        analogy = self.find_analogy(source_domain, target_domain)
        
        lines = [f"Transfer: {source_domain} → {target_domain}"]
        lines.append(f"Similarity: {analogy.overall_similarity:.0%}")
        lines.append("\nMappings:")
        
        for m in analogy.mappings:
            lines.append(f"  {m.source_concept} ≈ {m.target_concept}")
        
        return "\n".join(lines)
