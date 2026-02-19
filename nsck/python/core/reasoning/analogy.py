"""
NSCK Analogy Module
Cross-task transfer via structural alignment of concepts.

Uses VSA binding and symbolic mapping to identify analogous patterns
across different task domains for zero-shot knowledge transfer.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import python.core.vsa.hypervec_shim as hypervec_rs


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
    
    Key idea: domains can share abstract structure:
    - AGENT that moves
    - TARGET it moves toward
    - DANGER it avoids
    
    By mapping concepts, rules transfer automatically.
    
    Abstractions can be:
    1. Manually registered via ``register_abstract()``
    2. Auto-discovered via ``auto_discover_abstractions()`` using HV similarity
    3. Loaded from a built-in game preset via ``load_game_defaults()``
    """
    
    def __init__(self, load_defaults: bool = False):
        """Initialize the analogy engine.
        
        Args:
            load_defaults: If True, load built-in snake/pong/maze groundings.
                          Default False — caller should register domain groundings
                          explicitly or use auto_discover_abstractions().
        """
        # Abstract concept library (domain-independent)
        self.abstract_concepts: Dict[str, AbstractConcept] = {}
        
        # Domain-specific groundings: domain -> {local_concept: abstract_concept}
        self.groundings: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Cached analogies
        self.analogies: Dict[Tuple[str, str], Analogy] = {}
        
        if load_defaults:
            self.load_game_defaults()
    
    def load_game_defaults(self):
        """Load built-in snake/pong/maze abstract groundings.
        
        Separated from __init__ so non-game domains don't carry
        irrelevant mappings. Call explicitly when working with
        the built-in game environments.
        """
        _GAME_DEFAULTS = {
            "AGENT": {
                "description": "The entity the player controls",
                "groundings": {"snake": "SNAKE_HEAD", "pong": "PLAYER_PADDLE", "maze": "MAZE_PLAYER"},
            },
            "TARGET": {
                "description": "What the agent moves toward",
                "groundings": {"snake": "SNAKE_FOOD", "pong": "PONG_BALL", "maze": "MAZE_EXIT"},
            },
            "DANGER": {
                "description": "What the agent must avoid",
                "groundings": {"snake": "SNAKE_BODY", "pong": "PONG_MISS", "maze": "MAZE_WALL"},
            },
            "TARGET_ABOVE": {
                "description": "Target is above agent",
                "groundings": {"snake": "REL_ABOVE", "pong": "BALL_ABOVE", "maze": "EXIT_ABOVE"},
            },
            "TARGET_BELOW": {
                "description": "Target is below agent",
                "groundings": {"snake": "REL_BELOW", "pong": "BALL_BELOW", "maze": "EXIT_BELOW"},
            },
            "MOVE_UP": {
                "description": "Move toward top of screen",
                "groundings": {"snake": "ACTION_UP", "pong": "ACTION_UP", "maze": "ACTION_UP"},
            },
            "MOVE_DOWN": {
                "description": "Move toward bottom of screen",
                "groundings": {"snake": "ACTION_DOWN", "pong": "ACTION_DOWN", "maze": "ACTION_DOWN"},
            },
        }
        for name, spec in _GAME_DEFAULTS.items():
            self.register_abstract(name, spec["description"], spec["groundings"])

    def load_sensor_domain_defaults(self):
        """Register cross-domain structural abstractions for robot navigation
        and environmental / industrial monitoring (IIT-inspired mappings).

        These abstract concepts capture the invariant structure shared by
        any "agent monitoring a hazardous environment and taking corrective
        action" task, regardless of the surface predicates used.

        Call this when working with robot_nav ↔ env_monitoring transfer.
        """
        _SENSOR_DEFAULTS = {
            # ── Proximity to a boundary / critical threshold ────────────────
            "BOUNDARY_CONDITION": {
                "description": "Agent is near a physical or logical boundary",
                "groundings": {
                    "source_nav": "WALL_ADJACENT",
                    "target_env": "HIGH_CO2",
                    "robot": "WALL_ADJACENT",
                    "env": "HIGH_CO2",
                    "navigation": "WALL_ADJACENT",
                },
            },
            # ── Severe / imminent danger ────────────────────────────────────
            "CRITICAL_HAZARD": {
                "description": "Severe hazard requiring urgent response",
                "groundings": {
                    "source_nav": "ON_HAZARD",
                    "target_env": "DANGEROUS_CO2",
                    "robot": "ON_HAZARD",
                    "env": "HAZARDOUS_AIR",
                    "navigation": "ON_HAZARD",
                },
            },
            # ── Making progress toward goal / safe zone ─────────────────────
            "PROGRESS_TOWARD_GOAL": {
                "description": "Moving toward a goal or safe operating region",
                "groundings": {
                    "source_nav": "MID_GOAL",
                    "target_env": "POOR_AIR",
                    "robot": "MID_GOAL",
                    "env": "POOR_AIR",
                    "navigation": "MID_GOAL",
                },
            },
            # ── At or near goal / safe state ────────────────────────────────
            "AT_GOAL": {
                "description": "Agent has reached goal or safe operating state",
                "groundings": {
                    "source_nav": "NEAR_GOAL",
                    "target_env": "HIGH_HUMIDITY",
                    "robot": "NEAR_GOAL",
                    "env": "HIGH_HUMIDITY",
                    "navigation": "NEAR_GOAL",
                },
            },
            # ── Resource depletion ──────────────────────────────────────────
            "RESOURCE_LOW": {
                "description": "Critical resource is running low",
                "groundings": {
                    "source_nav": "LOW_BATTERY",
                    "target_env": "DRY_AIR",
                    "robot": "LOW_BATTERY",
                    "env": "DRY_AIR",
                    "navigation": "LOW_BATTERY",
                },
            },
            # ── Normal / safe operation ──────────────────────────────────────
            "SAFE_OPERATION": {
                "description": "System is operating within safe parameters",
                "groundings": {
                    "source_nav": "ACTION_NORMAL_OPERATION",
                    "target_env": "NORMAL_OPERATION",
                    "robot": "ACTION_STAY",
                    "env": "NORMAL_OPERATION",
                    "navigation": "ACTION_STAY",
                },
            },
        }
        for name, spec in _SENSOR_DEFAULTS.items():
            self.register_abstract(name, spec["description"], spec["groundings"])
    
    def register_domain(
        self,
        domain: str,
        groundings: Dict[str, str],
        create_missing_abstracts: bool = True,
    ):
        """
        Register all concept groundings for a new domain in one call.
        
        Args:
            domain: Domain name (e.g., "hvac", "robot_arm")
            groundings: Mapping of abstract_concept -> local_concept
                       e.g. {"AGENT": "ROBOT_ARM", "TARGET": "GOAL_POSITION"}
            create_missing_abstracts: If True, auto-create abstract concepts
                                     that don't exist yet.
        """
        for abstract_name, local_concept in groundings.items():
            if abstract_name not in self.abstract_concepts:
                if create_missing_abstracts:
                    self.register_abstract(
                        abstract_name,
                        description=f"Auto-registered from domain {domain}",
                        groundings={domain: local_concept},
                    )
                else:
                    continue
            else:
                # Add grounding to existing abstract concept
                ac = self.abstract_concepts[abstract_name]
                ac.grounding_predicates[domain] = local_concept
                self.groundings[domain][local_concept] = abstract_name
        
        # Invalidate cached analogies involving this domain
        to_remove = [k for k in self.analogies if domain in k]
        for k in to_remove:
            del self.analogies[k]
    
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
        target_domain: str,
        field_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Adapt state representation from source to target domain.
        
        Uses an explicit field_mapping if provided, otherwise attempts
        to map keys through the abstract concept layer.
        
        Args:
            state: Source domain state
            source_domain: Where state is from
            target_domain: Target domain
            field_mapping: Optional explicit {source_key: target_key} overrides
            
        Returns:
            Adapted state dict
        """
        mapping = field_mapping or {}
        
        adapted = {}
        for src_key, value in state.items():
            if src_key in mapping:
                adapted[mapping[src_key]] = value
            else:
                # Try abstract-concept bridge
                abstract = self.lift_to_abstract(src_key, source_domain)
                if abstract:
                    tgt_key = self.ground_to_domain(abstract, target_domain)
                    if tgt_key:
                        adapted[tgt_key] = value
                        continue
                # Keep original key if no mapping found
                adapted[src_key] = value
        
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

    # ----------------------------------------------------------------
    #  Auto-Abstraction via HV Similarity Clustering
    # ----------------------------------------------------------------
    def auto_discover_abstractions(
        self,
        domain_a: str,
        domain_b: str,
        concept_hvs_a: Dict[str, Any],
        concept_hvs_b: Dict[str, Any],
        similarity_threshold: float = 0.52,
    ) -> List[ConceptMapping]:
        """
        Automatically discover cross-domain concept alignments by
        comparing HyperVector similarity without manual registration.

        Algorithm
        ---------
        1. For every concept HV in *domain_a*, compute Hamming similarity
           against every concept HV in *domain_b*.
        2. Greedily pick the best 1-to-1 matching above *similarity_threshold*.
        3. Additionally, if names share a common stem (edit distance ≤ 3),
           boost the similarity score to allow softer matching.
        4. For each discovered pair, register a new auto-abstract concept
           so that future ``transfer_rule`` / ``find_analogy`` calls pick
           them up automatically.

        Parameters
        ----------
        domain_a, domain_b : Domain tags.
        concept_hvs_a : ``{concept_name: HyperVector}`` for domain_a.
        concept_hvs_b : ``{concept_name: HyperVector}`` for domain_b.
        similarity_threshold : Minimum similarity to accept a pairing (default lowered from 0.55 to 0.52).

        Returns
        -------
        List of newly created ``ConceptMapping`` objects.
        """
        pairs: list[tuple[float, str, str]] = []

        for name_a, hv_a in concept_hvs_a.items():
            for name_b, hv_b in concept_hvs_b.items():
                sim = hv_a.similarity(hv_b)
                # Name-similarity bonus: boost if names partially match
                name_bonus = 0.0
                na, nb = name_a.lower(), name_b.lower()
                # Check for shared prefix (≥ 3 chars)
                shared = 0
                for i in range(min(len(na), len(nb))):
                    if na[i] == nb[i]:
                        shared += 1
                    else:
                        break
                if shared >= 3:
                    name_bonus = 0.05
                # Check if one name contains the other
                if na in nb or nb in na:
                    name_bonus = 0.08
                adjusted_sim = sim + name_bonus
                if adjusted_sim >= similarity_threshold:
                    pairs.append((adjusted_sim, name_a, name_b))

        # Sort descending by similarity
        pairs.sort(key=lambda x: -x[0])

        used_a: set[str] = set()
        used_b: set[str] = set()
        discovered: list[ConceptMapping] = []

        for sim, name_a, name_b in pairs:
            if name_a in used_a or name_b in used_b:
                continue
            used_a.add(name_a)
            used_b.add(name_b)

            # Register a new auto-abstract concept
            abstract_name = f"AUTO_{name_a}_{name_b}".upper()
            self.register_abstract(
                abstract_name,
                description=f"Auto-discovered: {name_a} ({domain_a}) ≈ {name_b} ({domain_b})",
                groundings={domain_a: name_a, domain_b: name_b},
            )

            discovered.append(ConceptMapping(
                source_domain=domain_a,
                target_domain=domain_b,
                source_concept=name_a,
                target_concept=name_b,
                similarity=sim,
                relation_type="auto-discovered",
            ))

        # Invalidate cached analogy between these two domains
        self.analogies.pop((domain_a, domain_b), None)
        self.analogies.pop((domain_b, domain_a), None)

        return discovered

    def get_all_abstractions(self) -> Dict[str, Dict[str, str]]:
        """Return a summary of all abstract concepts and their groundings."""
        return {
            name: dict(ac.grounding_predicates)
            for name, ac in self.abstract_concepts.items()
        }
