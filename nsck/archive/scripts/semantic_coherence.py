"""
NSCK Semantic Coherence Module
Validates that symbolic states and rules are logically consistent.

Detects and resolves contradictions in the knowledge base.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict


@dataclass
class Contradiction:
    """A detected logical contradiction."""
    type: str  # "mutual_exclusion", "rule_conflict", "state_impossible"
    severity: float  # 0.0 to 1.0
    predicates: List[str]
    description: str
    resolution: Optional[str] = None


@dataclass
class CoherenceCheck:
    """Result of coherence checking."""
    is_coherent: bool
    contradictions: List[Contradiction]
    warnings: List[str]
    confidence: float


class SemanticCoherence:
    """
    Validates logical coherence of symbolic knowledge.
    
    Checks:
    - Mutual exclusions (can't be UP and DOWN at same time)
    - State consistency (predicates match actual state)
    - Rule conflicts (two rules suggest opposite actions)
    - Temporal consistency (history doesn't contradict)
    """
    
    def __init__(self):
        """Initialize with default constraints."""
        # Mutually exclusive predicate pairs
        self.exclusions: Dict[str, Set[str]] = defaultdict(set)
        
        # Required implications (if A then must have B)
        self.implications: Dict[str, Set[str]] = defaultdict(set)
        
        # Action conflicts (opposite actions)
        self.action_conflicts = {
            ("ACTION_UP", "ACTION_DOWN"),
            ("ACTION_LEFT", "ACTION_RIGHT"),
            ("UP", "DOWN"),
            ("LEFT", "RIGHT"),
        }
        
        # Initialize default exclusions
        self._init_default_exclusions()
    
    def _init_default_exclusions(self):
        """Set up default mutual exclusions."""
        # Directional exclusions
        self.add_exclusion("REL_ABOVE", "REL_BELOW")
        self.add_exclusion("REL_LEFT", "REL_RIGHT")
        
        # Pong exclusions
        self.add_exclusion("BALL_ABOVE", "BALL_BELOW")
        self.add_exclusion("BALL_ABOVE", "BALL_ALIGNED")
        self.add_exclusion("BALL_BELOW", "BALL_ALIGNED")
        
        # State exclusions
        self.add_exclusion("ALIVE", "DEAD")
        self.add_exclusion("GAME_RUNNING", "GAME_OVER")
    
    def add_exclusion(self, pred1: str, pred2: str):
        """Add a mutual exclusion constraint."""
        self.exclusions[pred1].add(pred2)
        self.exclusions[pred2].add(pred1)
    
    def add_implication(self, antecedent: str, consequent: str):
        """Add an implication: if antecedent then consequent."""
        self.implications[antecedent].add(consequent)
    
    def check_predicates(self, predicates: Set[str]) -> CoherenceCheck:
        """
        Check if a set of predicates is internally coherent.
        
        Args:
            predicates: Set of active predicates
            
        Returns:
            CoherenceCheck with any contradictions
        """
        contradictions = []
        warnings = []
        
        # Check mutual exclusions
        for pred in predicates:
            excluded = self.exclusions.get(pred, set())
            violations = excluded & predicates
            
            for violation in violations:
                if pred < violation:  # Avoid duplicates
                    contradictions.append(Contradiction(
                        type="mutual_exclusion",
                        severity=0.9,
                        predicates=[pred, violation],
                        description=f"{pred} and {violation} cannot both be true",
                        resolution=f"Remove one of: {pred}, {violation}"
                    ))
        
        # Check implications
        for pred in predicates:
            required = self.implications.get(pred, set())
            missing = required - predicates
            
            for miss in missing:
                warnings.append(f"{pred} implies {miss}, but {miss} is not present")
        
        is_coherent = len(contradictions) == 0
        confidence = 1.0 - (0.1 * len(contradictions)) - (0.05 * len(warnings))
        
        return CoherenceCheck(
            is_coherent=is_coherent,
            contradictions=contradictions,
            warnings=warnings,
            confidence=max(0.0, confidence)
        )
    
    def check_state_consistency(
        self,
        state: Dict[str, Any],
        predicates: Set[str],
        task_tag: str
    ) -> CoherenceCheck:
        """
        Check if predicates match the actual state.
        
        Args:
            state: Actual game state
            predicates: Active predicates
            task_tag: Which task
            
        Returns:
            CoherenceCheck with any inconsistencies
        """
        contradictions = []
        warnings = []
        
        if task_tag == "snake":
            head = state.get("head", (5, 5))
            food = state.get("food", (5, 5))
            
            dy = food[1] - head[1]
            dx = food[0] - head[0]
            
            # Check directional predicates match
            if "REL_ABOVE" in predicates and dy >= 0:
                contradictions.append(Contradiction(
                    type="state_impossible",
                    severity=0.7,
                    predicates=["REL_ABOVE"],
                    description=f"REL_ABOVE claimed but food.y ({food[1]}) >= head.y ({head[1]})"
                ))
            
            if "REL_BELOW" in predicates and dy <= 0:
                contradictions.append(Contradiction(
                    type="state_impossible",
                    severity=0.7,
                    predicates=["REL_BELOW"],
                    description=f"REL_BELOW claimed but food.y ({food[1]}) <= head.y ({head[1]})"
                ))
            
            if "REL_LEFT" in predicates and dx >= 0:
                contradictions.append(Contradiction(
                    type="state_impossible",
                    severity=0.7,
                    predicates=["REL_LEFT"],
                    description=f"REL_LEFT claimed but food.x ({food[0]}) >= head.x ({head[0]})"
                ))
            
            if "REL_RIGHT" in predicates and dx <= 0:
                contradictions.append(Contradiction(
                    type="state_impossible",
                    severity=0.7,
                    predicates=["REL_RIGHT"],
                    description=f"REL_RIGHT claimed but food.x ({food[0]}) <= head.x ({head[0]})"
                ))
        
        elif task_tag == "pong":
            ball_y = state.get("ball_y", 15)
            paddle_y = state.get("p1_y", 10)
            paddle_center = paddle_y + 3
            
            if "BALL_ABOVE" in predicates and ball_y >= paddle_center:
                contradictions.append(Contradiction(
                    type="state_impossible",
                    severity=0.7,
                    predicates=["BALL_ABOVE"],
                    description=f"BALL_ABOVE claimed but ball_y ({ball_y}) >= paddle_center ({paddle_center})"
                ))
            
            if "BALL_BELOW" in predicates and ball_y <= paddle_center:
                contradictions.append(Contradiction(
                    type="state_impossible",
                    severity=0.7,
                    predicates=["BALL_BELOW"],
                    description=f"BALL_BELOW claimed but ball_y ({ball_y}) <= paddle_center ({paddle_center})"
                ))
        
        is_coherent = len(contradictions) == 0
        confidence = 1.0 - (0.1 * len(contradictions))
        
        return CoherenceCheck(
            is_coherent=is_coherent,
            contradictions=contradictions,
            warnings=warnings,
            confidence=max(0.0, confidence)
        )
    
    def check_rule_conflict(
        self,
        rules: List[Tuple[Set[str], str]],  # [(conditions, action), ...]
        active_predicates: Set[str]
    ) -> CoherenceCheck:
        """
        Check if multiple rules fire with conflicting actions.
        
        Args:
            rules: List of (condition_set, action) tuples
            active_predicates: Currently active predicates
            
        Returns:
            CoherenceCheck with any conflicts
        """
        contradictions = []
        warnings = []
        
        # Find all rules that fire
        firing_rules = []
        for conditions, action in rules:
            if conditions.issubset(active_predicates):
                firing_rules.append((conditions, action))
        
        # Check for conflicting actions
        if len(firing_rules) > 1:
            actions = [action for _, action in firing_rules]
            
            for i, (cond1, act1) in enumerate(firing_rules):
                for cond2, act2 in firing_rules[i+1:]:
                    # Normalize actions
                    a1 = act1.replace("ACTION_", "")
                    a2 = act2.replace("ACTION_", "")
                    
                    if (a1, a2) in self.action_conflicts or (a2, a1) in self.action_conflicts:
                        contradictions.append(Contradiction(
                            type="rule_conflict",
                            severity=0.8,
                            predicates=list(cond1 | cond2),
                            description=f"Rules fire with conflicting actions: {act1} vs {act2}",
                            resolution="Add priority or more specific conditions"
                        ))
        
        is_coherent = len(contradictions) == 0
        confidence = 1.0 - (0.15 * len(contradictions))
        
        return CoherenceCheck(
            is_coherent=is_coherent,
            contradictions=contradictions,
            warnings=warnings,
            confidence=max(0.0, confidence)
        )
    
    def resolve_contradiction(
        self,
        contradiction: Contradiction,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Attempt to resolve a contradiction.
        
        Args:
            contradiction: The contradiction to resolve
            context: Additional context
            
        Returns:
            Resolution action or None if cannot resolve
        """
        if contradiction.type == "mutual_exclusion":
            # Keep the predicate that matches state
            preds = contradiction.predicates
            if len(preds) == 2:
                # Check which is grounded
                # For now, just take first
                return f"Keep {preds[0]}, remove {preds[1]}"
        
        elif contradiction.type == "rule_conflict":
            return "Use higher-priority rule or more recent rule"
        
        elif contradiction.type == "state_impossible":
            return f"Remove invalid predicate: {contradiction.predicates[0]}"
        
        return None
    
    def full_check(
        self,
        state: Dict[str, Any],
        predicates: Set[str],
        rules: List[Tuple[Set[str], str]],
        task_tag: str
    ) -> CoherenceCheck:
        """
        Run all coherence checks.
        
        Args:
            state: Current game state
            predicates: Active predicates
            rules: Available rules
            task_tag: Which task
            
        Returns:
            Combined CoherenceCheck
        """
        all_contradictions = []
        all_warnings = []
        
        # Check predicate coherence
        pred_check = self.check_predicates(predicates)
        all_contradictions.extend(pred_check.contradictions)
        all_warnings.extend(pred_check.warnings)
        
        # Check state consistency
        state_check = self.check_state_consistency(state, predicates, task_tag)
        all_contradictions.extend(state_check.contradictions)
        all_warnings.extend(state_check.warnings)
        
        # Check rule conflicts
        rule_check = self.check_rule_conflict(rules, predicates)
        all_contradictions.extend(rule_check.contradictions)
        all_warnings.extend(rule_check.warnings)
        
        is_coherent = len(all_contradictions) == 0
        confidence = min(
            pred_check.confidence,
            state_check.confidence,
            rule_check.confidence
        )
        
        return CoherenceCheck(
            is_coherent=is_coherent,
            contradictions=all_contradictions,
            warnings=all_warnings,
            confidence=confidence
        )
