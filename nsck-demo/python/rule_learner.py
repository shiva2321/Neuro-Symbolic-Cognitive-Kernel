"""
NSCK Rule Learner Module
Automatic rule induction from experience using frequency-based ILP.

NO GRADIENT DESCENT - uses counting, set logic, and symbolic induction only.
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, FrozenSet, Optional, Tuple, Any
import hypervec_rs
from persistence import BrainStore, Episode, Rule
from grounding_verifier import GroundingVerifier


@dataclass
class RuleCandidate:
    """A candidate rule before validation."""
    condition: FrozenSet[str]
    action: str
    support: int  # How many times this pattern was seen
    successes: int  # How many times it led to positive reward
    task_tag: str
    first_seen: float = 0.0
    last_seen: float = 0.0


class RuleLearner:
    """
    Learns rules from experience using frequency counting and symbolic logic.
    
    NO matrices, NO gradients — just counting patterns and validating them.
    
    Algorithm:
    1. Observe (state, action, outcome) tuples
    2. Extract active predicates from state
    3. Track frequency of predicate→action→outcome patterns
    4. Induce rules when pattern reaches min_support
    5. Validate rules against held-out episodes
    6. Prune low-performing rules
    """
    
    def __init__(
        self,
        verifier: GroundingVerifier,
        store: Optional[BrainStore] = None,
        min_support: int = 5,
        min_success_rate: float = 0.7,
        max_rules_per_task: int = 50
    ):
        """
        Initialize rule learner.
        
        Args:
            verifier: Grounding verifier for predicate extraction
            store: Optional persistence store
            min_support: Minimum observations before inducing rule
            min_success_rate: Minimum success rate for rule to be valid
            max_rules_per_task: Maximum rules to keep per task
        """
        self.verifier = verifier
        self.store = store
        self.min_support = min_support
        self.min_success_rate = min_success_rate
        self.max_rules_per_task = max_rules_per_task
        
        # Tenure thresholds for stability
        self.tenure_support_threshold = 1000  # High-support rules get tenure
        self.bootstrap_threshold = 0.5  # Bootstrap rules need 50%
        self.tenured_threshold = 0.6  # Tenured rules need 60%
        
        # Pattern tracking: task -> {(predicates, action): RuleCandidate}
        self.candidates: Dict[str, Dict[Tuple[FrozenSet[str], str], RuleCandidate]] = defaultdict(dict)
        
        # Learned rules: task -> [Rule]
        self.learned_rules: Dict[str, List[Rule]] = defaultdict(list)
        
        # Global predicate set (excludes task-specific)
        self.global_predicates = {
            "REL_ABOVE", "REL_BELOW", "REL_LEFT", "REL_RIGHT",
            "DANGER_UP", "DANGER_DOWN", "DANGER_LEFT", "DANGER_RIGHT",
            "TARGET_NEAR", "TARGET_FAR", "SAFE_PATH"
        }
    
    def _get_tenure_threshold(self, rule: Rule) -> float:
        """
        Compute success rate threshold based on rule tenure.
        
        Foundational rules get relaxed thresholds to prevent deletion
        during temporary performance dips.
        
        Args:
            rule: Rule to evaluate
            
        Returns:
            Minimum success rate required (0.0-1.0)
        """
        # Bootstrap rules: Most protected (hardcoded foundations)
        if rule.source == "bootstrap":
            return self.bootstrap_threshold
        
        # High-support learned rules: Protected (proven over time)
        if rule.support_count >= self.tenure_support_threshold:
            return self.tenured_threshold
        
        # New learned rules: Standard threshold
        return self.min_success_rate
    
    def observe(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        task_tag: str,
        outcome: str = "neutral"
    ):
        """
        Record an observation for pattern learning.
        
        Args:
            state: Game state dict
            action: Action taken (e.g., "ACTION_UP")
            reward: Reward received
            task_tag: Which task/game
            outcome: Outcome label ("success", "failure", "neutral")
        """
        # 1. Extract active predicates from state
        active_preds = self.verifier.get_active_predicates(state, context=task_tag)
        
        if not active_preds:
            return  # Can't learn from ungrounded state
        
        # 2. Normalize action
        action = action.upper()
        if not action.startswith("ACTION_"):
            action = f"ACTION_{action}"
        
        # 3. Create pattern key (use frozenset for hashability)
        pred_set = frozenset(active_preds)
        pattern_key = (pred_set, action)
        
        # 4. Update or create candidate
        now = time.time()
        candidates = self.candidates[task_tag]
        
        if pattern_key not in candidates:
            candidates[pattern_key] = RuleCandidate(
                condition=pred_set,
                action=action,
                support=0,
                successes=0,
                task_tag=task_tag,
                first_seen=now,
                last_seen=now
            )
        
        cand = candidates[pattern_key]
        cand.support += 1
        cand.last_seen = now
        
        # Track success (positive reward = success)
        if reward > 0 or outcome == "success":
            cand.successes += 1
    
    def induce_rules(self, task_tag: Optional[str] = None) -> List[Rule]:
        """
        Induce new rules from accumulated observations.
        
        Args:
            task_tag: Optional task to focus on, or all tasks
            
        Returns:
            List of newly induced rules
        """
        new_rules = []
        tasks = [task_tag] if task_tag else list(self.candidates.keys())
        
        for task in tasks:
            candidates = self.candidates.get(task, {})
            
            for pattern_key, cand in list(candidates.items()):
                # Check if meets threshold
                if cand.support < self.min_support:
                    continue
                
                success_rate = cand.successes / cand.support if cand.support > 0 else 0
                
                if success_rate < self.min_success_rate:
                    continue
                
                # Check if already learned
                existing = self._find_existing_rule(cand.condition, cand.action, task)
                if existing:
                    # Update existing rule's stats
                    existing.support_count = cand.support
                    existing.success_rate = success_rate
                    continue
                
                # Create new rule
                rule = Rule(
                    id=None,
                    condition=cand.condition,
                    consequence=cand.action,
                    priority=1,
                    source="learned",
                    task_tag=task,
                    scope=self._determine_scope(cand.condition),
                    support_count=cand.support,
                    success_rate=success_rate,
                    created_at=time.time()
                )
                
                self.learned_rules[task].append(rule)
                new_rules.append(rule)
                
                # Persist if store available
                if self.store:
                    self.store.save_rule(rule)
                
                print(f"[LEARN] New rule: {set(cand.condition)} → {cand.action} "
                      f"(support={cand.support}, rate={success_rate:.2f})")
        
        return new_rules
    
    def _determine_scope(self, condition: FrozenSet[str]) -> str:
        """Determine if rule uses only global predicates."""
        # Strip context prefixes for comparison
        core_preds = set()
        for pred in condition:
            if "::" in pred:
                core_preds.add(pred.split("::")[-1])
            else:
                core_preds.add(pred)
        
        # If all predicates are global, rule is candidate for promotion
        if core_preds.issubset(self.global_predicates):
            return "candidate_global"
        return "task_local"
    
    def _find_existing_rule(
        self,
        condition: FrozenSet[str],
        action: str,
        task_tag: str
    ) -> Optional[Rule]:
        """Find if a rule with same condition/action already exists."""
        for rule in self.learned_rules.get(task_tag, []):
            if rule.condition == condition and rule.consequence == action:
                return rule
        return None
    
    def validate_rules(
        self,
        task_tag: str,
        validation_episodes: List[Episode]
    ) -> Dict[int, float]:
        """
        Validate learned rules against held-out episodes.
        
        Args:
            task_tag: Which task's rules to validate
            validation_episodes: Episodes to validate against
            
        Returns:
            Dict of rule_id -> accuracy
        """
        results = {}
        
        for rule in self.learned_rules.get(task_tag, []):
            if rule.id is None:
                continue
            
            correct = 0
            applicable = 0
            
            for ep in validation_episodes:
                state = ep.state_sketch
                if not state:
                    continue
                
                # Check if rule's conditions are met
                active = self.verifier.get_active_predicates(state, context=task_tag)
                active_set = set(active)
                
                if rule.condition.issubset(active_set):
                    applicable += 1
                    if ep.action == rule.consequence:
                        correct += 1
            
            if applicable > 0:
                results[rule.id] = correct / applicable
            else:
                results[rule.id] = 0.0
        
        return results
    
    def prune_rules(self, task_tag: str, keep_top_n: Optional[int] = None):
        """
        Remove underperforming rules.
        
        Args:
            task_tag: Which task to prune
            keep_top_n: Keep only top N rules (by success rate)
        """
        if task_tag not in self.learned_rules:
            return
        
        rules = self.learned_rules[task_tag]
        
        # Filter out rules that fall below tenure-aware thresholds
        filtered_rules = []
        for rule in rules:
            threshold = self._get_tenure_threshold(rule)
            if rule.success_rate >= threshold:
                filtered_rules.append(rule)
            else:
                # Rule fell below its tenure-specific threshold
                if self.store and rule.id:
                    self.store.delete_rule(rule.id)
                print(f"[TENURE_PRUNE] Deleted rule {rule.consequence} "
                      f"(rate={rule.success_rate:.2f} < threshold={threshold:.2f}, "
                      f"source={rule.source}, support={rule.support_count})")
        
        rules = filtered_rules
        
        # Sort by success rate * log(support) to balance confidence and volume
        import math
        rules.sort(
            key=lambda r: r.success_rate * math.log1p(r.support_count),
            reverse=True
        )
        
        # Keep only top N
        limit = keep_top_n or self.max_rules_per_task
        pruned = rules[limit:]
        self.learned_rules[task_tag] = rules[:limit]
        
        # Delete from store
        if self.store:
            for rule in pruned:
                if rule.id:
                    self.store.delete_rule(rule.id)
        
        if pruned:
            print(f"[PRUNE] Removed {len(pruned)} rules from {task_tag}")
    
    def get_rules(self, task_tag: str) -> List[Rule]:
        """Get learned rules for a task."""
        return self.learned_rules.get(task_tag, [])
    
    def get_applicable_rules(
        self,
        state: Dict[str, Any],
        task_tag: str
    ) -> List[Tuple[Rule, float]]:
        """
        Get rules whose conditions match current state.
        
        Args:
            state: Current game state
            task_tag: Which task
            
        Returns:
            List of (rule, match_score) tuples, sorted by score
        """
        active = self.verifier.get_active_predicates(state, context=task_tag)
        active_set = set(active)
        
        matches = []
        
        for rule in self.learned_rules.get(task_tag, []):
            if rule.condition.issubset(active_set):
                # Score based on specificity and success rate
                specificity = len(rule.condition) / max(len(active_set), 1)
                score = rule.success_rate * (0.5 + 0.5 * specificity)
                matches.append((rule, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def load_from_store(self, task_tag: Optional[str] = None):
        """Load previously learned rules from persistence."""
        if not self.store:
            return
        
        rules = self.store.load_rules(task_tag=task_tag, scope=None)
        
        for rule in rules:
            if rule.source == "learned":
                self.learned_rules[rule.task_tag].append(rule)
        
        print(f"[LOAD] Loaded {len(rules)} learned rules")
