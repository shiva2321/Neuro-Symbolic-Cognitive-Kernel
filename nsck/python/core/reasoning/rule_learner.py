"""
NSCK Rule Learner Module
Automatic rule induction from experience using frequency-based ILP.

Implements pure symbolic learning via predicate counting and confidence thresholding,
with tenure-based stability for bootstrap/tenured/new rules.

Integration: Implements WorkspaceModule interface for substrate architecture.
"""
import logging
import time
import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, FrozenSet, Optional, Tuple, Any
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.integration.persistence import BrainStore, Episode, Rule
from python.core.perception.grounding_verifier import GroundingVerifier
from python.core.reasoning.global_workspace import WorkspaceModule, Coalition

logger = logging.getLogger("nsck.rule_learner")


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


class RuleLearner(WorkspaceModule):
    """
    Learns rules from experience using frequency counting and symbolic logic.
    
    NO matrices, NO gradients — just counting patterns and validating them.
    
    Implements WorkspaceModule interface for Global Workspace integration.
    
    Algorithm:
    1. Observe (state, action, outcome) tuples
    2. Extract active predicates from state
    3. Track frequency of predicate→action→outcome patterns
    4. Induce rules when pattern reaches min_support (or low-confidence threshold)
    5.Validate rules against held-out episodes
    6. Prune low-performing rules
    """
    
    def __init__(
        self,
        verifier: GroundingVerifier,
        store: Optional[BrainStore] = None,
        min_support: int = 5,
        min_confidence: float = 0.3,
        min_success_rate: float = 0.7,
        max_rules_per_task: int = 50
    ):
        """
        Initialize rule learner.
        
        Args:
            verifier: Grounding verifier for predicate extraction
            store: Optional persistence store
            min_support: Minimum observations for rule to graduate to full confidence
            min_confidence: Minimum confidence to create low-confidence rules (default: 0.3)
            min_success_rate: Minimum success rate for rule to be valid
            max_rules_per_task: Maximum rules to keep per task
        """
        self.verifier = verifier
        self.store = store
        self.min_support = min_support
        self.min_confidence = min_confidence  # NEW: Allow low-confidence rules
        self.min_success_rate = min_success_rate
        self.max_rules_per_task = max_rules_per_task
        
        # Approximate matching: allow subset predicate patterns
        self.use_approximate_matching = True
        self.subset_min_overlap = 0.6  # 60% overlap counts as match
        
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
        
        # Task-specific verifiers (registered via register_verifier()).
        # These override self.verifier for predicate extraction within observe().
        self.task_verifiers: Dict[str, GroundingVerifier] = {}

        # WorkspaceModule telemetry
        self._proposals_count = 0
        self._wins_count = 0
        self._last_state_hv = None
        self._last_proposed_rule = None
        self._current_task = None
    
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
    
    def register_verifier(self, task_tag: str, verifier: 'GroundingVerifier') -> None:
        """Register a domain-specific verifier for a task.

        Called automatically by CognitiveEngine.register_task() so that observe()
        can extract the correct predicates for each domain.
        """
        self.task_verifiers[task_tag] = verifier

    def observe(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        task_tag: str,
        outcome: str = "neutral",
        *,
        active_preds: Optional[List[str]] = None,
    ):
        """
        Record an observation for pattern learning.

        Args:
            state: Game state dict
            action: Action taken (e.g., "ACTION_UP")
            reward: Reward received
            task_tag: Which task/game
            outcome: Outcome label ("success", "failure", "neutral")
            active_preds: Pre-computed predicate list (overrides internal extraction).
                          Pass this from CognitiveEngine.learn() to avoid using the
                          default verifier which returns [] for unknown tasks.
        """
        # 1. Extract active predicates from state
        if active_preds is None:
            # Use task-specific verifier if registered, else fall back to default.
            verifier = self.task_verifiers.get(task_tag, self.verifier)
            active_preds = verifier.get_active_predicates(state, context=task_tag)

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

        # V9: Periodically append confidence snapshot to existing learned rules
        if cand.support % 10 == 0:
            existing_rule = self._find_existing_rule(pred_set, action, task_tag)
            if existing_rule is not None:
                existing_rule.confidence_history.append(existing_rule.confidence)
        
        # 5. Approximate matching: also credit overlapping patterns
        if self.use_approximate_matching:
            for existing_key, existing_cand in list(candidates.items()):
                if existing_key == pattern_key:
                    continue
                existing_preds, existing_action = existing_key
                if existing_action != action:
                    continue
                # Check overlap ratio
                overlap = len(pred_set & existing_preds)
                max_len = max(len(pred_set), len(existing_preds), 1)
                if overlap / max_len >= self.subset_min_overlap:
                    existing_cand.support += 0.5  # partial credit
                    if reward > 0 or outcome == "success":
                        existing_cand.successes += 0.5
    
    def induce_rules(self, task_tag: Optional[str]= None) -> List[Rule]:
        """
        Induce new rules from accumulated observations.
        
        Supports both low-confidence rules (1-2 observations) and full-confidence rules
        (min_support observations). Low-confidence rules participate in competition but
        receive activation penalty until they graduate to full confidence.
        
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
                # Calculate success rate and confidence
                success_rate = cand.successes / cand.support if cand.support > 0 else 0
                
                # Determine confidence based on support count
                low_conf_support = max(2, int(math.ceil(self.min_support / 2)))

                if cand.support >= self.min_support:
                    # Full support: full confidence if success rate is good
                    if success_rate < self.min_success_rate:
                        continue  # Not good enough even with full support
                    confidence = min(1.0, success_rate)
                elif cand.support >= low_conf_support:
                    # Low support: allow low-confidence rule once support is at least half min_support
                    if success_rate < self.min_success_rate:
                        continue  # Even low-confidence rules need decent success rate
                    # Confidence scales with support: support=2 -> 0.4, support=3 -> 0.6, etc.
                    confidence = max(self.min_confidence, 0.2 * cand.support)
                    confidence = min(confidence, 0.8)  # Cap at 0.8 until full support
                else:
                    continue  # No support
                
                # Check if already learned
                existing = self._find_existing_rule(cand.condition, cand.action, task)
                if existing:
                    # Update existing rule's stats and confidence
                    existing.support_count = cand.support
                    existing.success_rate = success_rate
                    existing.confidence = confidence
                    if self.store:
                        self.store.save_rule(existing)
                    continue
                
                # Create new rule with calculated confidence
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
                    confidence=confidence,
                    created_at=time.time()
                )
                
                self.learned_rules[task].append(rule)
                new_rules.append(rule)
                
                # Persist if store available
                if self.store:
                    self.store.save_rule(rule)
                
                status = "LOW-CONF" if confidence < 0.7 else "LEARNED"
                print(f"[{status}] New rule: {set(cand.condition)} -> {cand.action} "
                      f"(support={cand.support}, rate={success_rate:.2f}, conf={confidence:.2f})")
        
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
        active_preds: List[str],
        task_tag: str
    ) -> List[Tuple[Rule, float]]:
        """
        Get rules whose conditions match current state.
        
        Supports both exact and fuzzy matching. Exact subset matches
        score higher, but approximate matches (>= subset_min_overlap
        overlap) are also returned with a reduced score — mirroring the
        approximate matching already used in observe().
        
        Args:
            active_preds: Already extracted active predicates
            task_tag: Which task
            
        Returns:
            List of (rule, match_score) tuples, sorted by score descending
        """
        active_set = set(active_preds)
        
        matches = []
        
        # Check task-local rules
        local_rules = self.learned_rules.get(task_tag, [])
        
        # [AGI 6.3] Check global rules for transfer
        global_rules = self.learned_rules.get("global", [])
        
        for rule in local_rules + global_rules:
            # --- Exact match (strict subset) ---
            if rule.condition.issubset(active_set):
                specificity = len(rule.condition) / max(len(active_set), 1)
                score = rule.success_rate * (0.5 + 0.5 * specificity)
                matches.append((rule, score))
                continue
            
            # --- Fuzzy match (approximate overlap) ---
            if self.use_approximate_matching and rule.condition:
                overlap = len(rule.condition & active_set)
                max_len = max(len(rule.condition), len(active_set), 1)
                overlap_ratio = overlap / max_len
                
                if overlap_ratio >= self.subset_min_overlap:
                    specificity = len(rule.condition) / max(len(active_set), 1)
                    # Discount score by overlap ratio to rank below exact matches
                    score = rule.success_rate * (0.5 + 0.5 * specificity) * overlap_ratio
                    matches.append((rule, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        # V9: update fire tracking for matched rules
        _now = time.time()
        for rule, _ in matches:
            rule.fire_count += 1
            rule.last_fired = _now
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

    # V9: Cross-domain transfer
    def add_transferred_rule(
        self,
        task_tag: str,
        condition: FrozenSet[str],
        consequence: str,
        source_task: str,
        source_confidence: float,
    ) -> None:
        """Inject a rule transferred from *source_task* into *task_tag* (V9).

        Only adds the rule if an equivalent condition+consequence pair does
        not already exist for *task_tag*.

        Parameters
        ----------
        task_tag : str
            Target task to inject the rule into.
        condition : frozenset of str
            Predicate set forming the rule condition (already translated to
            target domain vocabulary by AnalogyEngine.transfer_rule).
        consequence : str
            Action consequence.
        source_task : str
            Origin domain (stored in rule.source for traceability).
        source_confidence : float
            Confidence of the original rule, used as a starting confidence.
        """
        # Avoid duplicates
        if self._find_existing_rule(condition, consequence, task_tag):
            return

        rule = Rule(
            id=None,
            condition=condition,
            consequence=consequence,
            priority=1,
            source=f"transferred:{source_task}",
            task_tag=task_tag,
            scope=self._determine_scope(condition),
            support_count=0,
            success_rate=0.0,
            confidence=max(0.3, source_confidence * 0.8),  # slight discount
            created_at=time.time(),
        )
        self.learned_rules[task_tag].append(rule)
        if self.store:
            self.store.save_rule(rule)
        logger.info(
            "[TRANSFER] Injected rule %s→%s into '%s' from '%s'",
            set(condition), consequence, task_tag, source_task,
        )    
    # ========================================================================
    # WorkspaceModule Interface Implementation
    # ========================================================================
    
    def set_current_task(self, task_tag: str):
        """Set the current task context for proposal generation."""
        self._current_task = task_tag
    
    def receive_broadcast(self, content: Any):
        """
        Receive broadcast from GlobalWorkspace when another module wins.
        
        Handles both structured dict broadcasts (with 'winner' key) and
        raw action string broadcasts from older integration paths.
        
        Args:
            content: Winner's content — either a Dict with 'action'/'winner' keys,
                     or a raw action string.
        """
        if isinstance(content, dict):
            winner = content.get('winner', content.get('source', ''))
            if winner == 'RuleLearner':
                self._wins_count += 1
        elif isinstance(content, str):
            # Raw action string broadcast — can't determine winner from string
            pass
    
    def propose(self, state_hv: Optional[np.ndarray]) -> Optional[Coalition]:
        """
        Generate rule-based action proposal for current state.
        
        Args:
            state_hv: Current state encoded as hypervector (10240-bit), or None
        
        Returns:
            Coalition with best-matching rule proposal, or None if no applicable rules
        """
        self._proposals_count += 1
        self._last_state_hv = state_hv
        
        if not self._current_task:
            return None  # Need task context
        
        # Get applicable rules (requires active predicates from state)
        # NOTE: This requires state to be passed as dict with predicates
        # For now, we'll return None if we can't access state predicates
        # In full integration, CognitiveEngine would pass both state_hv and state_dict
        
        # Placeholder: In real integration, extract predicates from state
        # For now, check if we have any high-confidence rules
        task_rules = self.learned_rules.get(self._current_task, [])
        if not task_rules:
            return None
        
        # Find highest confidence rule
        best_rule = max(task_rules, key=lambda r: r.confidence * r.success_rate)
        
        if best_rule.confidence < self.min_confidence:
            return None  # Too low confidence to propose
        
        self._last_proposed_rule = best_rule
        
        # Calculate salience based on confidence and success rate
        base_salience = 0.5 + (0.3 * best_rule.confidence)
        relevance = best_rule.success_rate * 0.3
        
        return Coalition(
            source="RuleLearner",
            content={
                "action": best_rule.consequence,
                "rule_id": best_rule.id,
                "rule_condition": list(best_rule.condition),
                "reasoning": f"Rule #{best_rule.id}: {len(best_rule.condition)} predicates → {best_rule.consequence}"
            },
            base_salience=base_salience,
            relevance=relevance,
            affect_match=0.0,  # Rules are emotionally neutral
            sender_confidence=best_rule.confidence
        )
    
    def update(self, feedback_hv: Optional[np.ndarray], reward: float, info: Dict[str, Any]):
        """
        Learn from action outcome.
        
        Updates rule confidence based on whether the action succeeded.
        All modules learn observationally even if they didn't win.
        
        Args:
            feedback_hv: Resulting state after action
            reward: Scalar reward signal
            info: Context dict with 'winner', 'action_taken', 'success', etc.
        """
        winner = info.get('winner')
        action = info.get('action_taken')
        
        # If we won, update the rule we proposed
        if winner == 'RuleLearner' and self._last_proposed_rule is not None:
            rule = self._last_proposed_rule
            
            # Update confidence based on outcome
            if reward > 0:
                # Success: increase confidence (capped at 1.0)
                rule.confidence = min(1.0, rule.confidence + 0.05)
                rule.success_rate = min(1.0, rule.success_rate + 0.02)
            else:
                # Failure: decrease confidence (floored at min_confidence)
                rule.confidence = max(self.min_confidence, rule.confidence - 0.1)
                rule.success_rate = max(0.0, rule.success_rate - 0.05)
            
            # Persist updated rule
            if self.store and rule.id:
                self.store.save_rule(rule)
        
        # Store state for next update cycle
        self._last_state_hv = feedback_hv
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Return current RuleLearner status for monitoring.
        
        Returns:
            Dict with telemetry data
        """
        total_rules = sum(len(rules) for rules in self.learned_rules.values())
        low_conf_rules = sum(
            1 for rules in self.learned_rules.values()
            for rule in rules if rule.confidence < 0.7
        )
        
        win_rate = self._wins_count / max(1, self._proposals_count)
        
        avg_confidence = 0.0
        if total_rules > 0:
            all_rules = [r for rules in self.learned_rules.values() for r in rules]
            avg_confidence = sum(r.confidence for r in all_rules) / total_rules
        
        return {
            'active': True,
            'proposals_count': self._proposals_count,
            'wins_count': self._wins_count,
            'win_rate': win_rate,
            'confidence': avg_confidence,
            'memory_size': total_rules,
            'total_rules': total_rules,
            'low_confidence_rules': low_conf_rules,
            'high_confidence_rules': total_rules - low_conf_rules,
            'current_task': self._current_task,
            'tasks_tracked': len(self.learned_rules),
            'candidates_count': sum(len(c) for c in self.candidates.values())
        }