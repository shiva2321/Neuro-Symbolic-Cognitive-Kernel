"""
System 2: Slow Reasoning Layer

System 2 is the explicit, symbolic reasoning system that operates on top of
the fast spike-based System 1 and the monitoring System 1.5 control layer.

Philosophy:
- System 1 (spikes): Fast, automatic, unconscious
- System 1.5 (control): Monitors, gates plasticity, detects uncertainty
- System 2 (reasoning): Slow, deliberate, explicit

System 2 Handles:
- Explicit rule learning and application
- Goal-driven behavior
- Conflict resolution
- Consolidation and replay
- Conscious reasoning

Invariants:
- System 2 operates on derived facts (not raw spikes)
- Reasoning is asynchronous (not real-time)
- Results can gate System 1 behavior
- System 2 never directly modifies synapses (goes through System 1.5)
- Failures in System 2 don't crash System 1
"""

from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time


class RuleType(Enum):
    """Types of learned rules."""
    IF_THEN = "if_then"              # If condition, then action
    ASSOCIATION = "association"       # Stimulus -> Response
    CAUSAL = "causal"                # A causes B
    CONSTRAINT = "constraint"         # X must/must-not happen
    PREFERENCE = "preference"         # Prefer X over Y


@dataclass
class Rule:
    """An explicit learned rule."""
    rule_id: int
    rule_type: RuleType
    condition: str                    # Description of condition
    action: str                       # Description of action/consequence
    confidence: float                 # How confident are we? [0, 1]
    frequency: int = 0                # How many times observed
    last_applied: int = -1            # Last timestep applied

    def __repr__(self):
        return f"Rule({self.rule_type.value}: {self.condition} → {self.action}, conf={self.confidence:.2f})"


@dataclass
class Fact:
    """An explicit fact about the world."""
    fact_id: int
    content: str                      # Description of fact
    confidence: float                 # How confident? [0, 1]
    source: str                       # "system1", "system2", "external"
    timestamp: int                    # When learned

    def __repr__(self):
        return f"Fact({self.content}, conf={self.confidence:.2f})"


@dataclass
class Goal:
    """
    An explicit goal the system can pursue (Phase 5).

    Attributes:
        goal_id: Unique identifier for the goal
        state: Current state ("idle", "pursuing", "achieved", "failed")
        utility: Importance/priority of goal [0, 1]
        progress: Current progress toward goal [0, 1]
        reward_model: Function that estimates expected reward given current state
        description: Human-readable description of the goal
    """
    goal_id: int
    state: str = "idle"                    # "idle", "pursuing", "achieved", "failed"
    utility: float = 0.5                   # Importance [0, 1]
    progress: float = 0.0                  # Completion [0, 1]
    reward_model: Optional[Callable] = None  # f(state) -> expected_reward
    description: str = ""                  # Human-readable description

    def __repr__(self):
        return f"Goal({self.goal_id}: {self.description}, util={self.utility:.2f}, prog={self.progress:.2f}, state={self.state})"


@dataclass
class ConflictResolution:
    """Resolution of a conflict."""
    conflict_id: int
    patterns: List[str]               # Conflicting patterns
    resolution: str                   # How resolved
    chosen_action: str                # Which action selected
    reasoning: str                    # Why selected
    timestamp: int                    # When resolved


class KnowledgeBase:
    """
    Explicit knowledge storage for System 2.

    Stores:
    - Rules (if-then, causal, etc.)
    - Facts (learned knowledge)
    - Conflict resolutions
    - Success/failure history
    """

    def __init__(self):
        """Initialize knowledge base."""
        self._rules: Dict[int, Rule] = {}
        self._facts: Dict[int, Fact] = {}
        self._rule_counter = 0
        self._fact_counter = 0
        self._access_history: List[tuple] = []

    def add_rule(
        self,
        rule_type: RuleType,
        condition: str,
        action: str,
        confidence: float = 0.5
    ) -> int:
        """
        Add a rule to knowledge base.

        Args:
            rule_type: Type of rule
            condition: Condition description
            action: Action/consequence description
            confidence: Confidence in rule [0, 1]

        Returns:
            Rule ID
        """
        rule_id = self._rule_counter
        self._rule_counter += 1

        rule = Rule(
            rule_id=rule_id,
            rule_type=rule_type,
            condition=condition,
            action=action,
            confidence=confidence
        )

        self._rules[rule_id] = rule
        self._access_history.append(("add_rule", rule_id, time.time()))

        return rule_id

    def add_fact(
        self,
        content: str,
        confidence: float = 0.8,
        source: str = "system2"
    ) -> int:
        """
        Add a fact to knowledge base.

        Args:
            content: Fact description
            confidence: Confidence [0, 1]
            source: Where fact came from

        Returns:
            Fact ID
        """
        fact_id = self._fact_counter
        self._fact_counter += 1

        fact = Fact(
            fact_id=fact_id,
            content=content,
            confidence=confidence,
            source=source,
            timestamp=int(time.time())
        )

        self._facts[fact_id] = fact
        self._access_history.append(("add_fact", fact_id, time.time()))

        return fact_id

    def get_rules(self, rule_type: Optional[RuleType] = None) -> List[Rule]:
        """Get rules, optionally filtered by type."""
        rules = list(self._rules.values())
        if rule_type:
            rules = [r for r in rules if r.rule_type == rule_type]
        return rules

    def get_facts(self) -> List[Fact]:
        """Get all facts."""
        return list(self._facts.values())

    def increment_rule_frequency(self, rule_id: int) -> None:
        """Increment frequency of rule (used when rule applies)."""
        if rule_id in self._rules:
            self._rules[rule_id].frequency += 1

    def update_rule_confidence(self, rule_id: int, new_confidence: float) -> None:
        """Update confidence of rule based on outcomes."""
        if rule_id in self._rules:
            # EMA update
            old_conf = self._rules[rule_id].confidence
            self._rules[rule_id].confidence = 0.7 * old_conf + 0.3 * new_confidence

    def get_statistics(self) -> Dict:
        """Get knowledge base statistics."""
        return {
            "num_rules": len(self._rules),
            "num_facts": len(self._facts),
            "total_rule_frequency": sum(r.frequency for r in self._rules.values()),
            "mean_rule_confidence": sum(r.confidence for r in self._rules.values()) / max(len(self._rules), 1),
            "mean_fact_confidence": sum(f.confidence for f in self._facts.values()) / max(len(self._facts), 1),
        }


class ConflictResolver:
    """
    Resolves conflicts detected by System 1.5.

    When System 1.5 detects high conflict/low confidence,
    System 2 reasoning kicks in to resolve ambiguity.
    """

    def __init__(self):
        """Initialize conflict resolver."""
        self._resolutions: List[ConflictResolution] = []
        self._resolution_counter = 0
        self._conflict_history: List[tuple] = []

    def resolve_conflict(
        self,
        patterns: List[str],
        available_actions: List[str],
        knowledge_base: KnowledgeBase,
        context: Optional[Dict] = None
    ) -> str:
        """
        Resolve a conflict using reasoning and knowledge base.

        Args:
            patterns: Conflicting patterns
            available_actions: Available actions to choose
            knowledge_base: Knowledge base to consult
            context: Additional context

        Returns:
            Chosen action
        """
        # Strategy: Use rules to select best action
        rules = knowledge_base.get_rules()
        facts = knowledge_base.get_facts()

        action_scores = defaultdict(float)

        # Score each action based on matching rules
        for action in available_actions:
            for rule in rules:
                # If rule's action matches available action
                if action.lower() in rule.action.lower():
                    # Weight by rule confidence and frequency
                    score = rule.confidence * (1.0 + 0.1 * rule.frequency)
                    action_scores[action] += score

        # Choose action with highest score
        if action_scores:
            chosen = max(action_scores, key=action_scores.get)
        else:
            # Fallback: choose first action
            chosen = available_actions[0] if available_actions else "no_action"

        # Record resolution
        resolution = ConflictResolution(
            conflict_id=self._resolution_counter,
            patterns=patterns,
            resolution=f"Scored {len(available_actions)} actions",
            chosen_action=chosen,
            reasoning=f"Best score: {action_scores.get(chosen, 0.0):.2f}",
            timestamp=int(time.time())
        )

        self._resolutions.append(resolution)
        self._conflict_history.append((patterns, chosen))
        self._resolution_counter += 1

        return chosen

    def get_statistics(self) -> Dict:
        """Get conflict resolution statistics."""
        return {
            "total_conflicts": len(self._conflict_history),
            "resolutions": len(self._resolutions),
            "recent_resolutions": self._resolutions[-5:] if self._resolutions else [],
        }


class Consolidator:
    """
    Memory consolidation: converts temporary learning to long-term knowledge.

    Simulates the consolidation process where temporary (short-term)
    learning becomes permanent long-term rules and facts.
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        """Initialize consolidator."""
        self.knowledge_base = knowledge_base
        self._consolidation_history: List[tuple] = []
        self._replay_queue: List[Dict] = []

    def add_to_consolidation_queue(self, memory: Dict) -> None:
        """
        Add memory to consolidation queue (simulated sleep/replay).

        Args:
            memory: Memory to consolidate
                {
                    "pattern": str,
                    "outcome": str,
                    "confidence": float
                }
        """
        self._replay_queue.append(memory)

    def consolidate(self, memory: Dict) -> bool:
        """
        Consolidate a memory into permanent knowledge.

        Args:
            memory: Memory to consolidate

        Returns:
            True if consolidation successful
        """
        if memory.get("confidence", 0.0) < 0.6:
            # Low confidence memories not consolidated
            return False

        pattern = memory.get("pattern", "unknown")
        outcome = memory.get("outcome", "unknown")
        confidence = memory.get("confidence", 0.5)

        # Convert to rule
        rule_id = self.knowledge_base.add_rule(
            rule_type=RuleType.ASSOCIATION,
            condition=f"Pattern: {pattern}",
            action=f"Produces: {outcome}",
            confidence=confidence
        )

        self._consolidation_history.append((pattern, outcome, rule_id))

        return True

    def consolidate_all(self) -> int:
        """
        Consolidate all memories in queue.

        Returns:
            Number of memories consolidated
        """
        count = 0
        for memory in self._replay_queue:
            if self.consolidate(memory):
                count += 1

        self._replay_queue.clear()
        return count

    def get_statistics(self) -> Dict:
        """Get consolidation statistics."""
        return {
            "consolidations": len(self._consolidation_history),
            "in_queue": len(self._replay_queue),
            "success_rate": len(self._consolidation_history) / max(len(self._consolidation_history) + len(self._replay_queue), 1),
        }


class System2:
    """
    System 2: Slow, deliberate, explicit reasoning.

    Receives information from System 1.5 (conflicts, uncertainty)
    and reasons explicitly to resolve ambiguity and learn rules.
    """

    def __init__(self):
        """Initialize System 2."""
        self.knowledge_base = KnowledgeBase()
        self.conflict_resolver = ConflictResolver()
        self.consolidator = Consolidator(self.knowledge_base)

        self._reasoning_invocations = 0
        self._consolidation_invocations = 0
        self._timestamp = 0

        # Phase 5: Goal-directed behavior
        self._goals: Dict[int, Goal] = {}
        self._current_goal_id: Optional[int] = None
        self._goal_history: List[Tuple[int, str, int]] = []  # (goal_id, state, timestamp)

    def handle_conflict(
        self,
        patterns: List[str],
        available_actions: List[str],
        context: Optional[Dict] = None
    ) -> str:
        """
        Handle a conflict detected by System 1.5.

        Args:
            patterns: Conflicting patterns
            available_actions: Available actions
            context: Additional context

        Returns:
            Chosen action
        """
        self._reasoning_invocations += 1

        chosen_action = self.conflict_resolver.resolve_conflict(
            patterns=patterns,
            available_actions=available_actions,
            knowledge_base=self.knowledge_base,
            context=context
        )

        return chosen_action

    def handle_uncertainty(
        self,
        confidence: float,
        current_belief: str,
        options: List[str]
    ) -> str:
        """
        Handle uncertainty (low confidence) from System 1.5.

        Args:
            confidence: Confidence level [0, 1]
            current_belief: Current output/belief
            options: Available interpretations

        Returns:
            Best interpretation
        """
        self._reasoning_invocations += 1

        # Get facts from knowledge base
        facts = self.knowledge_base.get_facts()

        # Score options based on known facts
        option_scores = {}
        for option in options:
            score = 0.5  # Default
            for fact in facts:
                if option.lower() in fact.content.lower():
                    score = max(score, fact.confidence)
            option_scores[option] = score

        # Return best-scored option
        if option_scores:
            return max(option_scores, key=option_scores.get)
        else:
            return current_belief

    def learn_rule(
        self,
        condition: str,
        action: str,
        confidence: float = 0.7
    ) -> int:
        """
        Explicitly learn a rule.

        Args:
            condition: Condition description
            action: Action/outcome description
            confidence: Confidence in rule [0, 1]

        Returns:
            Rule ID
        """
        return self.knowledge_base.add_rule(
            rule_type=RuleType.IF_THEN,
            condition=condition,
            action=action,
            confidence=confidence
        )

    def store_fact(
        self,
        content: str,
        confidence: float = 0.8
    ) -> int:
        """
        Explicitly store a fact.

        Args:
            content: Fact description
            confidence: Confidence [0, 1]

        Returns:
            Fact ID
        """
        return self.knowledge_base.add_fact(
            content=content,
            confidence=confidence
        )

    def consolidate_memory(self, memory: Dict) -> bool:
        """
        Consolidate temporary memory to long-term.

        Args:
            memory: Memory to consolidate

        Returns:
            True if consolidation successful
        """
        self._consolidation_invocations += 1
        return self.consolidator.consolidate(memory)

    def consolidate_batch(self) -> int:
        """
        Consolidate batch of memories (simulated sleep).

        Returns:
            Number consolidated
        """
        self._consolidation_invocations += 1
        return self.consolidator.consolidate_all()

    def update(self, timestamp: int) -> None:
        """
        Update System 2 state.

        Args:
            timestamp: Current simulation time
        """
        self._timestamp = timestamp

    def get_full_statistics(self) -> Dict:
        """Get comprehensive System 2 statistics."""
        return {
            "timestamp": self._timestamp,
            "reasoning_invocations": self._reasoning_invocations,
            "consolidation_invocations": self._consolidation_invocations,
            "knowledge_base": self.knowledge_base.get_statistics(),
            "conflicts": self.conflict_resolver.get_statistics(),
            "consolidation": self.consolidator.get_statistics(),
        }

    # ===== Phase 5: Goal-Directed Behavior =====

    def set_goals(self, goals: List[Goal]) -> None:
        """
        Set goals for the system to pursue (Phase 5).

        Args:
            goals: List of Goal objects to register
        """
        for goal in goals:
            self._goals[goal.goal_id] = goal
            self._goal_history.append((goal.goal_id, goal.state, self._timestamp))

    def select_goal(
        self,
        policy: Optional[Callable[[List[Goal]], int]] = None,
        beta_curiosity: float = 0.1,
        novelty_scores: Optional[Dict[int, float]] = None
    ) -> Optional[Goal]:
        """
        Select next goal to pursue based on policy or default heuristic (Phase 5).

        Default policy: score = utility + beta_curiosity * novelty

        Args:
            policy: Optional custom policy function f(goals) -> goal_id
            beta_curiosity: Weight on curiosity/novelty component [0, 1]
            novelty_scores: Optional novelty estimates per goal {goal_id: novelty}

        Returns:
            Selected Goal object or None if no goals available
        """
        available_goals = [
            g for g in self._goals.values()
            if g.state in ["idle", "pursuing"]
        ]

        if not available_goals:
            return None

        # Use custom policy if provided
        if policy is not None:
            selected_id = policy(available_goals)
            selected = self._goals.get(selected_id)
            if selected:
                self._current_goal_id = selected_id
                if selected.state == "idle":
                    selected.state = "pursuing"
                    self._goal_history.append((selected_id, "pursuing", self._timestamp))
            return selected

        # Default policy: utility + curiosity
        best_goal = None
        best_score = -1.0

        for goal in available_goals:
            novelty = 0.0
            if novelty_scores and goal.goal_id in novelty_scores:
                novelty = novelty_scores[goal.goal_id]

            score = goal.utility + beta_curiosity * novelty

            if score > best_score:
                best_score = score
                best_goal = goal

        if best_goal:
            self._current_goal_id = best_goal.goal_id
            if best_goal.state == "idle":
                best_goal.state = "pursuing"
                self._goal_history.append((best_goal.goal_id, "pursuing", self._timestamp))

        return best_goal

    def evaluate_goal_progress(
        self,
        goal_id: int,
        current_state: Dict[str, Any]
    ) -> float:
        """
        Evaluate and update progress toward a goal (Phase 5).

        Args:
            goal_id: Goal to evaluate
            current_state: Current system/environment state

        Returns:
            Updated progress value [0, 1]
        """
        if goal_id not in self._goals:
            return 0.0

        goal = self._goals[goal_id]

        # Use reward model if available
        if goal.reward_model is not None:
            try:
                expected_reward = goal.reward_model(current_state)
                # Higher expected reward = closer to goal
                goal.progress = min(1.0, max(0.0, expected_reward))
            except Exception:
                # Fallback: keep current progress
                pass

        # Check for achievement
        if goal.progress >= 0.99:
            goal.state = "achieved"
            self._goal_history.append((goal_id, "achieved", self._timestamp))

        return goal.progress

    def get_current_goal(self) -> Optional[Goal]:
        """
        Get currently selected goal (Phase 5).

        Returns:
            Current Goal or None
        """
        if self._current_goal_id is not None:
            return self._goals.get(self._current_goal_id)
        return None

    def get_all_goals(self) -> List[Goal]:
        """
        Get all registered goals (Phase 5).

        Returns:
            List of all Goal objects
        """
        return list(self._goals.values())

    def mark_goal_failed(self, goal_id: int) -> None:
        """
        Mark a goal as failed (Phase 5).

        Args:
            goal_id: Goal to mark as failed
        """
        if goal_id in self._goals:
            self._goals[goal_id].state = "failed"
            self._goal_history.append((goal_id, "failed", self._timestamp))

    def __repr__(self):
        stats = self.knowledge_base.get_statistics()
        return (
            f"System2(rules={stats['num_rules']}, facts={stats['num_facts']}, "
            f"reasoning={self._reasoning_invocations}, goals={len(self._goals)})"
        )
