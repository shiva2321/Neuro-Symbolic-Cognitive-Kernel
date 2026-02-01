"""
NSCK Teaching Interface Module
Universal interface for human-in-the-loop learning.

Supports multiple teaching modes:
- Demonstration: "Watch me do this"
- Correction: "That was wrong, do this instead"
- Naming: "This concept is called X"
- Rules: "When A and B, do C"
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union
from enum import Enum
import hypervec_shim as hypervec_rs
from .persistence import BrainStore, Rule, Concept
from .rule_learner import RuleLearner
from .grounding_verifier import GroundingVerifier


class TeachingMode(Enum):
    """Types of teaching interactions."""
    DEMONSTRATION = "demonstration"  # Teacher shows correct action
    CORRECTION = "correction"        # Teacher corrects wrong action
    NAMING = "naming"                # Teacher names a concept
    RULE = "rule"                    # Teacher defines a rule
    POSITIVE = "positive"            # Teacher gives positive feedback
    NEGATIVE = "negative"            # Teacher gives negative feedback


@dataclass
class TeachingEvent:
    """A single teaching interaction."""
    mode: TeachingMode
    timestamp: float
    task_tag: str
    context: Dict[str, Any]  # State/situation when teaching happened
    
    # Mode-specific fields
    correct_action: Optional[str] = None
    wrong_action: Optional[str] = None
    concept_name: Optional[str] = None
    concept_type: Optional[str] = None
    rule_condition: Optional[List[str]] = None
    rule_consequence: Optional[str] = None
    feedback_reward: float = 0.0


class TeachingInterface:
    """
    Universal interface for teaching the NSCK system.
    
    Accepts various forms of human feedback and converts them
    into learning signals for the cognitive system.
    """
    
    def __init__(
        self,
        rule_learner: RuleLearner,
        verifier: GroundingVerifier,
        store: Optional[BrainStore] = None
    ):
        """
        Initialize teaching interface.
        
        Args:
            rule_learner: Rule learner for inducing from demos
            verifier: Grounding verifier
            store: Persistence store
        """
        self.rule_learner = rule_learner
        self.verifier = verifier
        self.store = store
        
        # Teaching history
        self.history: List[TeachingEvent] = []
        
        # Concept definitions pending grounding
        self.pending_concepts: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks for real-time teaching
        self.on_demonstration: Optional[Callable] = None
        self.on_correction: Optional[Callable] = None
    
    def demonstrate(
        self,
        state: Dict[str, Any],
        correct_action: str,
        task_tag: str
    ):
        """
        Teacher demonstrates the correct action.
        
        This is the most common teaching mode. The system learns
        by observing what action should be taken in this state.
        
        Args:
            state: Current game state
            correct_action: What the teacher did
            task_tag: Which task
        """
        # Normalize action
        action = correct_action.upper()
        if not action.startswith("ACTION_"):
            action = f"ACTION_{action}"
        
        # Record in rule learner (as positive example)
        self.rule_learner.observe(
            state=state,
            action=action,
            reward=1.0,  # Teacher knows best
            task_tag=task_tag,
            outcome="success"
        )
        
        # Log event
        event = TeachingEvent(
            mode=TeachingMode.DEMONSTRATION,
            timestamp=time.time(),
            task_tag=task_tag,
            context=state,
            correct_action=action
        )
        self.history.append(event)
        
        # Trigger callback if registered
        if self.on_demonstration:
            self.on_demonstration(event)
        
        print(f"[TEACH] Demonstration: {action} in {task_tag}")
    
    def correct(
        self,
        state: Dict[str, Any],
        wrong_action: str,
        correct_action: str,
        task_tag: str
    ):
        """
        Teacher corrects a wrong action.
        
        Stronger learning signal than demonstration because it
        explicitly marks what NOT to do.
        
        Args:
            state: State where mistake was made
            wrong_action: What the system did wrong
            correct_action: What should have been done
            task_tag: Which task
        """
        # Normalize
        wrong = wrong_action.upper()
        correct = correct_action.upper()
        if not wrong.startswith("ACTION_"):
            wrong = f"ACTION_{wrong}"
        if not correct.startswith("ACTION_"):
            correct = f"ACTION_{correct}"
        
        # Record wrong action as negative example
        self.rule_learner.observe(
            state=state,
            action=wrong,
            reward=-1.0,
            task_tag=task_tag,
            outcome="failure"
        )
        
        # Record correct action as positive example
        self.rule_learner.observe(
            state=state,
            action=correct,
            reward=1.0,
            task_tag=task_tag,
            outcome="success"
        )
        
        # Log event
        event = TeachingEvent(
            mode=TeachingMode.CORRECTION,
            timestamp=time.time(),
            task_tag=task_tag,
            context=state,
            wrong_action=wrong,
            correct_action=correct
        )
        self.history.append(event)
        
        if self.on_correction:
            self.on_correction(event)
        
        print(f"[TEACH] Correction: {wrong} → {correct} in {task_tag}")
    
    def name_concept(
        self,
        name: str,
        concept_type: str,
        grounding_check: Callable[[Dict], bool],
        situation_hv: Optional[hypervec_rs.HyperVector] = None,
        task_tag: str = "global"
    ):
        """
        Teacher defines a new concept.
        
        The concept must be grounded (have a physical meaning that
        can be verified).
        
        Args:
            name: Concept name (e.g., "NEAR_WALL")
            concept_type: Type (e.g., "RELATION", "DANGER")
            grounding_check: Function to verify concept in state
            situation_hv: Optional prototype hypervector
            task_tag: Which task this applies to
        """
        # Register grounding check
        self.verifier.register_predicate(
            name=name,
            check=grounding_check,
            context=task_tag if task_tag != "global" else None
        )
        
        # Store concept if HV provided
        if situation_hv and self.store:
            concept = Concept(
                id=None,
                name=name,
                hv_bytes=bytes(situation_hv.bits) if hasattr(situation_hv, 'bits') else b'',
                concept_type=concept_type,
                task_tag=task_tag
            )
            self.store.save_concept(concept)
        
        # Log event
        event = TeachingEvent(
            mode=TeachingMode.NAMING,
            timestamp=time.time(),
            task_tag=task_tag,
            context={},
            concept_name=name,
            concept_type=concept_type
        )
        self.history.append(event)
        
        print(f"[TEACH] Named concept: {name} ({concept_type}) in {task_tag}")
    
    def teach_rule(
        self,
        condition: List[str],
        consequence: str,
        task_tag: str,
        priority: int = 1
    ):
        """
        Teacher explicitly defines a rule.
        
        The rule must use grounded predicates.
        
        Args:
            condition: List of predicate names (all must be true)
            consequence: Action to take
            task_tag: Which task
            priority: Rule priority (higher = more important)
        """
        # Validate predicates are grounded
        ungrounded = []
        for pred in condition:
            if pred not in self.verifier.predicate_checks:
                # Check task-specific
                if task_tag in self.verifier.context_predicates:
                    if pred not in self.verifier.context_predicates[task_tag]:
                        ungrounded.append(pred)
                else:
                    ungrounded.append(pred)
        
        if ungrounded:
            print(f"[WARN] Ungrounded predicates in rule: {ungrounded}")
            # Still allow, but mark as unverified
        
        # Normalize action
        action = consequence.upper()
        if not action.startswith("ACTION_"):
            action = f"ACTION_{action}"
        
        # Create rule
        rule = Rule(
            id=None,
            condition=frozenset(condition),
            consequence=action,
            priority=priority,
            source="instructed",  # Human-taught
            task_tag=task_tag,
            scope="task_local",
            support_count=1,
            success_rate=1.0,  # Trusted
            created_at=time.time()
        )
        
        # Store
        if self.store:
            self.store.save_rule(rule)
        
        # Also add to learner's known rules
        self.rule_learner.learned_rules[task_tag].append(rule)
        
        # Log event
        event = TeachingEvent(
            mode=TeachingMode.RULE,
            timestamp=time.time(),
            task_tag=task_tag,
            context={},
            rule_condition=condition,
            rule_consequence=action
        )
        self.history.append(event)
        
        print(f"[TEACH] Rule: {condition} → {action} in {task_tag}")
    
    def give_feedback(
        self,
        state: Dict[str, Any],
        action: str,
        is_positive: bool,
        task_tag: str,
        reward_magnitude: float = 1.0
    ):
        """
        Teacher gives simple positive/negative feedback.
        
        Args:
            state: State where feedback applies
            action: Action that was taken
            is_positive: True = good, False = bad
            task_tag: Which task
            reward_magnitude: How strong the feedback is
        """
        mode = TeachingMode.POSITIVE if is_positive else TeachingMode.NEGATIVE
        reward = reward_magnitude if is_positive else -reward_magnitude
        
        # Normalize action
        act = action.upper()
        if not act.startswith("ACTION_"):
            act = f"ACTION_{act}"
        
        # Record observation
        self.rule_learner.observe(
            state=state,
            action=act,
            reward=reward,
            task_tag=task_tag,
            outcome="success" if is_positive else "failure"
        )
        
        # Log event
        event = TeachingEvent(
            mode=mode,
            timestamp=time.time(),
            task_tag=task_tag,
            context=state,
            correct_action=act if is_positive else None,
            wrong_action=act if not is_positive else None,
            feedback_reward=reward
        )
        self.history.append(event)
        
        symbol = "👍" if is_positive else "👎"
        print(f"[TEACH] Feedback {symbol}: {act} in {task_tag}")
    
    def batch_demonstrations(
        self,
        episodes: List[Dict[str, Any]],
        task_tag: str
    ):
        """
        Batch import demonstrations.
        
        Args:
            episodes: List of {state, action} dicts
            task_tag: Which task
        """
        for ep in episodes:
            if "state" in ep and "action" in ep:
                self.demonstrate(
                    state=ep["state"],
                    correct_action=ep["action"],
                    task_tag=task_tag
                )
    
    def get_teaching_summary(self, task_tag: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of teaching for a task."""
        events = self.history
        if task_tag:
            events = [e for e in events if e.task_tag == task_tag]
        
        by_mode = {}
        for e in events:
            mode = e.mode.value
            by_mode[mode] = by_mode.get(mode, 0) + 1
        
        return {
            "total_events": len(events),
            "by_mode": by_mode,
            "rules_taught": sum(1 for e in events if e.mode == TeachingMode.RULE),
            "concepts_named": sum(1 for e in events if e.mode == TeachingMode.NAMING),
            "corrections": sum(1 for e in events if e.mode == TeachingMode.CORRECTION),
        }
    
    def induce_from_demonstrations(self, task_tag: str) -> int:
        """
        Trigger rule induction from accumulated demonstrations.
        
        Args:
            task_tag: Which task to induce rules for
            
        Returns:
            Number of new rules induced
        """
        new_rules = self.rule_learner.induce_rules(task_tag)
        return len(new_rules)
