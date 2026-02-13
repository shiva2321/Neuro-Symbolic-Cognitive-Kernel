"""
NSCK Explanation Module
Natural language explanation generation for agent decisions.

Converts symbolic reasoning traces into human-readable explanations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations."""
    ACTION = "action"            # Why did you do X?
    PREDICTION = "prediction"    # What will happen if X?
    REJECTION = "rejection"      # Why not Y?
    STATE = "state"              # What do you see?
    GOAL = "goal"                # What are you trying to do?
    COMPARISON = "comparison"    # Why X instead of Y?


@dataclass
class Explanation:
    """A structured explanation."""
    type: ExplanationType
    summary: str  # One-line summary
    details: str  # Full explanation
    confidence: float
    supporting_facts: List[str]
    trace: Dict[str, Any]


class ExplanationTemplates:
    """Natural language templates for explanations."""
    
    # Action explanations
    ACTION_GOAL = "I chose {action} because {goal_relation}"
    ACTION_AVOID = "I chose {action} to avoid {danger}"
    ACTION_RULE = "I chose {action} because the rule says: when {conditions}, do {action}"
    ACTION_FALLBACK = "I chose {action} as a safe fallback because {reason}"
    
    # Rejection explanations
    REJECT_DANGER = "I avoided {action} because it would cause {danger}"
    REJECT_UNSAFE = "I rejected {action} because simulation predicted {outcome}"
    REJECT_CONFLICT = "I rejected {action} due to conflicting rules"
    
    # Prediction explanations
    PREDICT_EFFECT = "If I do {action}, then {effect} will happen"
    PREDICT_CHAIN = "If I do {action}, it leads to {chain}"
    
    # State explanations
    STATE_RELATION = "I see {object} is {relation} to {reference}"
    STATE_DANGER = "There is danger in direction {direction}"
    STATE_GOAL = "The goal ({goal}) is {relation} to me"
    
    # Goal explanations  
    GOAL_PRIMARY = "My primary goal is to {goal}"
    GOAL_IMMEDIATE = "Right now I'm trying to {subgoal}"


class ExplanationGenerator:
    """
    Generates natural language explanations from symbolic traces.
    
    Takes reasoning traces from metacognition, causal reasoning, etc.
    and produces human-readable explanations.
    """
    
    def __init__(self, task_goals: Optional[Dict[str, str]] = None):
        """
        Initialize explanation generator.
        
        Args:
            task_goals: Dict mapping task_tag to goal description
        """
        self.task_goals = task_goals or {
            "snake": "eat food and avoid hitting myself or walls",
            "pong": "return the ball by aligning my paddle with it",
        }
        
        # Action name mappings for readability
        self.action_names = {
            "ACTION_UP": "move up",
            "ACTION_DOWN": "move down",
            "ACTION_LEFT": "move left",
            "ACTION_RIGHT": "move right",
            "ACTION_STAY": "stay still",
            "UP": "move up",
            "DOWN": "move down",
            "LEFT": "move left",
            "RIGHT": "move right",
        }
        
        # Relation descriptions
        self.relation_names = {
            "REL_ABOVE": "above me",
            "REL_BELOW": "below me",
            "REL_LEFT": "to my left",
            "REL_RIGHT": "to my right",
            "BALL_ABOVE": "above my paddle",
            "BALL_BELOW": "below my paddle",
            "BALL_ALIGNED": "aligned with my paddle",
        }
        
        # Danger descriptions
        self.danger_names = {
            "DANGER_UP": "collision above",
            "DANGER_DOWN": "collision below",
            "DANGER_LEFT": "collision to the left",
            "DANGER_RIGHT": "collision to the right",
        }
    
    def _readable_action(self, action: str) -> str:
        """Convert action name to readable form."""
        if not action:
            return "stay"
        return self.action_names.get(action, action.lower().replace("action_", ""))
    
    def _readable_relation(self, relation: str) -> str:
        """Convert relation name to readable form."""
        return self.relation_names.get(relation, relation.lower().replace("rel_", ""))
    
    def explain_action(
        self,
        action: str,
        state: Dict[str, Any],
        task_tag: str,
        trace: Dict[str, Any]
    ) -> Explanation:
        """
        Explain why an action was chosen.
        
        Args:
            action: The action taken
            state: Current game state
            task_tag: Which task
            trace: Reasoning trace from metacognition
            
        Returns:
            Explanation object
        """
        facts = []
        readable_action = self._readable_action(action)
        
        # Check for goal alignment
        goal_relation = self._get_goal_relation(action, state, task_tag)
        if goal_relation:
            summary = f"I {readable_action} because the goal is {goal_relation}"
            facts.append(f"Goal is {goal_relation}")
        # Check for danger avoidance
        elif trace.get("veto"):
            summary = f"I {readable_action} to avoid danger (safety veto active)"
            facts.append("Safety system overrode original choice")
        # Check for rule-based decision
        elif trace.get("layer_used") and "rule" in str(trace.get("layer_used")).lower():
            conditions = trace.get("matched_conditions", ["situation matched"])
            summary = f"I {readable_action} because rule matched: {', '.join(conditions)}"
            facts.extend(conditions)
        # Fallback
        else:
            summary = f"I {readable_action} based on learned patterns"
            facts.append("Pattern recognition from experience")
        
        details = self._build_detailed_explanation(
            action, state, task_tag, trace, facts
        )
        
        return Explanation(
            type=ExplanationType.ACTION,
            summary=summary,
            details=details,
            confidence=trace.get("confidence", 0.5),
            supporting_facts=facts,
            trace=trace
        )
    
    def _get_goal_relation(
        self,
        action: str,
        state: Dict[str, Any],
        task_tag: str
    ) -> Optional[str]:
        """Determine if action is aligned with goal direction."""
        if not action:
            return None # Changed from "STAY" to None as "STAY" is an action, not a relation.
                        # If action is None, it means the agent chose to stay.
                        # The function's purpose is to return a *relation* (e.g., "above me")
                        # if the action aligns with a goal direction.
                        # If no directional action is taken, or if "stay" doesn't align with a specific
                        # directional goal, returning None is appropriate.
        action_core = action.replace("ACTION_", "").upper()
        
        if task_tag == "snake":
            head = state.get("head", (5, 5))
            food = state.get("food", (5, 5))
            
            dx = food[0] - head[0]
            dy = food[1] - head[1]
            
            if action_core == "UP" and dy < 0:
                return "above me"
            elif action_core == "DOWN" and dy > 0:
                return "below me"
            elif action_core == "LEFT" and dx < 0:
                return "to my left"
            elif action_core == "RIGHT" and dx > 0:
                return "to my right"
        
        elif task_tag == "pong":
            ball_y = state.get("ball_y", 15)
            paddle_y = state.get("p1_y", 10)
            paddle_center = paddle_y + 3
            
            if action_core == "UP" and ball_y < paddle_center:
                return "above my paddle"
            elif action_core == "DOWN" and ball_y > paddle_center:
                return "below my paddle"
        
        return None
    
    def _build_detailed_explanation(
        self,
        action: str,
        state: Dict[str, Any],
        task_tag: str,
        trace: Dict[str, Any],
        facts: List[str]
    ) -> str:
        """Build a detailed multi-line explanation."""
        lines = []
        readable_action = self._readable_action(action)
        
        # Opening
        lines.append(f"Decision: {readable_action}")
        
        # Goal context
        lines.append(f"Goal: {self.task_goals.get(task_tag, 'achieve objective')}")
        
        # State summary
        if task_tag == "snake":
            head = state.get("head", (0, 0))
            food = state.get("food", (0, 0))
            body_len = len(state.get("body", []))
            lines.append(f"State: head at {head}, food at {food}, length {body_len}")
        elif task_tag == "pong":
            ball_y = state.get("ball_y", 0)
            paddle_y = state.get("p1_y", 0)
            lines.append(f"State: ball at y={ball_y}, paddle at y={paddle_y}")
        
        # Supporting facts
        if facts:
            lines.append("Reasoning:")
            for fact in facts:
                lines.append(f"  • {fact}")
        
        # Confidence
        conf = trace.get("confidence", 0.5)
        conf_label = "high" if conf > 0.7 else "medium" if conf > 0.4 else "low"
        lines.append(f"Confidence: {conf_label} ({conf:.0%})")
        
        return "\n".join(lines)
    
    def explain_rejection(
        self,
        rejected_action: str,
        chosen_action: str,
        state: Dict[str, Any],
        task_tag: str,
        reason: str = ""
    ) -> Explanation:
        """
        Explain why an action was rejected.
        
        Args:
            rejected_action: The action NOT taken
            chosen_action: The action that was taken
            state: Current state
            task_tag: Which task
            reason: Optional specific reason
            
        Returns:
            Explanation object
        """
        rejected_readable = self._readable_action(rejected_action)
        chosen_readable = self._readable_action(chosen_action)
        
        facts = []
        
        if reason:
            summary = f"I avoided {rejected_readable} because {reason}"
            facts.append(reason)
        else:
            # Infer reason
            if "DANGER" in rejected_action.upper():
                summary = f"I avoided {rejected_readable} due to collision risk"
                facts.append("Danger detected in that direction")
            else:
                summary = f"I chose {chosen_readable} over {rejected_readable}"
                facts.append(f"{chosen_readable} had higher priority")
        
        return Explanation(
            type=ExplanationType.REJECTION,
            summary=summary,
            details=f"Rejected: {rejected_readable}\nChosen: {chosen_readable}\nReason: {reason or 'safety/priority'}",
            confidence=0.8,
            supporting_facts=facts,
            trace={"rejected": rejected_action, "chosen": chosen_action}
        )
    
    def explain_state(
        self,
        state: Dict[str, Any],
        task_tag: str,
        active_predicates: List[str]
    ) -> Explanation:
        """
        Describe what the agent perceives.
        
        Args:
            state: Current state
            task_tag: Which task
            active_predicates: Active symbolic predicates
            
        Returns:
            Explanation describing perception
        """
        perceptions = []
        
        for pred in active_predicates:
            readable = self._readable_relation(pred)
            if readable != pred.lower():
                perceptions.append(readable)
        
        if task_tag == "snake":
            head = state.get("head", (0, 0))
            food = state.get("food", (0, 0))
            body_len = len(state.get("body", []))
            
            perceptions.append(f"my head is at {head}")
            perceptions.append(f"food is at {food}")
            perceptions.append(f"my body has {body_len} segments")
            
        elif task_tag == "pong":
            ball_y = state.get("ball_y", 0)
            paddle_y = state.get("p1_y", 0)
            
            perceptions.append(f"ball is at y={ball_y}")
            perceptions.append(f"my paddle is at y={paddle_y}")
        
        summary = f"I perceive: {', '.join(perceptions[:3])}"
        if len(perceptions) > 3:
            summary += f" and {len(perceptions) - 3} more"
        
        return Explanation(
            type=ExplanationType.STATE,
            summary=summary,
            details="Full perception:\n" + "\n".join(f"• {p}" for p in perceptions),
            confidence=1.0,
            supporting_facts=perceptions,
            trace={"state": state, "predicates": active_predicates}
        )
    
    def explain_goal(self, task_tag: str, immediate_subgoal: Optional[str] = None) -> Explanation:
        """Explain the agent's goal."""
        primary = self.task_goals.get(task_tag, "achieve the objective")
        
        summary = f"My goal is to {primary}"
        details = f"Primary goal: {primary}"
        
        if immediate_subgoal:
            details += f"\nImmediate focus: {immediate_subgoal}"
        
        return Explanation(
            type=ExplanationType.GOAL,
            summary=summary,
            details=details,
            confidence=1.0,
            supporting_facts=[primary],
            trace={"task": task_tag}
        )

    def explain_contrastive(
        self,
        action_taken: str,
        hypothetical_action: str,
        counterfactual: Dict[str, Any],
        task_tag: str
    ) -> Explanation:
        """
        Explain why one action was chosen instead of another.
        
        Args:
            action_taken: Chosen action
            hypothetical_action: Rejected action to compare against
            counterfactual: Diff from CausalReasoner.simulate_counterfactual
            task_tag: Context
        """
        action_readable = self._readable_action(action_taken)
        hypo_readable = self._readable_action(hypothetical_action)
        
        added = counterfactual.get("diff_added", [])
        removed = counterfactual.get("diff_removed", [])
        
        # Determine summary sentiment
        if any(e in added for e in ["DEATH", "FAILURE", "REWARD_NEG"]):
            summary = f"I chose {action_readable} instead of {hypo_readable} to avoid {added[0].lower()}."
        elif any(e in removed for e in ["SUCCESS", "REWARD_POS"]):
            summary = f"I chose {action_readable} over {hypo_readable} because {hypo_readable} would have missed a positive outcome."
        else:
            summary = f"I chose {action_readable} over {hypo_readable} based on predicted outcomes."

        details = f"Contrastive Analysis:\n"
        details += f"  • Taken: {action_readable}\n"
        details += f"  • Hypothetical: {hypo_readable}\n"
        
        if added:
            details += f"\nIf I had {hypo_readable}, I predict it would have ALSO caused:\n"
            for e in added:
                details += f"  • {e.lower()}"
                
        if removed:
            details += f"\nIf I had {hypo_readable}, I predict I would have MISSED these effects:\n"
            for e in removed:
                details += f"  • {e.lower()}"

        return Explanation(
            type=ExplanationType.COMPARISON,
            summary=summary,
            details=details,
            confidence=0.9,
            supporting_facts=added + removed,
            trace=counterfactual
        )
