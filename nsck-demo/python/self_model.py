"""
NSCK Self-Model Module
Phase 3.2 + 3.3: Self-Performance Modeling with Context-Aware Prediction

Tracks the agent's historical performance on tasks to generate calibrated 
confidence estimates. This enables the agent to "know what it knows".

Phase 3.3 Enhancement: Context-aware prediction uses state features
(e.g., action history, situation tags) to provide situational confidence
rather than a flat task-level average.
"""
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

import hypervec_shim as hypervec_rs


def _extract_context_key(state: Any) -> Optional[str]:
    """
    Extract a hashable context key from a state object.

    Supports dicts (looks for 'situation' or 'context' keys),
    strings, and falls back to str() for other types.
    """
    if state is None:
        return None
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        # Prefer explicit situation / context tags
        for tag in ("situation", "context", "position", "zone"):
            if tag in state:
                return str(state[tag])
        return None
    return str(state)


class SelfModel:
    """
    Maintains a statistical model of the agent's own performance.
    
    Used for:
    1. Metacognition: "Am I likely to succeed at this?"
    2. Curriculum: "Which task do I need to practice?"
    3. Calibration: Tracking difference between confidence and reality.
    4. Identity: VSA representation of Self.
    5. Context-aware prediction: "I'm bad at corners" (Phase 3.3).
    """
    
    def __init__(self):
        # task_tag -> stats dict
        self.task_stats = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "total_reward": 0.0,
            "confidence_errors": []  # List of (predicted, actual_outcome_float)
        })
        self.current_confidence = defaultdict(lambda: 0.5)
        
        # [Phase 3.2] Identity HV
        self.identity_hv = hypervec_rs.HyperVector(hash("SELF_NSCK_V1") % (2**32))
        
        # [Phase 3.2] Capability Map: action -> success_score
        self.capabilities: Dict[str, float] = defaultdict(lambda: 0.0)

        # [Phase 3.3] Context-specific performance tracking
        # (task_tag, context_key) -> {attempts, successes}
        self.context_stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0}
        )

        # [Phase 3.3] Recent performance window for trend detection
        self.recent_window: Dict[str, List[bool]] = defaultdict(list)
        self.RECENT_WINDOW_SIZE = 20
        
        print("SelfModel Initialized with Identity HV.")

    def update_confidence(self, task_tag: str, confidence: float):
        """Update current confidence level based on active module."""
        self.current_confidence[task_tag] = confidence

    def get_confidence(self, task_tag: str) -> float:
        """Get current confidence level."""
        return self.current_confidence[task_tag]
        
    def predict_success(self, task_tag: str, state: Any = None) -> float:
        """
        Predict probability of success on this task.
        
        Phase 3.3: When *state* is provided, the prediction blends
        task-level base rate with context-specific performance data,
        giving a situational confidence estimate (e.g., "I'm bad at corners").
        
        Args:
            task_tag: Task identifier (e.g., 'snake')
            state: Optional current state (for context-aware prediction)
            
        Returns:
            Probability of success (0.0 - 1.0)
        """
        stats = self.task_stats[task_tag]
        
        # Cold start problem: return uncertain
        if stats["attempts"] < 10:
            return 0.5
            
        # Base rate: overall frequency-based probability
        base_rate = stats["successes"] / stats["attempts"]

        # Phase 3.3: Context-aware adjustment
        ctx_key = _extract_context_key(state)
        if ctx_key is not None:
            ctx = self.context_stats.get((task_tag, ctx_key))
            if ctx and ctx["attempts"] >= 3:
                ctx_rate = ctx["successes"] / ctx["attempts"]
                # Blend: weight context more as evidence accumulates (max 60%)
                ctx_weight = min(0.6, ctx["attempts"] / 50.0)
                return (1.0 - ctx_weight) * base_rate + ctx_weight * ctx_rate

        # Trend adjustment from recent window
        recent = self.recent_window.get(task_tag)
        if recent and len(recent) >= 5:
            recent_rate = sum(recent) / len(recent)
            # Light blend (20%) toward recent trend
            return 0.8 * base_rate + 0.2 * recent_rate

        return base_rate
    
    def update(self, task_tag: str, predicted_confidence: float, actual_success: bool, action: str = "UNKNOWN", reward: float = 0.0, state: Any = None):
        """
        Update model with new experience.
        
        Args:
            task_tag: Task identifier
            predicted_confidence: What the agent THOUGHT would happen (0.0-1.0)
            actual_success: Did it actually succeed?
            action: Action taken (optional, for capability tracking)
            reward: Reward received
            state: Optional state context for context-aware tracking (Phase 3.3)
        """
        stats = self.task_stats[task_tag]
        
        stats["attempts"] += 1
        stats["total_reward"] += reward
        
        if actual_success:
            stats["successes"] += 1
            # Boost capability for this action
            self.capabilities[action] = min(1.0, self.capabilities[action] + 0.1)
        else:
            # Drop capability slightly
            self.capabilities[action] = max(0.0, self.capabilities[action] - 0.05)
            
        # Record calibration error
        actual_val = 1.0 if actual_success else 0.0
        stats["confidence_errors"].append((predicted_confidence, actual_val))
        
        # Keep history finite
        if len(stats["confidence_errors"]) > 1000:
            stats["confidence_errors"].pop(0)

        # Phase 3.3: Update context-specific stats
        ctx_key = _extract_context_key(state)
        if ctx_key is not None:
            ctx = self.context_stats[(task_tag, ctx_key)]
            ctx["attempts"] += 1
            if actual_success:
                ctx["successes"] += 1

        # Phase 3.3: Update recent window for trend detection
        window = self.recent_window[task_tag]
        window.append(actual_success)
        if len(window) > self.RECENT_WINDOW_SIZE:
            window.pop(0)

    def get_identity(self) -> hypervec_rs.HyperVector:
        """Get VSA identity of self."""
        return self.identity_hv

    def get_capability(self, action: str) -> float:
        """Get current proficiency estimate for an action."""
        return self.capabilities[action]

    def get_calibration_error(self, task_tag: str) -> float:
        """
        Compute Expected Calibration Error (ECE).
        Lower is better (means confidence matches reality).
        """
        stats = self.task_stats[task_tag]
        if not stats["confidence_errors"]:
            return 0.0
            
        # Simple Mean Absolute Error for now
        total_error = sum(abs(pred - actual) for pred, actual in stats["confidence_errors"])
        return total_error / len(stats["confidence_errors"])

    def get_stats(self, task_tag: str) -> Dict[str, Any]:
        """Return raw stats for a task."""
        return dict(self.task_stats[task_tag])

    def get_context_performance(self, task_tag: str) -> Dict[str, Dict[str, Any]]:
        """
        Return context-specific performance breakdown for a task.
        
        Useful for identifying weak spots (e.g., low success in 'corner'
        situations while overall task performance is average).
        """
        result = {}
        for (t, ctx), stats in self.context_stats.items():
            if t == task_tag and stats["attempts"] > 0:
                result[ctx] = {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": stats["successes"] / stats["attempts"],
                }
        return result

    def get_improvement_trend(self, task_tag: str) -> Optional[str]:
        """
        Detect whether the agent is improving, declining, or stable on a task.
        
        Returns:
            'improving', 'declining', or 'stable' (or None if insufficient data).
        """
        window = self.recent_window.get(task_tag)
        if not window or len(window) < 10:
            return None
        mid = len(window) // 2
        first_half = sum(window[:mid]) / mid
        second_half = sum(window[mid:]) / (len(window) - mid)
        diff = second_half - first_half
        if diff > 0.15:
            return "improving"
        elif diff < -0.15:
            return "declining"
        return "stable"
