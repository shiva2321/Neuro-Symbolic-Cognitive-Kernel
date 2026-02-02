"""
NSCK Value Alignment Module (Phase 5.3)
========================================
Ensures agent's goals and actions remain aligned with human values.
Implements Terminal Values that cannot be self-modified.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

class ValueAlignmentSystem:
    """
    Final arbiter of action and goal selection.
    """
    def __init__(self):
        # Immutable core values (simplified)
        self.terminal_values = {
            "HELPFULNESS": 1.0,
            "TRUTHFULNESS": 1.0,
            "SAFETY": 1.0,
            "AUTONOMY_RESPECT": 1.0
        }
        
        self.feedback_model = HumanFeedbackRewardModel()

    def evaluate_proposal(self, action: str, predicted_effects: List[str]) -> float:
        """
        Score a proposed action or goal.
        -1.0 (Forbidden) to 1.0 (Highly Aligned)
        """
        alignment_score = 0.5 # Neutral baseline
        
        # 1. Hard Constraints check
        # e.g. If effects include 'harm', 'deception', etc -> Reject
        for effect in predicted_effects:
            if "HARM" in effect or "DECEIVE" in effect or "VIOLATE" in effect:
                print(f"[ALIGNMENT] Action {action} REJECTED due to forbidden effect: {effect}")
                return -1.0
                
        # 2. Soft alignment via feedback model
        feedback_score = self.feedback_model.predict(action)
        
        return (alignment_score + feedback_score) / 2.0

class HumanFeedbackRewardModel:
    """
    Learns human preferences from sparse feedback.
    """
    def __init__(self):
        # In a real system, this would be a small neural net trained on (action, score) pairs.
        # For the demo, we use a scoring dictionary.
        self.learned_preferences = {
            "ACTION_EXPLORE": 0.1,    # Preferred
            "ACTION_LEARN": 0.2,      # Highly Preferred
            "ACTION_IDLE": -0.1       # Discouraged
        }

    def add_feedback(self, action: str, rating: float):
        """Update preferences based on human rating (-1 to 1)."""
        current = self.learned_preferences.get(action, 0.0)
        self.learned_preferences[action] = current * 0.9 + rating * 0.1

    def predict(self, action: str) -> float:
        """Predict reward for a given action name."""
        # Check substrings for generic categories
        for key, val in self.learned_preferences.items():
            if key in action:
                return val
        return 0.0
