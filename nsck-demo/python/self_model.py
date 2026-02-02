"""
NSCK Self-Model Module
Phase 3.2: Self-Performance Modeling

Tracks the agent's historical performance on tasks to generate calibrated 
confidence estimates. This enables the agent to "know what it knows".
"""
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

import hypervec_shim as hypervec_rs

class SelfModel:
    """
    Maintains a statistical model of the agent's own performance.
    
    Used for:
    1. Metacognition: "Am I likely to succeed at this?"
    2. Curriculum: "Which task do I need to practice?"
    3. Calibration: Tracking difference between confidence and reality.
    4. Identity: VSA representation of Self.
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
        
        # [Phase 3.2] Identity HVe
        self.identity_hv = hypervec_rs.HyperVector(hash("SELF_NSCK_V1") % (2**32))
        
        # [Phase 3.2] Capability Map: action -> success_score
        self.capabilities: Dict[str, float] = defaultdict(lambda: 0.0)
        
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
            
        # Simple frequency-based probability
        # TODO Phase 3.3: Use 'state' for contextual prediction (e.g., "I'm bad at corners")
        return stats["successes"] / stats["attempts"]
    
    def update(self, task_tag: str, action: str, predicted_confidence: float, actual_success: bool, reward: float = 0.0):
        """
        Update model with new experience.
        
        Args:
            task_tag: Task identifier
            action: Action taken
            predicted_confidence: What the agent THOUGHT would happen (0.0-1.0)
            actual_success: Did it actually succeed?
            reward: Reward received
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
