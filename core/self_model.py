"""
Self-Model: Phase 5 outcome prediction and self-understanding.

The self-model allows the system to predict the outcomes of its own actions,
track prediction errors, and improve its understanding of cause-and-effect.

Philosophy:
- Simple linear/shallow model (MVP for Phase 5)
- Tracks prediction residual variance
- Updates from actual outcomes
- Can be extended to deeper models in Phase 6+
"""

from typing import List, Tuple, Dict, Optional
from collections import deque
import math


class SelfModel:
    """
    Predicts outcomes of actions for self-understanding (Phase 5).

    Uses a simple linear model to predict next state given current state and action.
    Tracks prediction error variance to estimate uncertainty.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, num_actions: int = 4):
        """
        Initialize self-model.

        Args:
            input_dim: Dimension of state representation
            hidden_dim: Hidden layer dimension (not used in linear model MVP)
            num_actions: Number of possible actions
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Simple linear model: next_state = W * [state, action_onehot]
        # For MVP, use simple averaging/heuristics
        self._prediction_history: deque = deque(maxlen=100)
        self._residual_history: deque = deque(maxlen=100)
        self._update_count = 0

        # Simple learned model: track state transitions
        self._transition_memory: Dict[Tuple, List[float]] = {}  # (state_hash, action) -> [next_states]

    def predict_outcome(
        self,
        state: List[float],
        action: int
    ) -> Tuple[List[float], float]:
        """
        Predict next state and uncertainty given current state and action.

        Args:
            state: Current state vector
            action: Action to take

        Returns:
            (predicted_next_state, uncertainty)
            - predicted_next_state: Predicted state vector
            - uncertainty: Prediction uncertainty [0, 1]
        """
        # Hash state for lookup (simple discretization)
        state_hash = self._hash_state(state)
        key = (state_hash, action)

        # If we've seen this (state, action) before, use average of outcomes
        if key in self._transition_memory and self._transition_memory[key]:
            outcomes = self._transition_memory[key]
            # Average the outcomes
            predicted = [sum(s[i] for s in outcomes) / len(outcomes) for i in range(len(outcomes[0]))]

            # Uncertainty based on variance of outcomes
            if len(outcomes) > 1:
                variance = sum(
                    sum((s[i] - predicted[i]) ** 2 for i in range(len(s)))
                    for s in outcomes
                ) / len(outcomes)
                uncertainty = min(1.0, variance)
            else:
                uncertainty = 0.5  # Moderate uncertainty for single observation
        else:
            # Novel (state, action) - high uncertainty, use simple heuristic
            predicted = state.copy()  # Default: state doesn't change
            uncertainty = 1.0  # High uncertainty

        return predicted, uncertainty

    def update(
        self,
        state: List[float],
        action: int,
        actual_next_state: List[float]
    ) -> float:
        """
        Learn from actual outcome.

        Args:
            state: State before action
            action: Action taken
            actual_next_state: Actual resulting state

        Returns:
            Prediction error (residual)
        """
        # Get prediction
        predicted, uncertainty = self.predict_outcome(state, action)

        # Compute residual error (MSE)
        if len(predicted) != len(actual_next_state):
            predicted = actual_next_state.copy()  # Handle dimension mismatch

        residual = self.compute_residual_error(predicted, actual_next_state)

        # Store outcome in transition memory
        state_hash = self._hash_state(state)
        key = (state_hash, action)

        if key not in self._transition_memory:
            self._transition_memory[key] = []
        self._transition_memory[key].append(actual_next_state.copy())

        # Track history
        self._prediction_history.append((state, action, actual_next_state))
        self._residual_history.append(residual)
        self._update_count += 1

        return residual

    def compute_residual_error(
        self,
        prediction: List[float],
        actual: List[float]
    ) -> float:
        """
        Compute MSE between prediction and actual.

        Args:
            prediction: Predicted state
            actual: Actual state

        Returns:
            Mean squared error
        """
        if len(prediction) != len(actual):
            return 1.0  # Maximum error for dimension mismatch

        squared_errors = [(p - a) ** 2 for p, a in zip(prediction, actual)]
        mse = sum(squared_errors) / max(len(squared_errors), 1)
        return mse

    def get_residual_variance(self) -> float:
        """
        Get prediction error variance over recent history.

        Returns:
            Variance of residual errors
        """
        if len(self._residual_history) < 2:
            return 1.0  # High variance initially

        residuals = list(self._residual_history)
        mean_residual = sum(residuals) / len(residuals)
        variance = sum((r - mean_residual) ** 2 for r in residuals) / len(residuals)

        return variance

    def get_mean_residual(self) -> float:
        """
        Get mean prediction error over recent history.

        Returns:
            Mean residual error
        """
        if not self._residual_history:
            return 1.0

        return sum(self._residual_history) / len(self._residual_history)

    def get_statistics(self) -> Dict:
        """Get self-model statistics."""
        return {
            "update_count": self._update_count,
            "mean_residual": self.get_mean_residual(),
            "residual_variance": self.get_residual_variance(),
            "num_transitions_learned": len(self._transition_memory),
            "prediction_history_size": len(self._prediction_history),
        }

    def _hash_state(self, state: List[float], bins: int = 5) -> Tuple:
        """
        Hash a continuous state to discrete bins for lookup.

        Args:
            state: State vector
            bins: Number of bins per dimension

        Returns:
            Hashable tuple
        """
        # Simple discretization: bin each dimension
        discretized = tuple(int(s * bins) for s in state)
        return discretized

    def clear(self) -> None:
        """Clear all learned transitions and history."""
        self._prediction_history.clear()
        self._residual_history.clear()
        self._transition_memory.clear()
        self._update_count = 0
