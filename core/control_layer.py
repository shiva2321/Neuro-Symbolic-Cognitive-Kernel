"""
Control Layer: Cognitive control for neural substrate (System 1.5).

This layer sits between pure spike propagation (System 1) and symbolic reasoning (System 2).
It monitors the neural substrate and makes decisions about when and how learning should occur.

Key Components:
1. NoveltyDetector - Detects out-of-distribution patterns
2. ConfidenceEstimator - Measures output certainty
3. ConflictMonitor - Identifies competing spike patterns
4. PlasticityGateController - Context-aware learning gate

Philosophy:
- Learning is OFF by default
- Learning turns ON only when:
  * Novelty detected
  * Reward present
  * Prediction error high
  * Explicit learning mode

This prevents:
- Catastrophic forgetting
- Weight chaos
- Unstable learning
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import math


@dataclass
class LearningContext:
    """
    Context information for plasticity gating decisions.

    Attributes:
        novelty: Novelty score [0, 1] (higher = more novel)
        confidence: Confidence score [0, 1] (higher = more confident)
        conflict: Conflict score [0, 1] (higher = more conflict)
        reward: Reward signal (0 = none, >0 = positive)
        prediction_error: Prediction error magnitude
        explicit_learning_mode: Manual override
    """
    novelty: float = 0.0
    confidence: float = 1.0
    conflict: float = 0.0
    reward: float = 0.0
    prediction_error: float = 0.0
    explicit_learning_mode: bool = False

    def should_learn(
        self,
        novelty_threshold: float = 0.5,
        reward_threshold: float = 0.1,
        error_threshold: float = 0.3
    ) -> bool:
        """
        Decide if learning should be active based on context.

        Args:
            novelty_threshold: Minimum novelty to trigger learning
            reward_threshold: Minimum reward to trigger learning
            error_threshold: Minimum prediction error to trigger learning

        Returns:
            True if learning should be enabled
        """
        if self.explicit_learning_mode:
            return True
        if self.novelty > novelty_threshold:
            return True
        if self.reward > reward_threshold:
            return True
        if self.prediction_error > error_threshold:
            return True
        return False


class NoveltyDetector:
    """
    Detects novel (out-of-distribution) spike patterns.

    Uses exponential moving average of spike patterns to build a baseline.
    Novelty = distance from expected pattern.

    Invariant: Novelty score always in [0, 1]
    """

    def __init__(self, decay_rate: float = 0.99, history_size: int = 100):
        """
        Initialize novelty detector.

        Args:
            decay_rate: EMA decay for expected patterns (0.99 = slow adaptation)
            history_size: Number of recent patterns to track
        """
        self.decay_rate = decay_rate
        self.history_size = history_size

        # Track expected firing patterns
        self._expected_pattern: Dict[int, float] = defaultdict(float)  # neuron_id -> firing probability
        self._recent_patterns: deque = deque(maxlen=history_size)
        self._pattern_count = 0

        # Novelty statistics
        self._novelty_history: deque = deque(maxlen=100)
        self._peak_novelty = 0.0

    def update(self, firing_neurons: Set[int], all_neurons: Set[int]) -> float:
        """
        Update expected patterns and compute novelty.

        Args:
            firing_neurons: Set of neuron IDs that fired this timestep
            all_neurons: Set of all neuron IDs in network

        Returns:
            Novelty score [0, 1]
        """
        self._pattern_count += 1

        # Compute novelty before updating baseline
        novelty = self._compute_novelty(firing_neurons, all_neurons)

        # Update expected pattern with EMA
        for neuron_id in all_neurons:
            is_firing = 1.0 if neuron_id in firing_neurons else 0.0
            self._expected_pattern[neuron_id] = (
                self.decay_rate * self._expected_pattern[neuron_id] +
                (1 - self.decay_rate) * is_firing
            )

        # Track history
        self._recent_patterns.append(firing_neurons.copy())
        self._novelty_history.append(novelty)
        self._peak_novelty = max(self._peak_novelty, novelty)

        return novelty

    def _compute_novelty(self, firing_neurons: Set[int], all_neurons: Set[int]) -> float:
        """
        Compute novelty as prediction error.

        Novelty = mean absolute deviation from expected pattern
        """
        if self._pattern_count == 0:
            return 1.0  # Everything is novel initially

        total_error = 0.0
        for neuron_id in all_neurons:
            expected = self._expected_pattern.get(neuron_id, 0.0)
            actual = 1.0 if neuron_id in firing_neurons else 0.0
            total_error += abs(expected - actual)

        # Normalize by number of neurons
        novelty = total_error / max(len(all_neurons), 1)
        return min(novelty, 1.0)  # Clamp to [0, 1]

    def get_statistics(self) -> Dict:
        """Get novelty statistics."""
        return {
            "current_novelty": self._novelty_history[-1] if self._novelty_history else 0.0,
            "mean_novelty": sum(self._novelty_history) / len(self._novelty_history) if self._novelty_history else 0.0,
            "peak_novelty": self._peak_novelty,
            "pattern_count": self._pattern_count,
        }


class ConfidenceEstimator:
    """
    Estimates confidence in network outputs.

    Confidence = consistency of output firing patterns over recent timesteps.
    High confidence = stable output
    Low confidence = unstable / uncertain output

    Invariant: Confidence score always in [0, 1]
    """

    def __init__(self, window_size: int = 10):
        """
        Initialize confidence estimator.

        Args:
            window_size: Number of timesteps to consider for stability
        """
        self.window_size = window_size

        # Track output patterns
        self._output_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._confidence_history: deque = deque(maxlen=100)

    def update(self, output_neurons: Dict[int, bool]) -> float:
        """
        Update output history and compute confidence.

        Args:
            output_neurons: Dict mapping output neuron_id -> is_firing

        Returns:
            Confidence score [0, 1]
        """
        # Record outputs
        for neuron_id, is_firing in output_neurons.items():
            self._output_history[neuron_id].append(1.0 if is_firing else 0.0)

        # Compute confidence as inverse of variance
        confidence = self._compute_confidence(output_neurons.keys())
        self._confidence_history.append(confidence)

        return confidence

    def _compute_confidence(self, output_ids: Set[int]) -> float:
        """
        Compute confidence as stability of outputs.

        Confidence = 1 - mean_variance across output neurons
        """
        if not output_ids:
            return 1.0  # No outputs = fully confident (nothing to be uncertain about)

        variances = []
        for neuron_id in output_ids:
            history = self._output_history[neuron_id]
            if len(history) < 2:
                variances.append(0.0)  # Not enough data
            else:
                mean_val = sum(history) / len(history)
                variance = sum((x - mean_val) ** 2 for x in history) / len(history)
                variances.append(variance)

        mean_variance = sum(variances) / len(variances) if variances else 0.0
        confidence = 1.0 - mean_variance
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

    def get_statistics(self) -> Dict:
        """Get confidence statistics."""
        return {
            "current_confidence": self._confidence_history[-1] if self._confidence_history else 1.0,
            "mean_confidence": sum(self._confidence_history) / len(self._confidence_history) if self._confidence_history else 1.0,
            "min_confidence": min(self._confidence_history) if self._confidence_history else 1.0,
        }


class ConflictMonitor:
    """
    Monitors for conflicting spike patterns.

    Conflict = multiple competing output patterns with similar strength.
    High conflict = ambiguous input requiring System 2 reasoning.

    Invariant: Conflict score always in [0, 1]
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize conflict monitor.

        Args:
            window_size: Number of timesteps to detect conflict over
        """
        self.window_size = window_size

        # Track output patterns
        self._output_history: deque = deque(maxlen=window_size)
        self._conflict_history: deque = deque(maxlen=100)
        self._conflict_events = 0

    def update(self, output_neurons: Dict[int, bool]) -> float:
        """
        Update output history and compute conflict.

        Args:
            output_neurons: Dict mapping output neuron_id -> is_firing

        Returns:
            Conflict score [0, 1]
        """
        # Record current pattern
        firing_pattern = frozenset(nid for nid, firing in output_neurons.items() if firing)
        self._output_history.append(firing_pattern)

        # Compute conflict
        conflict = self._compute_conflict()
        self._conflict_history.append(conflict)

        if conflict > 0.5:
            self._conflict_events += 1

        return conflict

    def _compute_conflict(self) -> float:
        """
        Compute conflict as pattern diversity.

        Conflict = entropy of output patterns (normalized)
        High entropy = many different patterns = conflict
        """
        if len(self._output_history) < 2:
            return 0.0  # Not enough data

        # Count unique patterns
        pattern_counts = defaultdict(int)
        for pattern in self._output_history:
            pattern_counts[pattern] += 1

        # Compute entropy
        total = len(self._output_history)
        entropy = 0.0
        for count in pattern_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)

        # Normalize by max possible entropy
        max_entropy = math.log2(total)
        conflict = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(conflict, 1.0)  # Clamp to [0, 1]

    def get_statistics(self) -> Dict:
        """Get conflict statistics."""
        return {
            "current_conflict": self._conflict_history[-1] if self._conflict_history else 0.0,
            "mean_conflict": sum(self._conflict_history) / len(self._conflict_history) if self._conflict_history else 0.0,
            "conflict_events": self._conflict_events,
        }


class PlasticityGateController:
    """
    Context-aware plasticity gating.

    Extends the basic LearningGate to make context-sensitive decisions.
    Learning is OFF by default, turned ON only when context warrants it.

    This is the "System 1.5" decision layer.
    """

    def __init__(
        self,
        novelty_threshold: float = 0.5,
        reward_threshold: float = 0.1,
        error_threshold: float = 0.3,
        default_learning: bool = False
    ):
        """
        Initialize plasticity gate controller.

        Args:
            novelty_threshold: Minimum novelty to open gate
            reward_threshold: Minimum reward to open gate
            error_threshold: Minimum prediction error to open gate
            default_learning: Default state (should be False for Phase 3)
        """
        self.novelty_threshold = novelty_threshold
        self.reward_threshold = reward_threshold
        self.error_threshold = error_threshold

        self._learning_enabled = default_learning
        self._gate_open_count = 0
        self._gate_close_count = 0
        self._last_context: Optional[LearningContext] = None

    def update(self, context: LearningContext) -> bool:
        """
        Update gate state based on context.

        Args:
            context: Current learning context

        Returns:
            True if learning should be enabled
        """
        self._last_context = context

        should_learn = context.should_learn(
            self.novelty_threshold,
            self.reward_threshold,
            self.error_threshold
        )

        # Track state changes
        if should_learn and not self._learning_enabled:
            self._gate_open_count += 1
        elif not should_learn and self._learning_enabled:
            self._gate_close_count += 1

        self._learning_enabled = should_learn
        return self._learning_enabled

    def is_learning_active(self) -> bool:
        """Check if learning is currently enabled."""
        return self._learning_enabled

    def force_enable(self) -> None:
        """Force learning ON (manual override)."""
        self._learning_enabled = True

    def force_disable(self) -> None:
        """Force learning OFF (manual override)."""
        self._learning_enabled = False

    def get_statistics(self) -> Dict:
        """Get gating statistics."""
        return {
            "learning_enabled": self._learning_enabled,
            "gate_open_count": self._gate_open_count,
            "gate_close_count": self._gate_close_count,
            "last_context": self._last_context,
        }


class ControlLayer:
    """
    Unified control layer (System 1.5).

    Integrates:
    - NoveltyDetector
    - ConfidenceEstimator
    - ConflictMonitor
    - PlasticityGateController

    This is the "brain" that watches the "neurons".
    """

    def __init__(
        self,
        novelty_threshold: float = 0.5,
        reward_threshold: float = 0.1,
        error_threshold: float = 0.3,
        default_learning: bool = False,
        beta_blend: float = 0.5  # Phase 5: reward blending weight
    ):
        """
        Initialize control layer.

        Args:
            novelty_threshold: Novelty threshold for plasticity gating
            reward_threshold: Reward threshold for plasticity gating
            error_threshold: Error threshold for plasticity gating
            default_learning: Default learning state (should be False)
            beta_blend: Phase 5 reward blending weight [0, 1] (0=all extrinsic, 1=all intrinsic)
        """
        self.novelty_detector = NoveltyDetector()
        self.confidence_estimator = ConfidenceEstimator()
        self.conflict_monitor = ConflictMonitor()
        self.plasticity_gate = PlasticityGateController(
            novelty_threshold=novelty_threshold,
            reward_threshold=reward_threshold,
            error_threshold=error_threshold,
            default_learning=default_learning
        )

        self._timestep = 0
        self._system2_invocations = 0

        # Phase 5: Intrinsic motivation
        self.beta_blend = beta_blend

    def update(
        self,
        firing_neurons: Set[int],
        all_neurons: Set[int],
        output_neurons: Dict[int, bool],
        reward: float = 0.0,
        prediction_error: float = 0.0
    ) -> LearningContext:
        """
        Update control layer and compute learning context.

        Args:
            firing_neurons: Set of neurons that fired this timestep
            all_neurons: Set of all neuron IDs
            output_neurons: Dict of output neuron states
            reward: Reward signal
            prediction_error: Prediction error magnitude

        Returns:
            LearningContext with gating decision
        """
        self._timestep += 1

        # Compute metrics
        novelty = self.novelty_detector.update(firing_neurons, all_neurons)
        confidence = self.confidence_estimator.update(output_neurons)
        conflict = self.conflict_monitor.update(output_neurons)

        # Build context
        context = LearningContext(
            novelty=novelty,
            confidence=confidence,
            conflict=conflict,
            reward=reward,
            prediction_error=prediction_error,
            explicit_learning_mode=False
        )

        # Update plasticity gate
        self.plasticity_gate.update(context)

        # Check if System 2 should be invoked
        if conflict > 0.7 and confidence < 0.3:
            self._system2_invocations += 1

        return context

    def is_learning_active(self) -> bool:
        """Check if learning is currently enabled."""
        return self.plasticity_gate.is_learning_active()

    def force_learning(self, enabled: bool) -> None:
        """Force learning state (manual override)."""
        if enabled:
            self.plasticity_gate.force_enable()
        else:
            self.plasticity_gate.force_disable()

    def blend_rewards(
        self,
        extrinsic: float,
        intrinsic: float,
        beta: Optional[float] = None
    ) -> float:
        """
        Blend extrinsic and intrinsic rewards (Phase 5).

        Combines external rewards (from environment) with internal rewards
        (from curiosity/novelty) using weighted average.

        Formula: blended = extrinsic * (1 - β) + intrinsic * β

        Args:
            extrinsic: External reward signal [0, 1]
            intrinsic: Internal reward signal (curiosity) [0, 1]
            beta: Optional override for blend weight [0, 1]. If None, uses self.beta_blend

        Returns:
            Blended reward [0, 1]
        """
        beta_to_use = beta if beta is not None else self.beta_blend
        beta_to_use = max(0.0, min(1.0, beta_to_use))  # Clamp to [0, 1]

        blended = extrinsic * (1.0 - beta_to_use) + intrinsic * beta_to_use
        return blended

    def get_full_statistics(self) -> Dict:
        """Get all control layer statistics."""
        return {
            "timestep": self._timestep,
            "novelty": self.novelty_detector.get_statistics(),
            "confidence": self.confidence_estimator.get_statistics(),
            "conflict": self.conflict_monitor.get_statistics(),
            "plasticity_gate": self.plasticity_gate.get_statistics(),
            "system2_invocations": self._system2_invocations,
            "beta_blend": self.beta_blend,  # Phase 5
        }

    def __repr__(self):
        learning_state = "ON" if self.is_learning_active() else "OFF"
        return f"ControlLayer(learning={learning_state}, timestep={self._timestep})"
