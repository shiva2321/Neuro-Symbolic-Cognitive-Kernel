"""
Phase 3.4: Meta-Learning - Learning to Learn

Meta-Learning means the system learns HOW to learn, not just WHAT to learn.

Key Concepts:
1. Adaptive Thresholds: Novelty, reward, error thresholds adapt based on outcomes
2. Intrinsic Motivation: System develops curiosity (wants to learn)
3. Region-Specific Learning: Different brain areas learn at different rates
4. Learning Rate Adaptation: Speed of learning adapts to domain
5. Confidence-Based Control: Actions based on confidence in predictions

Philosophy:
- System 1 (spikes): Learns facts about the world
- System 1.5 (control): Learns when to learn
- System 2 (reasoning): Learns explicit rules
- Meta-Learning: Learns how to learn better

Invariants:
- Thresholds always in valid ranges
- Adaptation based on recent outcomes (EMA)
- System improves over time (better learning decisions)
- Regional specialization without catastrophic forgetting
- Intrinsic motivation complements extrinsic (reward)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class AdaptationSignal(Enum):
    """Signals that drive meta-learning."""
    SUCCESS = "success"          # Learning led to good outcome
    FAILURE = "failure"          # Learning led to poor outcome
    STAGNATION = "stagnation"    # No improvement
    BREAKTHROUGH = "breakthrough" # Major improvement
    CONFLICT = "conflict"        # Conflicting signals


@dataclass
class ThresholdAdaptation:
    """Adaptive threshold for a learning signal."""
    name: str                     # "novelty", "reward", "error"
    current_value: float          # Current threshold [0, 1]
    min_value: float = 0.1        # Minimum allowed
    max_value: float = 0.9        # Maximum allowed
    adaptation_rate: float = 0.01 # How fast it adapts
    success_count: int = 0        # Times it led to success
    failure_count: int = 0        # Times it led to failure

    def adapt(self, signal: AdaptationSignal) -> float:
        """
        Adapt threshold based on outcome signal.

        Args:
            signal: Whether learning decision was good/bad

        Returns:
            New threshold value
        """
        if signal == AdaptationSignal.SUCCESS:
            self.success_count += 1
            # Success: slightly lower threshold (learn more)
            delta = -self.adaptation_rate * 0.1
        elif signal == AdaptationSignal.FAILURE:
            self.failure_count += 1
            # Failure: slightly raise threshold (learn less)
            delta = self.adaptation_rate * 0.1
        elif signal == AdaptationSignal.BREAKTHROUGH:
            self.success_count += 3  # Major success
            # Breakthrough: significantly lower threshold
            delta = -self.adaptation_rate * 0.3
        elif signal == AdaptationSignal.STAGNATION:
            # Stagnation: lower threshold to encourage exploration
            delta = -self.adaptation_rate * 0.05
        else:  # CONFLICT
            # Conflict: be conservative
            delta = self.adaptation_rate * 0.05

        self.current_value = max(
            self.min_value,
            min(self.max_value, self.current_value + delta)
        )

        return self.current_value

    def get_success_rate(self) -> float:
        """Get success rate of this threshold."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Neutral
        return self.success_count / total


@dataclass
class RegionalLearningProfile:
    """Learning profile for a specific region."""
    region_id: int
    learning_rate: float = 0.01       # Base STDP learning rate
    plasticity_enabled: bool = True
    specialization: str = "general"   # "sensory", "motor", "memory", etc.

    # Adaptive properties
    activity_level: float = 0.0       # Recent firing rate
    performance: float = 0.5          # Recent performance (0-1)
    adaptation_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def update_learning_rate(self, performance_delta: float) -> float:
        """
        Adapt learning rate based on performance change.

        Args:
            performance_delta: Change in performance (-1 to +1)

        Returns:
            New learning rate
        """
        # Increase learning rate if improving, decrease if regressing
        factor = 1.0 + (performance_delta * 0.1)
        self.learning_rate = max(0.0001, min(0.5, self.learning_rate * factor))

        self.adaptation_history.append(self.learning_rate)
        return self.learning_rate

    def get_mean_learning_rate(self) -> float:
        """Get mean learning rate over recent history."""
        if not self.adaptation_history:
            return self.learning_rate
        return sum(self.adaptation_history) / len(self.adaptation_history)


class IntrinsicMotivation:
    """
    Models intrinsic motivation (curiosity, exploration drive).

    The system wants to learn new things, even without external reward.
    Complementary to extrinsic reward signals.
    """

    def __init__(self):
        """Initialize intrinsic motivation system."""
        self._curiosity_level = 0.5    # Base curiosity [0, 1]
        self._explored_patterns: set = set()
        self._novelty_appetite = 0.5   # How much system wants novelty
        self._exploration_bonus: List[float] = []

    def compute_intrinsic_reward(self, novelty: float, confidence: float) -> float:
        """
        Compute intrinsic reward based on novelty and uncertainty.

        High intrinsic reward when:
        - Pattern is novel (haven't seen before)
        - Prediction is uncertain (want to understand)

        Args:
            novelty: Novelty score [0, 1]
            confidence: Confidence in prediction [0, 1]

        Returns:
            Intrinsic reward [0, 1]
        """
        # Curiosity = novelty + uncertainty
        uncertainty = 1.0 - confidence
        intrinsic = (novelty * 0.6) + (uncertainty * 0.4)

        # Scale by curiosity level
        intrinsic = intrinsic * self._curiosity_level

        self._exploration_bonus.append(intrinsic)
        if len(self._exploration_bonus) > 100:
            self._exploration_bonus.pop(0)

        return min(intrinsic, 1.0)

    def record_pattern(self, pattern_id: str) -> None:
        """Record that a pattern was explored."""
        self._explored_patterns.add(pattern_id)

    def adapt_curiosity(self, learning_progress: float) -> float:
        """
        Adapt curiosity level based on learning progress.

        Args:
            learning_progress: How well learning is going [0, 1]

        Returns:
            New curiosity level
        """
        # If learning well: slightly increase curiosity (explore more)
        # If learning poorly: maintain or decrease curiosity (focus)
        if learning_progress > 0.7:
            self._curiosity_level = min(1.0, self._curiosity_level + 0.05)
        elif learning_progress < 0.3:
            self._curiosity_level = max(0.1, self._curiosity_level - 0.05)

        return self._curiosity_level

    def get_statistics(self) -> Dict:
        """Get intrinsic motivation statistics."""
        return {
            "curiosity_level": self._curiosity_level,
            "patterns_explored": len(self._explored_patterns),
            "mean_exploration_bonus": sum(self._exploration_bonus) / len(self._exploration_bonus) if self._exploration_bonus else 0.0,
        }


class MetaLearner:
    """
    Meta-Learning system that learns how to learn.

    Adapts:
    - Plasticity thresholds (novelty, reward, error)
    - Region-specific learning rates
    - Intrinsic motivation
    - Decision-making strategy
    """

    def __init__(self):
        """Initialize meta-learner."""
        # Adaptive thresholds
        self._novelty_threshold = ThresholdAdaptation("novelty", 0.5)
        self._reward_threshold = ThresholdAdaptation("reward", 0.1)
        self._error_threshold = ThresholdAdaptation("error", 0.3)

        # Regional learning profiles
        self._regional_profiles: Dict[int, RegionalLearningProfile] = {}

        # Intrinsic motivation
        self._intrinsic_motivation = IntrinsicMotivation()

        # Performance tracking
        self._performance_history: deque = deque(maxlen=100)
        self._learning_decisions: List[Tuple[str, bool]] = []  # (decision, outcome)
        self._timestep = 0

    def add_region(
        self,
        region_id: int,
        specialization: str = "general",
        initial_learning_rate: float = 0.01
    ) -> None:
        """
        Add a region for meta-learning.

        Args:
            region_id: ID of region
            specialization: Type of region
            initial_learning_rate: Starting learning rate
        """
        profile = RegionalLearningProfile(
            region_id=region_id,
            learning_rate=initial_learning_rate,
            specialization=specialization
        )
        self._regional_profiles[region_id] = profile

    def get_novelty_threshold(self) -> float:
        """Get current novelty threshold."""
        return self._novelty_threshold.current_value

    def get_reward_threshold(self) -> float:
        """Get current reward threshold."""
        return self._reward_threshold.current_value

    def get_error_threshold(self) -> float:
        """Get current error threshold."""
        return self._error_threshold.current_value

    def get_region_learning_rate(self, region_id: int) -> float:
        """Get learning rate for specific region."""
        if region_id in self._regional_profiles:
            return self._regional_profiles[region_id].learning_rate
        return 0.01

    def record_learning_outcome(
        self,
        decision: str,
        outcome: bool,
        signal: AdaptationSignal,
        performance_change: float = 0.0
    ) -> None:
        """
        Record outcome of a learning decision.

        Args:
            decision: What learning decision was made
            outcome: Whether it was beneficial
            signal: Type of adaptation signal
            performance_change: Change in performance
        """
        self._timestep += 1

        # Record decision
        self._learning_decisions.append((decision, outcome))

        # Adapt thresholds based on signal
        self._novelty_threshold.adapt(signal)
        self._reward_threshold.adapt(signal)
        self._error_threshold.adapt(signal)

        # Update regional learning rates if applicable
        if performance_change != 0.0:
            for profile in self._regional_profiles.values():
                profile.update_learning_rate(performance_change)

        # Track performance
        self._performance_history.append(1.0 if outcome else 0.0)

    def compute_combined_intrinsic_extrinsic(
        self,
        extrinsic_reward: float,
        novelty: float,
        confidence: float
    ) -> float:
        """
        Combine intrinsic and extrinsic rewards.

        Args:
            extrinsic_reward: External reward signal
            novelty: Novelty score
            confidence: Confidence in prediction

        Returns:
            Combined reward signal
        """
        intrinsic = self._intrinsic_motivation.compute_intrinsic_reward(
            novelty, confidence
        )

        # Combine: extrinsic + intrinsic (when no extrinsic signal)
        combined = extrinsic_reward + (intrinsic * (1.0 - extrinsic_reward))

        return min(combined, 1.0)

    def get_full_statistics(self) -> Dict:
        """Get comprehensive meta-learning statistics."""
        return {
            "timestep": self._timestep,
            "thresholds": {
                "novelty": {
                    "value": self._novelty_threshold.current_value,
                    "success_rate": self._novelty_threshold.get_success_rate(),
                },
                "reward": {
                    "value": self._reward_threshold.current_value,
                    "success_rate": self._reward_threshold.get_success_rate(),
                },
                "error": {
                    "value": self._error_threshold.current_value,
                    "success_rate": self._error_threshold.get_success_rate(),
                },
            },
            "performance": {
                "current": self._performance_history[-1] if self._performance_history else 0.5,
                "mean": sum(self._performance_history) / len(self._performance_history) if self._performance_history else 0.5,
            },
            "regions": {
                rid: {
                    "learning_rate": profile.learning_rate,
                    "mean_learning_rate": profile.get_mean_learning_rate(),
                    "specialization": profile.specialization,
                }
                for rid, profile in self._regional_profiles.items()
            },
            "intrinsic_motivation": self._intrinsic_motivation.get_statistics(),
            "total_decisions": len(self._learning_decisions),
            "decisions_successful": sum(1 for _, outcome in self._learning_decisions if outcome),
        }

    def __repr__(self):
        success_rate = sum(1 for _, o in self._learning_decisions if o) / max(len(self._learning_decisions), 1)
        return (
            f"MetaLearner(novelty={self._novelty_threshold.current_value:.2f}, "
            f"regions={len(self._regional_profiles)}, "
            f"success_rate={success_rate:.1%})"
        )

    def update(self, hyperparams: Dict, signals: Dict) -> Dict:
        """
        Update meta-learning hyperparameters based on signals.

        Contract:
        - Inputs: hyperparams (dict of current params), signals dict with keys
          {'confidence': float[0..1], 'novelty': float[0..1], 'error': float[0..1], 'stability': float[0..1], 'performance_delta': float[-1..1]}
        - Output: updated hyperparams dict (may include 'stdp_lr', 'region_lr_multiplier', 'novelty_threshold', 'reward_threshold', 'error_threshold')
        - Pure: does not mutate inputs; internal state may track stats.
        """
        confidence = float(signals.get('confidence', 0.5))
        novelty = float(signals.get('novelty', 0.0))
        error = float(signals.get('error', 0.0))
        stability = float(signals.get('stability', 0.5))
        perf_delta = float(signals.get('performance_delta', 0.0))

        # Adapt thresholds via existing helpers, using heuristic mapping of signals
        # Success heuristic: high confidence & low error
        if confidence > 0.7 and error < 0.3:
            sig = AdaptationSignal.SUCCESS
        elif error > 0.7:
            sig = AdaptationSignal.FAILURE
        elif novelty > 0.7 and confidence < 0.5:
            sig = AdaptationSignal.BREAKTHROUGH
        elif abs(perf_delta) < 0.05:
            sig = AdaptationSignal.STAGNATION
        else:
            sig = AdaptationSignal.CONFLICT

        # Record threshold adaptation (decision label is generic here)
        self.record_learning_outcome(
            decision="meta_update", outcome=(sig in (AdaptationSignal.SUCCESS, AdaptationSignal.BREAKTHROUGH)), signal=sig, performance_change=perf_delta
        )

        # Compute adapted learning rate multiplier: reduce when unstable
        region_lr_multiplier = max(0.2, min(2.0, 1.0 + perf_delta - (stability * 0.3)))

        updated = dict(hyperparams)
        updated['stdp_lr'] = self.get_region_learning_rate(hyperparams.get('region_id', -1))
        updated['region_lr_multiplier'] = region_lr_multiplier
        updated['novelty_threshold'] = self.get_novelty_threshold()
        updated['reward_threshold'] = self.get_reward_threshold()
        updated['error_threshold'] = self.get_error_threshold()

        return updated

    def get_signals_schema(self) -> Dict[str, str]:
        """Describe expected signals for update()."""
        return {
            'confidence': 'float[0..1] prediction confidence',
            'novelty': 'float[0..1] novelty score',
            'error': 'float[0..1] prediction/TD error proxy',
            'stability': 'float[0..1] stability index (higher=stable)',
            'performance_delta': 'float[-1..1] recent performance change',
        }
