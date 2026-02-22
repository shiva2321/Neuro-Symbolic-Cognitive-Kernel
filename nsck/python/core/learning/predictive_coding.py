"""
Predictive Coding Layer — top-down prediction meets bottom-up error.

Theoretical grounding
---------------------
Predictive Coding (Rao & Ballard, 1999; Friston, 2005) is a Bayesian
brain theory: at every level of the cognitive hierarchy the system
maintains a *generative model* that predicts lower-level representations.
The difference between prediction and actual input is a *prediction error*
that is propagated upward, while predictions are propagated downward.

This implements a lightweight, graph-native, VSA-compatible version:

  • Each concept in semantic memory carries a *predicted activation*
    (top-down expectation from spreading-activation context).
  • When actual evidence arrives (bottom-up percept), the difference
    is the *prediction error*.
  • Large errors → high surprise → trigger deeper System-2 reasoning
    (see dual-process engine) and stronger learning updates (Hebbian +
    free-energy revision).
  • Small errors → fast System-1 path, no costly update.

Connections to other NSCK modules:
  - Integrates with SemanticMemory.spread_activation()
  - Feeds surprise scores into CognitiveEngine (System-1 vs System-2 gating)
  - Surprise guides CuriosityModule (seek out high-surprise inputs)
  - Free-energy computation is consistent with Friston's active inference

Mathematical note
-----------------
For binary HSVs, the "prediction error" is the Hamming distance between
predicted and observed HVs, normalised by dimension.  For distributional
HVs (bipolar), cosine distance is used (more robust to noise).

Usage
-----
    pc = PredictiveCodingLayer(semantic_memory)
    error = pc.compute_error("apple", observed_hv)
    pc.update_prediction("apple", observed_hv, learning_rate=0.1)
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nsck.predictive_coding")


class PredictiveCodingLayer:
    """
    Lightweight predictive coding layer over a SemanticMemory.

    Maintains per-concept *expected activation* (top-down prediction) as a
    running weighted average of observed HVs, and computes per-observation
    *prediction errors* used to gate learning and arousal.

    All operations are O(D) in HV dimension — trivially fast.
    """

    def __init__(
        self,
        semantic_memory: object,
        decay: float = 0.9,
        error_threshold: float = 0.15,
    ):
        """
        Args:
            semantic_memory:  A SemanticMemory instance (used for read-only
                              access to concept_hvs and spread_activation).
            decay:            Exponential decay weight for the running
                              prediction average (higher = more inertia /
                              slower adaptation).
            error_threshold:  Prediction error above this value is considered
                              *surprising* and triggers System-2 gating.
        """
        self.sm = semantic_memory
        self.decay = decay
        self.error_threshold = error_threshold

        # concept → predicted HV (as numpy float32 array for speed)
        self._predictions: Dict[str, np.ndarray] = {}

        # Running stats for monitoring
        self._error_history: List[float] = []
        self._max_history = 1000

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def predict(self, concept: str) -> Optional[np.ndarray]:
        """
        Return current top-down prediction for *concept*, or None if not yet
        seeded (first observation not yet received).
        """
        return self._predictions.get(concept)

    def compute_error(
        self,
        concept: str,
        observed_hv: object,
    ) -> float:
        """
        Compute the prediction error for *concept* given an observed HV.

        Returns:
            float in [0, 1] — 0 = perfect prediction, 1 = maximum surprise.
        """
        pred = self._predictions.get(concept)
        if pred is None:
            # No prior prediction — maximum uncertainty (surprise = 0.5,
            # not 1.0, because we genuinely had no prior).
            return 0.5

        try:
            obs = np.asarray(getattr(observed_hv, "bits", observed_hv), dtype=np.float32)
        except Exception:
            return 0.5

        if obs.shape != pred.shape:
            return 0.5

        # Use cosine distance for robustness (works for both binary and
        # distributional HVs after converting to float).
        norm_pred = np.linalg.norm(pred)
        norm_obs  = np.linalg.norm(obs)
        if norm_pred < 1e-9 or norm_obs < 1e-9:
            return 0.5

        cosine_sim = float(np.dot(pred, obs) / (norm_pred * norm_obs))
        # Convert cosine similarity [−1, 1] → error [0, 1]
        error = (1.0 - cosine_sim) / 2.0

        self._error_history.append(error)
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)

        return error

    def update_prediction(
        self,
        concept: str,
        observed_hv: object,
        learning_rate: float = 0.1,
    ) -> None:
        """
        Update the stored prediction for *concept* using an exponential
        moving average:

            pred ← decay * pred + (1 - decay) * observed

        The *learning_rate* overrides (1 - decay) when explicitly provided,
        allowing call-site control over plasticity.

        This is equivalent to the *precision-weighted prediction-error
        update* in Friston's free-energy formalism, with a fixed (rather
        than inferred) precision.
        """
        try:
            obs = np.asarray(getattr(observed_hv, "bits", observed_hv), dtype=np.float32)
        except Exception:
            return

        if concept not in self._predictions:
            self._predictions[concept] = obs.copy()
        else:
            alpha = learning_rate
            self._predictions[concept] = (
                (1.0 - alpha) * self._predictions[concept] + alpha * obs
            )

    def is_surprising(self, concept: str, observed_hv: object) -> bool:
        """Return True if the prediction error exceeds *error_threshold*."""
        return self.compute_error(concept, observed_hv) > self.error_threshold

    # ------------------------------------------------------------------
    # Batch / context-driven prediction
    # ------------------------------------------------------------------

    def prime_from_context(
        self,
        context_concepts: List[str],
        target_concept: str,
        steps: int = 2,
    ) -> Optional[np.ndarray]:
        """
        Generate a context-driven top-down prediction for *target_concept*
        by spreading activation from *context_concepts* in SemanticMemory and
        blending the HVs of activated neighbours.

        This models how the brain "pre-activates" expected inputs based on
        prior context — a key mechanism in predictive coding.

        Returns the predicted HV array or None if target is not known.
        """
        try:
            activation = self.sm.spread_activation(context_concepts, steps=steps)
        except Exception:
            return None

        if target_concept not in activation:
            return None

        # Build prediction as a weighted average of nearby concepts' HVs
        weighted_sum: Optional[np.ndarray] = None
        total_weight = 0.0

        for concept, act in activation.items():
            hv_obj = self.sm.concept_hvs.get(concept)
            if hv_obj is None:
                continue
            try:
                bits = np.asarray(getattr(hv_obj, "bits", hv_obj), dtype=np.float32)
            except Exception:
                continue
            w = float(act)
            if weighted_sum is None:
                weighted_sum = w * bits
            else:
                weighted_sum = weighted_sum + w * bits
            total_weight += w

        if weighted_sum is None or total_weight < 1e-9:
            return None

        prediction = weighted_sum / total_weight
        self._predictions[target_concept] = prediction
        return prediction

    # ------------------------------------------------------------------
    # Monitoring / introspection
    # ------------------------------------------------------------------

    def mean_recent_error(self, window: int = 100) -> float:
        """Return the mean prediction error over the last *window* observations."""
        if not self._error_history:
            return 0.5
        recent = self._error_history[-window:]
        return float(sum(recent) / len(recent))

    def surprise_score(self, concept: str, observed_hv: object) -> float:
        """
        Compute a *surprisal* score in natural units (nats).

        Surprisal S = -log P(observed | predicted)

        We approximate P using the Gaussian assumption:
            P ∝ exp(-error² / 2σ²)   where σ = mean_recent_error
        → S = error² / (2 * mean_error²)  (dimensionless ratio)

        This is consistent with information-theoretic surprise (Shannon)
        and the variational free-energy formulation (Friston).
        """
        error = self.compute_error(concept, observed_hv)
        sigma = max(self.mean_recent_error(), 1e-3)
        return (error ** 2) / (2 * sigma ** 2)

    def stats(self) -> Dict:
        """Return monitoring statistics."""
        return {
            "concepts_tracked": len(self._predictions),
            "mean_error": self.mean_recent_error(),
            "error_threshold": self.error_threshold,
            "observations": len(self._error_history),
        }
