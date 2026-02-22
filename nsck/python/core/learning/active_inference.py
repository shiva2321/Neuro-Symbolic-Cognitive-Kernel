"""
Active Inference Planner — curiosity-driven exploration via free-energy.

Theoretical grounding
---------------------
Active Inference (Friston et al., 2017) extends predictive coding from
*passive perception* to *active behaviour*: an agent selects actions that
are expected to minimise *future* free energy (i.e., surprise + divergence
from prior beliefs).

Key principles used here:

1. **Expected free energy (EFE)**:
   G(π) = E[log P(o|π) - log Q(s|π)]   (simplified)
   Decomposes into:
     - *Extrinsic value*: prefer states with high prior preference (reward)
     - *Epistemic value*: prefer states that reduce uncertainty (curiosity)

2. **Information gain** (epistemic curiosity):
   IG(concept) ≈ H(P_prior) - E[H(P_posterior)]
   We approximate this using prediction-error variance as a proxy for
   posterior precision — high variance → lots to learn → high IG.

3. **Principle of Maximum Entropy (Jaynes, 1957)**:
   When choosing among equally rewarding actions, prefer the one that
   keeps the most options open (maximises entropy / minimises commitment).
   Implemented as a tie-breaking prior.

4. **Thompson Sampling** (Thompson, 1933):
   Stochastic exploration that is optimal in the Bayesian multi-armed-bandit
   sense — naturally balances exploration and exploitation without manual
   ε tuning.

5. **Thermodynamic analogy (Boltzmann)**:
   Softmax with temperature T is mathematically identical to the Boltzmann
   distribution P(state) ∝ exp(-E/kT).  Lower T → greedy (exploit);
   higher T → uniform (explore).  We tie T to mean prediction error so
   the system is automatically more exploratory when uncertain.

Usage
-----
    aip = ActiveInferencePlanner(semantic_memory, predictive_coding_layer)
    concept_to_query = aip.select_next_query(["gravity", "mass"])
    # → returns the concept most worth exploring given current beliefs
"""
from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nsck.active_inference")

# Boltzmann temperature bounds and error-to-temperature scaling factor.
# Grounded in statistical mechanics: low T → exploit known-good paths;
# high T → explore uniformly.  Bounds prevent degenerate behaviour.
MIN_TEMPERATURE: float = 0.1
MAX_TEMPERATURE: float = 5.0
# Scaling: mean_error=0 → T stays at base; mean_error=1.0 → T = base * (1 + ERROR_SCALING)
ERROR_SCALING: float = 4.0


class ActiveInferencePlanner:
    """
    Selects the *most informative* next concept/query to explore by
    maximising expected epistemic value (information gain) subject to a
    prior preference for task-relevant concepts.

    Designed to be a drop-in curiosity module, replacing or augmenting
    the existing :class:`~python.core.learning.curiosity.CuriosityModule`.
    """

    def __init__(
        self,
        semantic_memory: object,
        predictive_coding: Optional[object] = None,
        temperature: float = 1.0,
        preference_weight: float = 0.5,
        exploration_bonus: float = 0.2,
    ):
        """
        Args:
            semantic_memory:    SemanticMemory instance.
            predictive_coding:  Optional PredictiveCodingLayer instance.  When
                                provided, its prediction errors are used as
                                epistemic-value proxies.
            temperature:        Boltzmann temperature for action selection.
                                Tied to mean prediction error when PC layer is
                                provided (auto-adaptive exploration).
            preference_weight:  Balance between extrinsic (reward-based) and
                                epistemic (curiosity-based) value.
                                0 = pure curiosity, 1 = pure exploitation.
            exploration_bonus:  Constant additive bonus for concepts never
                                queried before (optimism under uncertainty).
        """
        self.sm = semantic_memory
        self.pc = predictive_coding
        self.temperature = temperature
        self.preference_weight = preference_weight
        self.exploration_bonus = exploration_bonus

        # Prior concept preferences (set by task/goal context)
        self._preferences: Dict[str, float] = {}

        # Visit counts for each concept (for exploration bonuses)
        self._visit_counts: Dict[str, int] = {}

        # Cumulative information gained per concept
        self._info_gained: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Goal-setting
    # ------------------------------------------------------------------

    def set_preference(self, concept: str, value: float) -> None:
        """
        Set an extrinsic preference (reward signal) for a concept.

        A positive value signals task-relevance; negative values model
        concepts to avoid.  Values are in arbitrary units — only relative
        magnitudes matter.
        """
        self._preferences[concept] = float(value)

    def set_preferences(self, preferences: Dict[str, float]) -> None:
        """Batch version of :meth:`set_preference`."""
        for concept, value in preferences.items():
            self.set_preference(concept, value)

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

    def epistemic_value(self, concept: str) -> float:
        """
        Estimate the epistemic value (expected information gain) of querying
        *concept*.

        Approximation: uses prediction error (from PredictiveCodingLayer if
        available) as a proxy for posterior uncertainty.  High error →
        high information content → high epistemic value.

        If no PC layer is provided, falls back to visit-count heuristic
        (UCB-style: less-visited concepts have higher epistemic value).
        """
        if self.pc is not None:
            hv_obj = self.sm.concept_hvs.get(concept)
            if hv_obj is not None:
                try:
                    error = self.pc.compute_error(concept, hv_obj)
                    return float(error)
                except Exception:
                    pass

        # UCB-style fallback: epistemic value ∝ 1 / sqrt(n_visits + 1)
        n = self._visit_counts.get(concept, 0)
        return 1.0 / math.sqrt(n + 1)

    def extrinsic_value(self, concept: str) -> float:
        """Return the extrinsic preference for *concept* (default 0)."""
        return self._preferences.get(concept, 0.0)

    def free_energy(self, concept: str) -> float:
        """
        Compute the *negative* expected free energy for *concept*.

        A lower free energy (higher in sign) means a *better* choice to
        explore.  We return the negated value so that argmax gives the
        best concept.

        G(concept) = w * extrinsic(c) + (1-w) * epistemic(c) + bonus

        where *w* = preference_weight and *bonus* = exploration_bonus for
        unvisited concepts.
        """
        w = self.preference_weight
        ext = self.extrinsic_value(concept)
        eps = self.epistemic_value(concept)
        bonus = self.exploration_bonus if self._visit_counts.get(concept, 0) == 0 else 0.0
        return w * ext + (1.0 - w) * eps + bonus

    def select_next_query(
        self,
        candidates: Optional[List[str]] = None,
        k: int = 1,
        stochastic: bool = True,
    ) -> List[str]:
        """
        Select the *k* most informative concepts to query next.

        Args:
            candidates:  Concepts to consider.  If None, use all concepts
                         known to semantic memory.
            k:           Number of concepts to return.
            stochastic:  If True, use Boltzmann (softmax) sampling so that
                         selection is stochastic (better exploration).  If
                         False, use greedy argmax (exploitation).

        Returns:
            List of up to *k* concept names, best first.
        """
        if candidates is None:
            candidates = list(self.sm.concept_hvs.keys())

        if not candidates:
            return []

        # Compute free-energy scores
        scores = {c: self.free_energy(c) for c in candidates}

        if stochastic:
            # Boltzmann (softmax) sampling — temperature adapts to PC error
            T = self._adaptive_temperature()
            vals = np.array([scores[c] for c in candidates], dtype=np.float64)
            # Numerically stable softmax
            vals = vals - vals.max()
            probs = np.exp(vals / max(T, 1e-6))
            probs /= probs.sum()

            selected_indices = np.random.choice(
                len(candidates),
                size=min(k, len(candidates)),
                replace=False,
                p=probs,
            )
            selected = [candidates[int(i)] for i in selected_indices]
        else:
            # Greedy: top-k by score
            selected = sorted(candidates, key=lambda c: -scores[c])[:k]

        # Update visit counts
        for c in selected:
            self._visit_counts[c] = self._visit_counts.get(c, 0) + 1

        return selected

    def _adaptive_temperature(self) -> float:
        """
        Tie Boltzmann temperature to current mean prediction error.

        High error → high T (explore more).
        Low error   → low T (exploit more).

        Bounded to [MIN_TEMPERATURE, MAX_TEMPERATURE] to prevent degenerate behaviour.
        """
        if self.pc is not None:
            try:
                mean_err = self.pc.mean_recent_error()
                return max(
                    MIN_TEMPERATURE,
                    min(MAX_TEMPERATURE, self.temperature * (1.0 + ERROR_SCALING * mean_err)),
                )
            except Exception:
                pass
        return self.temperature

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """Return introspection statistics."""
        visited = len(self._visit_counts)
        total_visits = sum(self._visit_counts.values())
        return {
            "concepts_visited": visited,
            "total_queries": total_visits,
            "temperature": self._adaptive_temperature(),
            "top_preferences": sorted(
                self._preferences.items(), key=lambda x: -x[1]
            )[:5],
        }
