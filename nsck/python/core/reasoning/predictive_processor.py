"""
NSCK Predictive Processor — Active Inference / Free-Energy Minimisation
========================================================================
Based on Karl Friston's **Free Energy Principle (FEP)** and the associated
framework of **Active Inference** (Friston et al., 2010-2022).

Theoretical background
-----------------------
The FEP holds that any self-organising system resists entropy by minimising
its *variational free energy* (F), which is an upper bound on **surprisal**:

    F = E_q[ln q(s) - ln p(o,s)]
      = KL[q(s) || p(s|o)] - ln p(o)          (ELBO form)

where:
  * s  = hidden states (beliefs about the world)
  * o  = observations / sensory input
  * q  = the agent's approximate posterior (recognition model)
  * p  = the generative model (what the agent expects)

Minimising F via **perception** → update beliefs q(s) to better explain o.
Minimising F via **action**     → change o to match predictions (act to
                                  confirm beliefs).

In NSCK we implement the perception arm of Active Inference using
VSA hypervectors as the belief state representation:

    prediction  = most likely next concept given current context (HV)
    observation = actual next input (HV from TextKnowledgeLearner / SRL)
    error       = 1 − cosine_similarity(prediction, observation)

    Belief update (Bayesian):
        posterior_hv = alpha * observation_hv ⊕ (1-alpha) * prior_hv
    where alpha = learning_rate * error  (high error → big update).

Connections to other NSCK modules
-----------------------------------
* **SemanticMemory** supplies the generative model: the spreading-activation
  neighbourhood of the current context is the prior prediction.
* **BeliefRevision** handles contradictions; PredictiveProcessor handles
  *surprise* (unexpectedly novel observations).
* **SchemaInducer** benefits: prediction errors are the signal for forming
  new schemas (accommodation).
* **CognitiveState.trace** gets a new ``predictive_processing`` field.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger("nsck.predictive_processor")


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """One prediction–observation cycle."""
    context: str
    predicted_concept: str
    observed_concept: str
    prediction_error: float          # 0 = perfect; 1 = maximally surprising
    free_energy: float
    belief_updated: bool


@dataclass
class PredictiveState:
    """Snapshot returned after each process() call."""
    context: str
    predicted: str
    observed: str
    prediction_error: float
    free_energy: float
    surprise_level: str    # "low" | "medium" | "high"
    belief_updated: bool
    history_len: int


# ---------------------------------------------------------------------------
# PredictiveProcessor
# ---------------------------------------------------------------------------

class PredictiveProcessor:
    """
    Active-Inference-inspired predictive processing engine for NSCK.

    For every new observation the processor:
      1. **Predicts** the most likely concept given the current belief state
         (prior) by querying SemanticMemory's spreading activation.
      2. **Computes prediction error** (PE) = 1 − sim(prediction, observation).
      3. **Updates the belief state** proportionally to PE (large surprise →
         large belief revision).
      4. Records the event in a rolling history for transparency.

    Usage::

        from python.core.reasoning.predictive_processor import PredictiveProcessor
        from python.core.memory.semantic_memory import SemanticMemory

        sem = SemanticMemory()
        pp  = PredictiveProcessor(semantic_memory=sem)

        # As TKL learns new facts, inform the processor:
        pp.process("rain", "flooding")     # context, observation
        pp.process("flooding", "damage")

        state = pp.get_state()
        print(state.surprise_level)        # 'high' first time, 'low' after learning
    """

    def __init__(
        self,
        semantic_memory=None,          # SemanticMemory instance (optional)
        learning_rate: float = 0.15,
        surprise_low: float = 0.25,
        surprise_high: float = 0.65,
        history_size: int = 500,
        min_prior_sim: float = 0.3,    # Threshold to accept a prior prediction
    ):
        """
        Args:
            semantic_memory: SemanticMemory to use as generative model.
                             If None the processor uses a flat prior.
            learning_rate: Base step size for belief updates (0 < lr ≤ 1).
            surprise_low: PE below this → 'low' surprise.
            surprise_high: PE above this → 'high' surprise.
            history_size: Rolling window of PredictionRecord entries.
            min_prior_sim: Min similarity for a SM neighbour to be used as prior.
        """
        self.semantic_memory = semantic_memory
        self.learning_rate = learning_rate
        self.surprise_low = surprise_low
        self.surprise_high = surprise_high
        self.min_prior_sim = min_prior_sim

        # Current belief state as a HyperVector (starts as None → uniform prior)
        self._belief_hv: Optional[Any] = None
        self._current_context: str = ""

        # Rolling prediction history
        self._history: Deque[PredictionRecord] = deque(maxlen=history_size)

        # Running metrics
        self._total_pe: float = 0.0
        self._steps: int = 0

        logger.info(
            "[PredictiveProcessor] init: lr=%.3f, surprise_thresholds=(%.2f, %.2f)",
            learning_rate,
            surprise_low,
            surprise_high,
        )

    # ------------------------------------------------------------------
    # Core prediction–update loop
    # ------------------------------------------------------------------

    def process(self, context: str, observation: str) -> PredictiveState:
        """
        Run one perception cycle: predict → observe → compute PE → update.

        Args:
            context:     The current concept / situation label (string).
            observation: The actually observed concept / input (string).

        Returns:
            A ``PredictiveState`` snapshot.
        """
        self._current_context = context

        # 1. Build prior prediction from generative model (SemanticMemory)
        predicted_concept, prior_hv = self._generate_prior(context)

        # 2. Encode observation as HV
        obs_hv = hypervec_rs.HyperVector(hash(observation) & 0x7FFFFFFF)

        # 3. Compute prediction error (PE) = 1 − sim(prior, obs)
        if prior_hv is not None:
            try:
                sim = float(prior_hv.similarity(obs_hv))
            except Exception:
                try:
                    sim = float(obs_hv.similarity(prior_hv))
                except Exception:
                    sim = 0.5
        else:
            sim = 0.5   # flat prior → moderate surprise

        pe = max(0.0, min(1.0, 1.0 - sim))

        # 4. Variational free energy approximation
        #    F ≈ PE + KL(q||p_prior)
        #    We approximate KL as a small constant when PE is high
        kl_approx = 0.1 * pe
        free_energy = pe + kl_approx

        # 5. Bayesian belief update: posterior = alpha*obs ⊕ (1-alpha)*prior
        alpha = min(1.0, self.learning_rate * (1.0 + pe))
        belief_updated = False
        if self._belief_hv is None:
            self._belief_hv = obs_hv
            belief_updated = True
        elif pe > 0.05:                  # only update if genuinely surprising
            try:
                self._belief_hv = obs_hv.bundle(self._belief_hv)
            except Exception:
                self._belief_hv = obs_hv
            belief_updated = True

        # 6. Record
        record = PredictionRecord(
            context=context,
            predicted_concept=predicted_concept,
            observed_concept=observation,
            prediction_error=pe,
            free_energy=free_energy,
            belief_updated=belief_updated,
        )
        self._history.append(record)
        self._total_pe += pe
        self._steps += 1

        surprise_level = self._classify_surprise(pe)

        logger.debug(
            "[PP] ctx='%s' pred='%s' obs='%s' PE=%.3f FE=%.3f surp='%s'",
            context, predicted_concept, observation, pe, free_energy, surprise_level,
        )

        return PredictiveState(
            context=context,
            predicted=predicted_concept,
            observed=observation,
            prediction_error=pe,
            free_energy=free_energy,
            surprise_level=surprise_level,
            belief_updated=belief_updated,
            history_len=len(self._history),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        """Return current summary statistics of the processor."""
        avg_pe = self._total_pe / max(1, self._steps)
        recent = list(self._history)[-10:]
        recent_pe = (
            sum(r.prediction_error for r in recent) / len(recent)
            if recent else 0.0
        )
        return {
            "steps": self._steps,
            "average_prediction_error": round(avg_pe, 4),
            "recent_prediction_error": round(recent_pe, 4),
            "surprise_level": self._classify_surprise(recent_pe),
            "belief_hv_set": self._belief_hv is not None,
            "history_length": len(self._history),
        }

    def get_history(self, n: int = 20) -> List[Dict]:
        """Return the last n prediction records as dicts."""
        records = list(self._history)[-n:]
        return [
            {
                "context": r.context,
                "predicted": r.predicted_concept,
                "observed": r.observed_concept,
                "prediction_error": round(r.prediction_error, 4),
                "free_energy": round(r.free_energy, 4),
                "belief_updated": r.belief_updated,
            }
            for r in records
        ]

    def average_surprise(self) -> float:
        """Return mean prediction error over all recorded steps."""
        return self._total_pe / max(1, self._steps)

    def reset(self) -> None:
        """Clear belief state and history (e.g. task switch)."""
        self._belief_hv = None
        self._current_context = ""
        self._history.clear()
        self._total_pe = 0.0
        self._steps = 0

    # ------------------------------------------------------------------
    # Generative model
    # ------------------------------------------------------------------

    def _generate_prior(self, context: str) -> Tuple[str, Optional[Any]]:
        """
        Query SemanticMemory to predict the most likely next concept.

        Falls back to a HV seeded from context if SM is unavailable or
        returns no neighbours above threshold.
        """
        if self.semantic_memory is not None:
            try:
                neighbours = self.semantic_memory.query(
                    context, top_k=5, spreading_steps=1
                )
                for concept, sim in neighbours:
                    if (
                        concept != context
                        and sim >= self.min_prior_sim
                    ):
                        hv = self.semantic_memory.concept_hvs.get(concept)
                        if hv is None and hasattr(self.semantic_memory, "_get_hv"):
                            hv = self.semantic_memory._get_hv(concept)
                        if hv is not None:
                            return concept, hv
            except Exception as exc:
                logger.debug("[PP] SM query failed: %s", exc)

        # Flat prior: seed from context name
        prior_hv = hypervec_rs.HyperVector(hash(context) & 0x7FFFFFFF)
        return context, prior_hv

    def _classify_surprise(self, pe: float) -> str:
        if pe < self.surprise_low:
            return "low"
        if pe < self.surprise_high:
            return "medium"
        return "high"
