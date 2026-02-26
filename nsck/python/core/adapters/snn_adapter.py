"""
SNNAdapter — wraps the SNN perception logic from perceive_and_decide()
into a PerceptPacket, so the SNN pathway uses the same contract.
"""
from __future__ import annotations

from typing import Any, Optional

import python.core.vsa.hypervec_shim as hypervec_rs

from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter


class SNNAdapter(ModalityAdapter):
    """Adapts a raw sensory array via the SNN perception module.

    Wraps the logic previously inline in
    ``CognitiveEngine.perceive_and_decide()`` (lines 366–427) into the
    PerceptPacket contract.

    Parameters
    ----------
    perception :
        An initialised :class:`~python.core.perception.snn_perception.SNNPerceptionModule`
        (or ``None`` if SNN is unavailable).
    episodic_memory :
        Reference to the engine's EpisodicMemory for ``create_situation_hv``.
    verifier :
        GroundingVerifier for the task (may produce empty predicates for
        raw SNN signals — that is acceptable).
    """

    def __init__(
        self,
        perception: Any,
        episodic_memory: Any,
        verifier: Any,
        perception_call_counter: Optional[list] = None,
    ) -> None:
        self._perception = perception
        self._episodic = episodic_memory
        self._verifier = verifier
        # Mutable counter shared with the engine for lazy-STDP scheduling
        self._counter: list = perception_call_counter if perception_call_counter is not None else [0]

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        """Encode a raw sensory ndarray via SNN into a PerceptPacket.

        Parameters
        ----------
        raw_input : np.ndarray
            Sensory input vector.
        task_tag : str
            Registered task domain identifier.

        Returns
        -------
        PerceptPacket
        """
        import numpy as np

        sensory_input = raw_input
        snn_result = None

        if self._perception:
            self._counter[0] += 1
            learn_this_step = (self._counter[0] % 5 == 0)
            snn_result = self._perception.perceive(sensory_input, learn=learn_this_step)

        # Build state dict (mirrors perceive_and_decide lines 381-404)
        state: dict = {"snn_active": True}
        if sensory_input is not None and hasattr(sensory_input, "__len__") and len(sensory_input) > 0:
            arr = np.asarray(sensory_input, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            max_val = float(np.max(arr))
            state["signal_mean"] = mean_val
            state["signal_std"] = std_val
            state["signal_max"] = max_val
            state["high_activity"] = int(mean_val > 0.6)
            state["low_activity"] = int(mean_val < 0.2)
            state["noisy"] = int(std_val > 0.4)
            if snn_result:
                concept_id = snn_result.get("concept_id", 0)
                state["concept_id"] = concept_id
                state["snn_strength"] = float(snn_result.get("strength", 0.0))
                n_spikes = snn_result.get("n_spikes", 0)
                state["active_firing"] = int(n_spikes > 10)
                state["sparse_firing"] = int(n_spikes <= 10)
                if self._perception and hasattr(self._perception, "concept_mapper"):
                    label = self._perception.concept_mapper.get_concept_label(concept_id)
                    state["concept_label"] = label

        active_preds = self._verifier.get_active_predicates(state, context=task_tag)
        situation_hv = self._episodic.create_situation_hv(state, task_tag, active_preds)

        # Resolve concept name for trace
        concept_name = "unknown"
        confidence = 0.5
        if snn_result:
            concept_id = snn_result.get("concept_id", 0)
            if self._perception and hasattr(self._perception, "concept_mapper"):
                concept_name = self._perception.concept_mapper.get_concept_label(concept_id)
            else:
                concept_name = f"Percept_{concept_id}"
            confidence = float(snn_result.get("strength", 0.5))

        return PerceptPacket.make(
            modality="snn",
            situation_hv=situation_hv,
            active_predicates=frozenset(active_preds),
            confidence=confidence,
            raw_state=state,
            adapter_name="SNNAdapter",
            adapter_trace={
                "concept_name": concept_name,
                "snn_result": snn_result or {},
            },
        )
