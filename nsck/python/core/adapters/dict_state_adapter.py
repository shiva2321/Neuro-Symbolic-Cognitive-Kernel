"""
DictStateAdapter — wraps existing GroundingVerifier + EpisodicMemory.create_situation_hv()
logic that was previously inline in CognitiveEngine.decide().

This is the default adapter used when a raw dict is passed to decide().
It ensures backward compatibility while cleanly separating perception
from cognition.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import python.core.vsa.hypervec_shim as hypervec_rs

from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter
from python.core.perception.grounding_verifier import GroundingVerifier


class DictStateAdapter(ModalityAdapter):
    """Adapts a raw state ``dict`` into a :class:`PerceptPacket`.

    Wraps the GroundingVerifier + EpisodicMemory.create_situation_hv()
    logic that was previously inline in ``CognitiveEngine.decide()``.

    Parameters
    ----------
    verifier : GroundingVerifier
        Domain-specific grounding verifier.  Pass the one registered
        for the relevant task, or a default ``GroundingVerifier()``.
    episodic_memory :
        Reference to the engine's :class:`EpisodicMemory` so that
        ``create_situation_hv()`` can be called with the same
        arguments as before.
    """

    def __init__(self, verifier: GroundingVerifier, episodic_memory: Any) -> None:
        self._verifier = verifier
        self._episodic_memory = episodic_memory

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        """Encode a raw state dict into a PerceptPacket.

        Parameters
        ----------
        raw_input : dict
            Raw environment / sensor state dictionary.
        task_tag : str
            Registered task domain identifier.

        Returns
        -------
        PerceptPacket
        """
        state: Dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}

        # Replicate the exact logic from decide() lines 457-463
        active_preds = self._verifier.get_active_predicates(state, context=task_tag)
        situation_hv = self._episodic_memory.create_situation_hv(
            state, task_tag, active_preds
        )

        return PerceptPacket.make(
            modality="dict",
            situation_hv=situation_hv,
            active_predicates=frozenset(active_preds),
            raw_state=state,
            adapter_name="DictStateAdapter",
            adapter_trace={"predicate_count": len(active_preds)},
        )
