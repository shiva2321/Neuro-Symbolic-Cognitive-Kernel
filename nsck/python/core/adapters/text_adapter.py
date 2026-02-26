"""
TextAdapter — wraps UniversalInput.ground_text() into a PerceptPacket.
"""
from __future__ import annotations

from typing import Any

from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter


class TextAdapter(ModalityAdapter):
    """Adapts a text string into a :class:`PerceptPacket`.

    Delegates to :class:`~python.core.language.universal_input.UniversalInput`
    for the actual VSA encoding.

    Parameters
    ----------
    universal_input :
        An initialised :class:`UniversalInput` instance from the engine.
    """

    def __init__(self, universal_input: Any) -> None:
        self._ui = universal_input

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        """Encode a text string into a PerceptPacket.

        Parameters
        ----------
        raw_input : str
            The text to encode.
        task_tag : str
            Registered task domain identifier.

        Returns
        -------
        PerceptPacket
        """
        text = str(raw_input)
        situation_hv = self._ui.ground_text(text)

        return PerceptPacket.make(
            modality="text",
            situation_hv=situation_hv,
            active_predicates=frozenset(),
            raw_state={"text": text},
            adapter_name="TextAdapter",
            adapter_trace={"length": len(text)},
        )
