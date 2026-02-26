"""
ModalityAdapter — abstract base class for all perception adapters.

Every concrete adapter converts a raw modality-specific input into a
PerceptPacket so that the cognitive core is always modality-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from python.core.types.percept_packet import PerceptPacket


class ModalityAdapter(ABC):
    """Abstract base class for modality adapters.

    Subclasses implement ``encode()`` to convert raw input into a
    :class:`~python.core.types.percept_packet.PerceptPacket`.

    The adapter pattern ensures that perception logic lives entirely
    outside the cognitive reasoning core.
    """

    @abstractmethod
    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        """Convert *raw_input* into a PerceptPacket for *task_tag*.

        Parameters
        ----------
        raw_input :
            Arbitrary input — dict, str, number, ndarray, etc.
        task_tag : str
            The registered task domain identifier.

        Returns
        -------
        PerceptPacket
            Fully populated percept ready for the cognitive engine.
        """
