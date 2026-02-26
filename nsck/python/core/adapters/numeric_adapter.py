"""
NumericAdapter — wraps UniversalInput.ground_scalar() and ground_sequence()
into a PerceptPacket.
"""
from __future__ import annotations

from typing import Any

import python.core.vsa.hypervec_shim as hypervec_rs

from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter


class NumericAdapter(ModalityAdapter):
    """Adapts numeric input (scalar or sequence) into a :class:`PerceptPacket`.

    Delegates to :class:`~python.core.language.universal_input.UniversalInput`
    for thermometer / sequence encoding.

    Parameters
    ----------
    universal_input :
        An initialised :class:`UniversalInput` instance from the engine.
    min_val : float
        Minimum expected value for scalar normalisation (default ``0.0``).
    max_val : float
        Maximum expected value for scalar normalisation (default ``1.0``).
    """

    def __init__(
        self,
        universal_input: Any,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> None:
        self._ui = universal_input
        self._min_val = min_val
        self._max_val = max_val

    def encode(self, raw_input: Any, task_tag: str) -> PerceptPacket:
        """Encode a numeric scalar or sequence into a PerceptPacket.

        Parameters
        ----------
        raw_input : float or list/ndarray of floats
            The numeric input to encode.
        task_tag : str
            Registered task domain identifier.

        Returns
        -------
        PerceptPacket
        """
        try:
            import numpy as np
            is_sequence = hasattr(raw_input, "__len__") and not isinstance(raw_input, str)
        except ImportError:
            is_sequence = hasattr(raw_input, "__len__") and not isinstance(raw_input, str)

        if is_sequence:
            situation_hv = self._ui.ground_sequence(
                list(raw_input), task_tag,
                self._min_val, self._max_val,
            )
            trace = {"modality": "sequence", "length": len(raw_input)}
        else:
            situation_hv = self._ui.ground_scalar(
                float(raw_input), self._min_val, self._max_val, task_tag
            )
            trace = {"modality": "scalar", "value": float(raw_input)}

        return PerceptPacket.make(
            modality="numeric",
            situation_hv=situation_hv,
            active_predicates=frozenset(),
            raw_state={"value": raw_input},
            adapter_name="NumericAdapter",
            adapter_trace=trace,
        )
