"""
NumericSequenceAdapter — encodes 1D numeric sequences (lists/arrays of numbers)
as PerceptPackets using TimeSeriesEncoder (NSCK V11).
"""
from __future__ import annotations

from typing import Any, List, Optional

import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter


class NumericSequenceAdapter(ModalityAdapter):
    """Adapter for 1D numeric sequences (lists/arrays of numbers).

    Uses :class:`~python.core.perception.stream_encoder.TimeSeriesEncoder`
    for FPE-based HV encoding and statistical predicate extraction.
    """

    def __init__(
        self,
        value_range: tuple = (-10.0, 10.0),
        channel_name: str = "values",
    ) -> None:
        self._value_range = value_range
        self._channel_name = channel_name
        # Lazy import to avoid circular dependencies
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from python.core.perception.stream_encoder import TimeSeriesEncoder
            self._encoder = TimeSeriesEncoder(value_range=self._value_range)
        return self._encoder

    def encode(self, values: Any, task_tag: str, channel_name: Optional[str] = None) -> PerceptPacket:
        """Encode a list/array of numbers as a PerceptPacket using TimeSeriesEncoder.

        Parameters
        ----------
        values : list or np.ndarray
            Numeric sequence to encode.
        task_tag : str
            Task domain identifier.
        channel_name : str, optional
            Override the default channel name.

        Returns
        -------
        PerceptPacket with modality="numeric_sequence".
        """
        channel = channel_name or self._channel_name
        enc = self._get_encoder()

        # Normalize to list of floats
        try:
            vals = [float(v) for v in values]
        except (TypeError, ValueError):
            vals = []

        situation_hv = enc.encode_sequence(vals)
        feats = enc.encode_features(vals)
        preds = enc.features_to_predicates(feats, channel=channel)

        return PerceptPacket.make(
            modality="numeric_sequence",
            situation_hv=situation_hv,
            active_predicates=frozenset(preds),
            raw_state={"values": vals, "channel": channel},
            adapter_name="NumericSequenceAdapter",
            adapter_trace={
                "length": len(vals),
                "channel": channel,
                "features": feats,
            },
        )
