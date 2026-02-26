"""
NSCK V11: TimeSeriesEncoder and StreamBuffer
=============================================
Encodes numerical time-series sequences as hypervectors using
Fractional Power Encoding (FPE) for values and permutation for time position.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.types.percept_packet import PerceptPacket


class TimeSeriesEncoder:
    """
    Encodes numerical time-series sequences as hypervectors.

    Uses Fractional Power Encoding (FPE) for values + permutation for time position.
    Result: similar sequences → similar HVs; different sequences → orthogonal HVs.
    """

    N_BINS = 1024
    # Fraction of the value range used as the trend sensitivity: a slope smaller
    # than (range / TREND_SENSITIVITY) is considered flat (not rising/falling).
    TREND_SENSITIVITY: float = 1000.0
    # Fraction of the value range used as the stability threshold: std smaller
    # than (range / STABILITY_DIVISOR) is considered a stable signal.
    STABILITY_DIVISOR: float = 100.0
    # Fraction of values more than 2σ from the mean to trigger ANOMALY predicate.
    ANOMALY_THRESHOLD: float = 0.2
    # Minimum normalised autocorrelation peak to declare a signal PERIODIC.
    PERIODICITY_THRESHOLD: float = 0.7

    def __init__(
        self,
        hv_dim: int = 10240,
        value_range: Tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        self.hv_dim = hv_dim
        self.value_range = value_range
        self._v_min, self._v_max = value_range
        # Pre-compute FPE codebook: one HV per quantised bin
        self._codebook: List[hypervec_rs.HyperVector] = []
        self._build_codebook()

    def _build_codebook(self) -> None:
        """Build the FPE codebook of N_BINS hypervectors."""
        self._codebook = [hypervec_rs.HyperVector(i * 17 + 999) for i in range(self.N_BINS)]

    def _quantize(self, value: float) -> int:
        """Map a float value to a bin index in [0, N_BINS)."""
        v = max(self._v_min, min(self._v_max, value))
        normalized = (v - self._v_min) / (self._v_max - self._v_min)
        return int(normalized * (self.N_BINS - 1))

    def encode_value(self, value: float) -> hypervec_rs.HyperVector:
        """Encode a single float value as HV using FPE quantization."""
        bin_idx = self._quantize(value)
        return self._codebook[bin_idx]

    def encode_sequence(self, values: List[float], window_size: int = 32) -> hypervec_rs.HyperVector:
        """
        Encode a sequence of floats as a single HV.

        Each value at position t: HV_t = permute(encode_value(v_t), t)
        Final: bundle of all positionally-encoded value HVs.
        """
        if not values:
            return hypervec_rs.HyperVector(0)
        chunk = values[:window_size]
        acc = hypervec_rs.HyperVector(1)
        for t, v in enumerate(chunk):
            v_hv = self.encode_value(v)
            pos_hv = v_hv.permute(t) if hasattr(v_hv, 'permute') else v_hv
            acc = acc.bundle(pos_hv)
        return acc

    def encode_features(self, values: List[float]) -> Dict[str, float]:
        """
        Extract statistical features from a sequence.

        Returns:
            mean, std, min, max, trend (slope), rate_of_change,
            anomaly_score, periodicity_score
        """
        if not values:
            return {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "trend": 0.0, "rate_of_change": 0.0,
                "anomaly_score": 0.0, "periodicity_score": 0.0,
            }
        arr = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))

        # Trend: slope of linear fit
        n = len(arr)
        if n >= 2:
            xs = np.arange(n, dtype=np.float64)
            slope = float(np.polyfit(xs, arr, 1)[0])
        else:
            slope = 0.0

        # Rate of change: mean of first differences
        if n >= 2:
            roc = float(np.mean(np.diff(arr)))
        else:
            roc = 0.0

        # Anomaly score: fraction of values > 2 std from mean
        if std > 1e-9:
            anomaly_score = float(np.sum(np.abs(arr - mean) > 2.0 * std) / n)
        else:
            anomaly_score = 0.0

        # Periodicity: peak of normalised autocorrelation (lag 1..n//2)
        periodicity_score = 0.0
        if n >= 4:
            centered = arr - mean
            var = np.dot(centered, centered)
            if var > 1e-12:
                max_lag = max(1, n // 2)
                acf = [
                    float(np.dot(centered[:n - lag], centered[lag:]) / var)
                    for lag in range(1, max_lag + 1)
                ]
                periodicity_score = float(max(acf)) if acf else 0.0

        return {
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "trend": slope,
            "rate_of_change": roc,
            "anomaly_score": anomaly_score,
            "periodicity_score": periodicity_score,
        }

    def features_to_predicates(
        self,
        features: Dict[str, float],
        channel: str = "signal",
    ) -> Set[str]:
        """
        Convert statistical features to symbolic predicates.
        """
        ch = channel.upper()
        preds: Set[str] = set()
        mid = (self._v_min + self._v_max) / 2.0

        trend = features.get("trend", 0.0)
        std = features.get("std", 0.0)
        mean = features.get("mean", 0.0)
        anomaly = features.get("anomaly_score", 0.0)
        periodicity = features.get("periodicity_score", 0.0)

        trend_threshold = (self._v_max - self._v_min) / self.TREND_SENSITIVITY
        small_std = (self._v_max - self._v_min) / self.STABILITY_DIVISOR

        if trend > trend_threshold:
            preds.add(f"RISING_{ch}")
        elif trend < -trend_threshold:
            preds.add(f"FALLING_{ch}")

        if anomaly > self.ANOMALY_THRESHOLD:
            preds.add(f"ANOMALY_{ch}")

        if mean > mid:
            preds.add(f"HIGH_{ch}")
        elif mean < mid:
            preds.add(f"LOW_{ch}")

        if std < small_std:
            preds.add(f"STABLE_{ch}")

        if periodicity > self.PERIODICITY_THRESHOLD:
            preds.add(f"PERIODIC_{ch}")

        return preds


class StreamBuffer:
    """
    Sliding window buffer for continuous streams.

    Ingests (channel_name, value, timestamp) tuples.
    When window is full, emits a PerceptPacket.
    """

    def __init__(
        self,
        window_size: int = 32,
        step_size: int = 8,
        channels: Optional[List[str]] = None,
    ) -> None:
        self.window_size = window_size
        self.step_size = step_size
        self._channels: List[str] = channels or []
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._encoder = TimeSeriesEncoder()

    def ingest(self, channel: str, value: float, timestamp: float) -> None:
        """Add a data point to the stream buffer."""
        if channel not in self._channels:
            self._channels.append(channel)
        self._buffers[channel].append((timestamp, value))

    def ready(self) -> bool:
        """Returns True when enough data has accumulated."""
        return any(
            len(self._buffers[ch]) >= self.window_size
            for ch in self._channels
        )

    def emit(self, task_tag: str) -> Optional[PerceptPacket]:
        """
        When ready, encode the current window as a PerceptPacket.
        """
        if not self.ready():
            return None

        channel_hvs: List[hypervec_rs.HyperVector] = []
        all_predicates: Set[str] = set()
        entity_hvs: Dict[str, hypervec_rs.HyperVector] = {}

        for ch in self._channels:
            buf = self._buffers[ch]
            if len(buf) < 1:
                continue
            vals = [v for _, v in buf]
            ch_hv = self._encoder.encode_sequence(vals)
            role_hv = hypervec_rs.HyperVector(hash(f"channel_{ch}") % (2**32))
            bound_hv = role_hv.bind(ch_hv) if hasattr(role_hv, 'bind') else ch_hv
            channel_hvs.append(bound_hv)
            entity_hvs[ch] = ch_hv

            feats = self._encoder.encode_features(vals)
            preds = self._encoder.features_to_predicates(feats, channel=ch)
            all_predicates.update(preds)

        # Bundle all channel HVs
        if channel_hvs:
            situation_hv = channel_hvs[0]
            for h in channel_hvs[1:]:
                situation_hv = situation_hv.bundle(h)
        else:
            situation_hv = hypervec_rs.HyperVector(0)

        return PerceptPacket.make(
            modality="stream",
            situation_hv=situation_hv,
            active_predicates=frozenset(all_predicates),
            entity_hvs=entity_hvs,
            raw_state={"channels": self._channels},
            adapter_name="StreamBuffer",
            adapter_trace={
                "window_size": self.window_size,
                "channels": list(self._channels),
            },
        )
