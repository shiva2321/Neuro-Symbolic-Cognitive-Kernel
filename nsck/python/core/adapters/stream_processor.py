"""
StreamProcessor — temporal sensor stream adapter (STEP 7).

Ingests a sequence of channel/value/timestamp data points, detects when
enough data has accumulated, and emits a PerceptPacket with temporal
features (trend, rate of change, anomaly detection).

StreamVerifier — a GroundingVerifier subclass that auto-generates
temporal predicates (RISING_X, FALLING_X, ANOMALY_X) from feature dicts.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs

from python.core.types.percept_packet import PerceptPacket
from python.core.perception.grounding_verifier import GroundingVerifier


# ---------------------------------------------------------------------------
# StreamProcessor
# ---------------------------------------------------------------------------

class StreamProcessor:
    """Accumulates sensor readings and emits PerceptPackets.

    Usage::

        sp = StreamProcessor(window_size=10)
        for t, v in sensor_stream:
            sp.ingest("temperature", v, t)
            if sp.ready():
                packet = sp.emit(adapter, task_tag)
                engine.decide(packet, task_tag)

    Parameters
    ----------
    window_size : int
        Minimum number of data points required before ``ready()`` returns
        ``True`` (default: 10).
    anomaly_sigma : float
        Number of standard deviations from the mean that counts as an
        anomaly (default: 2.0).
    """

    def __init__(self, window_size: int = 10, anomaly_sigma: float = 2.0) -> None:
        self.window_size = window_size
        self.anomaly_sigma = anomaly_sigma
        # channel → deque of (timestamp, value)
        self._data: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest(self, channel: str, value: float, timestamp: Optional[float] = None) -> None:
        """Add a data point to *channel*.

        Parameters
        ----------
        channel : str
            Sensor channel name (e.g. ``"temperature"``, ``"pressure"``).
        value : float
            Measured value.
        timestamp : float, optional
            Unix timestamp (defaults to ``time.time()``).
        """
        ts = timestamp if timestamp is not None else time.time()
        self._data[channel].append((ts, value))

    def ready(self) -> bool:
        """Return ``True`` if any channel has accumulated ``window_size`` points."""
        return any(len(v) >= self.window_size for v in self._data.values())

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, adapter: Any, task_tag: str) -> PerceptPacket:
        """Compute temporal features and return a PerceptPacket.

        Parameters
        ----------
        adapter : ModalityAdapter
            Any adapter that can encode the feature dict.  If *None*,
            a minimal packet is built directly.
        task_tag : str
            Registered task domain identifier.

        Returns
        -------
        PerceptPacket with ``modality="stream"``.
        """
        features = self._extract_features()

        # Build HV via adapter if given, else use feature dict hash
        if adapter is not None:
            base_packet = adapter.encode(features, task_tag)
            situation_hv = base_packet.situation_hv
        else:
            # Fallback: hash the feature dict into a stable HV seed
            import hashlib
            import json
            seed_bytes = hashlib.sha256(
                json.dumps(sorted(features.items()), default=str).encode()
            ).digest()
            seed = int.from_bytes(seed_bytes[:4], "little")
            situation_hv = hypervec_rs.HyperVector(seed)

        # Auto-generate predicates from features using StreamVerifier
        sv = StreamVerifier()
        active_preds = frozenset(sv._features_to_predicates(features))

        return PerceptPacket.make(
            modality="stream",
            situation_hv=situation_hv,
            active_predicates=active_preds,
            confidence=float(features.get("anomaly_score", 0.5)),
            raw_state=features,
            adapter_name="StreamProcessor",
            adapter_trace={"window_size": self.window_size, "channels": list(self._data.keys())},
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self) -> Dict[str, Any]:
        """Compute temporal features for all channels.

        Features per channel *X*:
        - ``X_mean``, ``X_min``, ``X_max`` — basic statistics
        - ``X_trend`` — linear trend coefficient (positive = rising)
        - ``X_rate`` — average rate of change per second
        - ``X_anomaly`` — True if latest value is > anomaly_sigma σ from mean
        - ``anomaly_score`` — fraction of channels with anomaly (0–1)

        Returns
        -------
        Dict of feature name → value.
        """
        import math

        features: Dict[str, Any] = {}
        anomaly_count = 0
        channel_count = 0

        for channel, points in self._data.items():
            if not points:
                continue
            channel_count += 1

            timestamps = [p[0] for p in points]
            values = [p[1] for p in points]

            n = len(values)
            mean_v = sum(values) / n
            min_v = min(values)
            max_v = max(values)

            # Variance and std
            var_v = sum((v - mean_v) ** 2 for v in values) / n if n > 1 else 0.0
            std_v = math.sqrt(var_v)

            # Linear trend via simple least squares
            if n >= 2:
                x_mean = sum(timestamps) / n
                num = sum((timestamps[i] - x_mean) * (values[i] - mean_v) for i in range(n))
                den = sum((timestamps[i] - x_mean) ** 2 for i in range(n))
                trend = num / den if den != 0 else 0.0
            else:
                trend = 0.0

            # Rate of change (last two points)
            if n >= 2 and (timestamps[-1] - timestamps[-2]) > 0:
                rate = (values[-1] - values[-2]) / (timestamps[-1] - timestamps[-2])
            else:
                rate = 0.0

            # Anomaly: latest value > anomaly_sigma standard deviations from mean
            anomaly = bool(std_v > 0 and abs(values[-1] - mean_v) > self.anomaly_sigma * std_v)
            if anomaly:
                anomaly_count += 1

            features[f"{channel}_mean"] = mean_v
            features[f"{channel}_min"] = min_v
            features[f"{channel}_max"] = max_v
            features[f"{channel}_trend"] = trend
            features[f"{channel}_rate"] = rate
            features[f"{channel}_anomaly"] = anomaly

        # Overall anomaly score
        features["anomaly_score"] = (
            anomaly_count / channel_count if channel_count > 0 else 0.0
        )
        return features


# ---------------------------------------------------------------------------
# StreamVerifier
# ---------------------------------------------------------------------------

class StreamVerifier(GroundingVerifier):
    """GroundingVerifier subclass that generates temporal predicates.

    Auto-generates:
    - ``RISING_X`` when channel *X* has a positive trend
    - ``FALLING_X`` when channel *X* has a negative trend
    - ``ANOMALY_X`` when channel *X* is anomalous

    Parameters
    ----------
    trend_threshold : float
        Minimum absolute trend coefficient to classify as rising/falling
        (default: 1e-6).
    """

    def __init__(self, trend_threshold: float = 1e-6) -> None:
        super().__init__()
        self.trend_threshold = trend_threshold

    def get_active_predicates(self, state: Dict[str, Any], context: str = "") -> List[str]:
        """Extract temporal predicates from a stream feature dict."""
        base = super().get_active_predicates(state, context=context)
        return base + self._features_to_predicates(state)

    def _features_to_predicates(self, features: Dict[str, Any]) -> List[str]:
        """Convert a feature dict into temporal symbolic predicates."""
        preds: List[str] = []
        for key, val in features.items():
            if key.endswith("_trend"):
                channel = key[: -len("_trend")].upper()
                if isinstance(val, (int, float)):
                    if val > self.trend_threshold:
                        preds.append(f"RISING_{channel}")
                    elif val < -self.trend_threshold:
                        preds.append(f"FALLING_{channel}")
            elif key.endswith("_anomaly") and val:
                channel = key[: -len("_anomaly")].upper()
                preds.append(f"ANOMALY_{channel}")
        return preds
