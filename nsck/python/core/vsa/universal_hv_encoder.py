"""
UniversalHVEncoder — adaptive, signal-agnostic hypervector encoding.

Encodes any TypedSignal (or raw numpy array) into a similarity-preserving
HyperVector using:
  - Hebbian projector weights (learned from experience)
  - FPE (Fractional Power Encoding) quantization
  - Feature importance tracking
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

import python.core.vsa.hypervec_shim as hv_mod
from python.core.perception.signal_ingestor import SignalIngestor, TypedSignal

_N_BINS: int = 256


def _build_codebook(n: int) -> List[hv_mod.HyperVector]:
    return [hv_mod.HyperVector(i * 31 + 9999) for i in range(n)]


_CODEBOOK: List[hv_mod.HyperVector] = _build_codebook(_N_BINS)


class UniversalHVEncoder:
    """
    Encodes TypedSignals (or any Python object via SignalIngestor) into HyperVectors.

    Attributes
    ----------
    projector_weights : np.ndarray  (n_features,)
        Hebbian-updated weights giving per-dimension importance.
    feature_importance : Dict[int, float]
        Mapping from feature index → cumulative importance score.
    """

    def __init__(self, n_features: int = 128) -> None:
        self.n_features = n_features
        self.projector_weights = np.ones(n_features, dtype=np.float64)
        self.feature_importance: Dict[int, float] = {}
        self._ingestor = SignalIngestor()
        self._encode_count = 0

    def encode(self, signal: Any, label: Optional[str] = None) -> hv_mod.HyperVector:
        """Encode any input into a HyperVector."""
        if isinstance(signal, TypedSignal):
            ts = signal
        else:
            ts = self._ingestor.ingest(signal, label)
        return self._encode_typed(ts)

    def encode_with_stats(self, signal: Any, label: Optional[str] = None) -> Dict[str, Any]:
        """Encode and return HV plus encoding statistics."""
        if isinstance(signal, TypedSignal):
            ts = signal
        else:
            ts = self._ingestor.ingest(signal, label)
        hv = self._encode_typed(ts)
        return {
            "hv": hv,
            "source_type": ts.source_type,
            "n_dims": len(ts.data),
            "stats": ts.stats,
            "top_features": self.get_top_features(k=5),
        }

    def _encode_typed(self, ts: TypedSignal) -> hv_mod.HyperVector:
        data = ts.data
        n = min(len(data), self.n_features)
        if n == 0:
            return hv_mod.HyperVector(0)

        # Project to n_features dimensions
        if len(data) != self.n_features:
            # Resample to n_features via linear interpolation
            indices = np.linspace(0, len(data) - 1, self.n_features)
            projected = np.interp(indices, np.arange(len(data)), data)
        else:
            projected = data.copy()

        # Apply Hebbian weights
        weighted = projected * self.projector_weights

        # Normalize to [0, 1]
        mn, mx = weighted.min(), weighted.max()
        if mx - mn > 1e-10:
            weighted = (weighted - mn) / (mx - mn)

        # FPE quantisation → HV accumulation
        bin_indices = np.clip(
            (weighted * (_N_BINS - 1)).astype(int), 0, _N_BINS - 1
        )
        acc = None
        for d, b in enumerate(bin_indices):
            val_hv = _CODEBOOK[b]
            role_hv = hv_mod.HyperVector((d * 1013 + 5003) % (2 ** 32))
            bound = val_hv.xor(role_hv)
            acc = bound if acc is None else acc.bundle(bound)

        self._encode_count += 1
        # Update feature importance (exponential moving average)
        for i, w in enumerate(projected):
            prev = self.feature_importance.get(i, 0.0)
            self.feature_importance[i] = 0.9 * prev + 0.1 * abs(float(w))

        return acc if acc is not None else hv_mod.HyperVector(0)

    def hebbian_update(self, signal: Any, reward: float) -> None:
        """Update projector weights based on reward signal (Hebbian rule)."""
        if isinstance(signal, TypedSignal):
            ts = signal
        else:
            ts = self._ingestor.ingest(signal)
        data = ts.data
        if len(data) != self.n_features:
            indices = np.linspace(0, len(data) - 1, self.n_features)
            projected = np.interp(indices, np.arange(len(data)), data)
        else:
            projected = data.copy()
        delta = 0.01 * reward * np.abs(projected)
        self.projector_weights = np.clip(self.projector_weights + delta, 0.1, 5.0)

    def get_top_features(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return the k most important feature dimensions."""
        sorted_feats = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return [{"index": i, "importance": v} for i, v in sorted_feats[:k]]

    def reset_weights(self) -> None:
        """Reset Hebbian weights to uniform."""
        self.projector_weights = np.ones(self.n_features, dtype=np.float64)
        self.feature_importance = {}
