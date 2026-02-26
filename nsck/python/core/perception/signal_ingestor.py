"""
SignalIngestor — universal TypedSignal conversion.

Any Python input (str, list, ndarray, dict, bytes, etc.) is normalized into a
metadata-rich numpy array with type info, shape, and statistics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class TypedSignal:
    """A normalized, metadata-rich numpy array."""
    data: np.ndarray             # Normalized float64 array
    source_type: str             # 'text', 'numeric', 'image', 'audio', 'dict', 'bytes', 'unknown'
    original_shape: tuple        # Shape before flattening/normalization
    stats: Dict[str, float]      # mean, std, min, max, l2_norm
    metadata: Dict[str, Any]     # Additional provenance info


class SignalIngestor:
    """
    Converts any Python object into a TypedSignal (normalized float64 numpy array).
    """

    def ingest(self, data: Any, label: Optional[str] = None) -> TypedSignal:
        """Convert data to TypedSignal."""
        if isinstance(data, str):
            return self._from_text(data, label)
        elif isinstance(data, bytes):
            return self._from_bytes(data, label)
        elif isinstance(data, np.ndarray):
            return self._from_array(data, label)
        elif isinstance(data, (list, tuple)):
            return self._from_sequence(data, label)
        elif isinstance(data, dict):
            return self._from_dict(data, label)
        elif isinstance(data, (int, float)):
            arr = np.array([float(data)], dtype=np.float64)
            return self._finalize(arr, "numeric", (1,), {"value": data}, label)
        else:
            arr = np.array([hash(str(data)) % 1000 / 1000.0], dtype=np.float64)
            return self._finalize(arr, "unknown", (1,), {"repr": str(data)[:80]}, label)

    def _from_text(self, text: str, label: Optional[str]) -> TypedSignal:
        # Convert to character codepoint array, normalized to [0,1]
        codes = np.array([ord(c) for c in text[:512]], dtype=np.float64)
        if len(codes) == 0:
            codes = np.zeros(1, dtype=np.float64)
        arr = codes / 128.0  # rough normalization
        return self._finalize(arr, "text", (len(text),), {"length": len(text), "preview": text[:40]}, label)

    def _from_bytes(self, data: bytes, label: Optional[str]) -> TypedSignal:
        arr = np.frombuffer(data[:512], dtype=np.uint8).astype(np.float64) / 255.0
        return self._finalize(arr, "bytes", (len(data),), {"length": len(data)}, label)

    def _from_array(self, arr: np.ndarray, label: Optional[str]) -> TypedSignal:
        orig_shape = arr.shape
        flat = arr.flatten().astype(np.float64)
        # Normalize to [0,1] if range is large
        mn, mx = flat.min(), flat.max()
        if mx - mn > 1e-8:
            flat = (flat - mn) / (mx - mn)
        ndim = arr.ndim
        source = "image" if ndim >= 2 else "numeric"
        return self._finalize(flat, source, orig_shape, {"original_min": float(mn), "original_max": float(mx)}, label)

    def _from_sequence(self, seq: Any, label: Optional[str]) -> TypedSignal:
        try:
            arr = np.array(seq, dtype=np.float64).flatten()
            return self._finalize(arr, "numeric", (len(seq),), {}, label)
        except (ValueError, TypeError):
            # Non-numeric sequence — hash each element
            arr = np.array([hash(str(x)) % 1000 / 1000.0 for x in seq[:512]], dtype=np.float64)
            return self._finalize(arr, "mixed", (len(seq),), {"sample": str(seq[:3])}, label)

    def _from_dict(self, data: dict, label: Optional[str]) -> TypedSignal:
        # Encode dict as sorted key+value hash pairs
        pairs = []
        for k in sorted(data.keys()):
            v = data[k]
            try:
                pairs.append(float(v))
            except (TypeError, ValueError):
                pairs.append(hash(str(v)) % 1000 / 1000.0)
        arr = np.array(pairs, dtype=np.float64) if pairs else np.zeros(1, dtype=np.float64)
        return self._finalize(arr, "dict", (len(data),), {"keys": list(data.keys())[:10]}, label)

    def _finalize(
        self,
        arr: np.ndarray,
        source_type: str,
        orig_shape: tuple,
        meta: Dict[str, Any],
        label: Optional[str],
    ) -> TypedSignal:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.float64)
        mn = float(arr.min())
        mx = float(arr.max())
        mu = float(arr.mean())
        std = float(arr.std())
        l2 = float(np.linalg.norm(arr))
        stats = {"mean": mu, "std": std, "min": mn, "max": mx, "l2_norm": l2}
        if label:
            meta["label"] = label
        return TypedSignal(
            data=arr,
            source_type=source_type,
            original_shape=orig_shape,
            stats=stats,
            metadata=meta,
        )
