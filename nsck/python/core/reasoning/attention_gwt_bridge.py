"""
Attention ↔ GWT Bridge
=======================
Multi-head attention scoring for Global Workspace Theory coalition competition.
"""
from __future__ import annotations

import math
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


class AttentionHead:
    """Single attention head for scoring coalitions."""

    def __init__(self, key_dim: int = 64, name: str = "head_0"):
        self.key_dim = key_dim
        self.name = name
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        self._Wq = rng.standard_normal((key_dim, key_dim)).astype(np.float32) * 0.1
        self._Wk = rng.standard_normal((key_dim, key_dim)).astype(np.float32) * 0.1

    def _project(self, vec: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Project vector to key_dim space."""
        v = np.asarray(vec, dtype=np.float32).flatten()
        if len(v) >= self.key_dim:
            v = v[: self.key_dim]
        else:
            v = np.pad(v, (0, self.key_dim - len(v)))
        return v @ W

    def score(self, query: np.ndarray, keys: List[np.ndarray]) -> np.ndarray:
        """Compute softmax attention scores of query against keys."""
        q = self._project(query, self._Wq)
        scores = []
        scale = self.key_dim ** 0.5
        for k in keys:
            kp = self._project(k, self._Wk)
            scores.append(float(np.dot(q, kp)) / (scale + 1e-9))
        return _softmax(np.array(scores, dtype=np.float32))

    def attend(
        self,
        query: np.ndarray,
        keys: List[np.ndarray],
        values: List[np.ndarray],
    ) -> np.ndarray:
        """Weighted sum of values by attention scores."""
        scores = self.score(query, keys)
        vals = [np.asarray(v, dtype=np.float32).flatten() for v in values]
        max_len = max(len(v) for v in vals)
        padded = [np.pad(v, (0, max_len - len(v))) for v in vals]
        return sum(s * v for s, v in zip(scores, padded))


class MultiHeadAttentionGWT:
    """Multi-head attention for GWT coalition competition."""

    def __init__(self, n_heads: int = 4, key_dim: int = 64):
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.heads = [AttentionHead(key_dim=key_dim, name=f"head_{i}") for i in range(n_heads)]

    def compute_attention(
        self,
        workspace_hv: np.ndarray,
        coalition_hvs: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute multi-head attention weights for each coalition.
        Returns dict mapping coalition_name → attention_weight.
        """
        if not coalition_hvs:
            return {}
        names = list(coalition_hvs.keys())
        keys = [np.asarray(coalition_hvs[n], dtype=np.float32).flatten() for n in names]
        query = np.asarray(workspace_hv, dtype=np.float32).flatten()

        # Average scores across heads
        all_scores = np.zeros(len(names), dtype=np.float32)
        for head in self.heads:
            scores = head.score(query, keys)
            all_scores += scores
        all_scores /= self.n_heads
        all_scores = _softmax(all_scores)
        return {name: float(score) for name, score in zip(names, all_scores)}

    def gwt_compete_with_attention(
        self,
        workspace_hv: np.ndarray,
        coalitions: Dict[str, Tuple[np.ndarray, float]],
    ) -> str:
        """
        GWT competition with attention.
        coalitions: {name: (hv, base_salience)}
        Returns winner name.
        """
        if not coalitions:
            return ""
        coalition_hvs = {name: hv for name, (hv, _) in coalitions.items()}
        attn_weights = self.compute_attention(workspace_hv, coalition_hvs)

        final_scores = {}
        for name, (_, base_salience) in coalitions.items():
            final_scores[name] = attn_weights.get(name, 0.0) * float(base_salience)

        return max(final_scores, key=final_scores.__getitem__)


class AttentionGWTBridge:
    """Bridges attention scoring with GWT competition."""

    def __init__(self, config=None):
        self._config = config
        self._mha = MultiHeadAttentionGWT()
        self._history: List[dict] = []

    def enhance_gwt_competition(
        self,
        workspace_hv: Any,
        coalitions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Convert coalition HVs to numpy, run attention-weighted competition."""
        coalition_hvs: Dict[str, np.ndarray] = {}
        for name, coalition in coalitions.items():
            if hasattr(coalition, "bits"):
                hv_arr = np.asarray(coalition.bits, dtype=np.float32)
            elif hasattr(coalition, "phasors"):
                hv_arr = np.abs(coalition.phasors).astype(np.float32)
            elif isinstance(coalition, np.ndarray):
                hv_arr = coalition.astype(np.float32)
            elif hasattr(coalition, "hv"):
                hv_arr = np.asarray(getattr(coalition, "hv"), dtype=np.float32).flatten()
            else:
                hv_arr = np.ones(64, dtype=np.float32)
            coalition_hvs[name] = hv_arr

        if hasattr(workspace_hv, "bits"):
            ws_arr = np.asarray(workspace_hv.bits, dtype=np.float32)
        elif isinstance(workspace_hv, np.ndarray):
            ws_arr = workspace_hv.astype(np.float32)
        else:
            ws_arr = np.ones(64, dtype=np.float32)

        return self._mha.compute_attention(ws_arr, coalition_hvs)

    def record_attention_trace(
        self, winner: str, scores: Dict[str, float]
    ) -> List[dict]:
        """Track history of attention decisions."""
        entry = {"winner": winner, "scores": dict(scores)}
        self._history.append(entry)
        return list(self._history)
