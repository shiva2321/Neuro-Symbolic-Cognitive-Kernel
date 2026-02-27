"""Hybrid perception distillation — tracks internal vs bridge encoding quality."""
from __future__ import annotations
from collections import deque
from typing import Dict, Any

import numpy as np


class PerceptionDistiller:
    """
    Tracks quality of internal (pure) vs bridge (neural) perception per modality.
    Graduates a modality to pure internal encoding when rolling avg >= threshold.
    """

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self._history: Dict[str, deque] = {}
        self._graduated: Dict[str, bool] = {}

    def _ensure_modality(self, modality: str) -> None:
        if modality not in self._history:
            self._history[modality] = deque(maxlen=100)
            self._graduated[modality] = False

    def observe(self, modality: str, bridge_hv, internal_hv) -> float:
        """
        Compare bridge vs internal HV quality; track rolling average.
        Returns current similarity score.
        """
        self._ensure_modality(modality)
        try:
            if hasattr(bridge_hv, "cosine_similarity"):
                # cosine_similarity returns [-1, 1]; normalize to [0, 1]
                sim = bridge_hv.cosine_similarity(internal_hv)
                sim = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
            elif hasattr(bridge_hv, "similarity"):
                # similarity() returns [0, 1] (Hamming-based); use directly
                sim = float(np.clip(bridge_hv.similarity(internal_hv), 0.0, 1.0))
            else:
                sim = 0.5
        except Exception:
            sim = 0.0
        
        self._history[modality].append(sim)
        
        # Check graduation
        if not self._graduated[modality] and len(self._history[modality]) >= 10:
            avg = float(np.mean(list(self._history[modality])))
            if avg >= self.threshold:
                self._graduated[modality] = True
        
        return sim

    def is_graduated(self, modality: str) -> bool:
        """True when rolling avg of last 100 observations >= threshold."""
        self._ensure_modality(modality)
        return self._graduated[modality]

    def get_report(self) -> Dict[str, Any]:
        """Per-modality stats."""
        report = {}
        for modality, history in self._history.items():
            hist_list = list(history)
            report[modality] = {
                "observations": len(hist_list),
                "current_quality": hist_list[-1] if hist_list else 0.0,
                "avg": float(np.mean(hist_list)) if hist_list else 0.0,
                "graduated": self._graduated.get(modality, False),
            }
        return report
