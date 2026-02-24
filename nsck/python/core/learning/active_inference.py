"""
NSCK Active Inference Module (V8)
===================================
Free energy minimization for action selection.

F(action) = prediction_error(action) - epistemic_value(action)
prediction_error = Hamming(predicted_state_hv, actual_state_hv) / dim
epistemic_value = curiosity_score from CuriosityModule
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any

import python.core.vsa.hypervec_shim as hypervec_rs


class ActiveInferenceLearner:
    """
    Computes free energy for actions and updates a world model.
    Integrates with CognitiveEngine.decide() to bias coalition scoring.
    """

    def __init__(
        self,
        curiosity_module=None,
        safety_threshold: float = 0.7,
    ):
        self.curiosity = curiosity_module
        self.safety_threshold = safety_threshold
        self._world_model: Dict[str, Dict[str, Any]] = {}
        self._last_states: Dict[str, Any] = {}

    def prediction_error(self, action: str, state_hv) -> float:
        """Hamming distance between predicted and actual state HV, normalised [0,1]."""
        key = self._hv_key(state_hv)
        predicted = self._world_model.get(key, {}).get(action)
        if predicted is None:
            return 0.5
        actual = self._last_states.get(key, state_hv)
        try:
            xored = predicted.xor(actual)
            bits = bytes(xored.bits) if hasattr(xored, 'bits') else b'\x00'
            dist = float(np.mean(np.frombuffer(bits, dtype=np.uint8) > 0))
        except Exception:
            dist = 0.5
        return dist

    def epistemic_value(self, action: str, state_hv) -> float:
        """Curiosity / epistemic value for action (reduces uncertainty)."""
        if self.curiosity is None:
            return 0.1
        try:
            if hasattr(self.curiosity, 'score'):
                return float(self.curiosity.score(action, state_hv))
            if hasattr(self.curiosity, 'compute_novelty'):
                return float(self.curiosity.compute_novelty(state_hv, "default"))
            key = self._hv_key(state_hv)
            return 0.3 if key not in self._world_model else 0.1
        except Exception:
            return 0.1

    def free_energy(self, action: str, state_hv) -> float:
        """F(action) = prediction_error - epistemic_value. Lower = more preferred."""
        return self.prediction_error(action, state_hv) - self.epistemic_value(action, state_hv)

    def should_veto(self, action: str, state_hv) -> bool:
        """Return True if free energy exceeds safety threshold."""
        return self.free_energy(action, state_hv) > self.safety_threshold

    def update_world_model(self, state_hv, action: str, next_state_hv) -> None:
        """Update predictive model after observing state transition."""
        key = self._hv_key(state_hv)
        if key not in self._world_model:
            self._world_model[key] = {}
        existing = self._world_model[key].get(action)
        if existing is None:
            self._world_model[key][action] = next_state_hv
        else:
            try:
                self._world_model[key][action] = existing.bundle(next_state_hv)
            except Exception:
                self._world_model[key][action] = next_state_hv
        self._last_states[key] = next_state_hv

    def _hv_key(self, hv) -> str:
        if hv is None:
            return "none"
        try:
            b = bytes(hv.bits[:16]) if hasattr(hv, 'bits') else b''
            return str(hash(b))
        except Exception:
            return str(id(hv))
