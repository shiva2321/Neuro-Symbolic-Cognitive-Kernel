"""
NSCK Active Inference Module (V8)
===================================
Free energy minimization for action selection.

F(action) = prediction_error(action) - epistemic_value(action)
prediction_error = Hamming(predicted_state_hv, actual_state_hv) / dim
epistemic_value = curiosity_score from CuriosityModule
"""
from __future__ import annotations
import hashlib
import logging
import numpy as np
from typing import Dict, Optional, Any

import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger("nsck.active_inference")


class ActiveInferenceLearner:
    """
    Computes free energy for actions and updates a world model.
    Integrates with CognitiveEngine.decide() to bias coalition scoring.

    NOTE on degeneracy: without a curiosity_module, epistemic_value() uses
    world-model familiarity as a proxy (0.3 unseen, 0.1 seen) to avoid a
    constant-0.1 return that would cause free_energy() ≈ 0.4 for all states.
    Use compute_degeneracy() to detect when this fallback is active.
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

    def compute_degeneracy(self) -> bool:
        """Return True when curiosity is None and world model is empty (near-constant free energy)."""
        return self.curiosity is None and len(self._world_model) == 0

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
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("prediction_error XOR failed (%s): returning 0.5", e)
            dist = 0.5
        return dist

    def epistemic_value(self, action: str, state_hv) -> float:
        """Curiosity / epistemic value for action (reduces uncertainty).

        NOTE: Without a curiosity_module, falls back to world-model familiarity:
        - 0.3 if the (state, action) transition has never been seen (explore)
        - 0.1 if it has been seen (exploit)
        This avoids the degenerate constant-0.1 behavior.
        """
        if self.curiosity is None:
            key = self._hv_key(state_hv)
            seen = key in self._world_model and action in self._world_model.get(key, {})
            return 0.1 if seen else 0.3
        try:
            if hasattr(self.curiosity, 'score'):
                return float(self.curiosity.score(action, state_hv))
            if hasattr(self.curiosity, 'compute_novelty'):
                return float(self.curiosity.compute_novelty(state_hv, "default"))
            key = self._hv_key(state_hv)
            return 0.3 if key not in self._world_model else 0.1
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("epistemic_value curiosity call failed (%s)", e)
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
        """Return a collision-resistant fingerprint for the given hypervector.

        Uses first 32 bytes (256 bits) with MD5 to substantially reduce the
        collision probability vs. the old 8-byte/16-byte approach.
        """
        if hv is None:
            return "none"
        try:
            b = bytes(hv.bits[:32]) if hasattr(hv, 'bits') else b''
            return str(int.from_bytes(hashlib.md5(b).digest()[:4], 'big') & 0x7FFFFFFF)
        except Exception:
            return str(id(hv) & 0x7FFFFFFF)
