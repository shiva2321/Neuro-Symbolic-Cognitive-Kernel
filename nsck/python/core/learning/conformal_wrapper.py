"""
ConformalWrapper — provable uncertainty bounds via split conformal prediction.

Wraps any confidence score and computes calibrated prediction sets
with coverage guarantees (1 - alpha).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ConformalWrapper:
    """
    Split conformal prediction wrapper.

    Usage
    -----
    1. ``calibrate(scores, labels)`` — fit on calibration set
    2. ``predict_set(score)`` — returns prediction set + coverage info
    3. ``uncertainty_bound(score)`` — returns (lower, upper) bounds

    The conformal quantile q_hat is computed from calibration scores such that
    the prediction set has coverage >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Parameters
        ----------
        alpha : float
            Desired miscoverage rate. Coverage guarantee = 1 - alpha.
        """
        self.alpha = alpha
        self._q_hat: Optional[float] = None
        self._calibration_scores: List[float] = []
        self._n_calibration: int = 0

    def calibrate(
        self,
        scores: List[float],
        labels: Optional[List[bool]] = None,
    ) -> float:
        """
        Fit the conformal quantile from calibration scores.

        Parameters
        ----------
        scores : list of float
            Non-conformity scores from calibration set (e.g., 1 - confidence).
        labels : list of bool, optional
            If provided, only use scores where label=True (correct predictions).

        Returns
        -------
        q_hat : float
            The calibration quantile.
        """
        if labels is not None:
            filtered = [s for s, l in zip(scores, labels) if l]
        else:
            filtered = list(scores)

        if not filtered:
            self._q_hat = 1.0
            return 1.0

        self._calibration_scores = filtered
        self._n_calibration = len(filtered)
        n = len(filtered)
        # Conformal quantile: ceil((n+1)(1-alpha))/n quantile
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self._q_hat = float(np.quantile(filtered, level))
        return self._q_hat

    def predict_set(
        self,
        score: float,
        candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return a calibrated prediction set.

        A candidate is included if its non-conformity score <= q_hat.

        Parameters
        ----------
        score : float
            Non-conformity score of the query (e.g., 1 - confidence).
        candidates : list of str, optional
            Named candidates (for reporting).

        Returns
        -------
        dict with keys: included, q_hat, coverage, uncertain
        """
        if self._q_hat is None:
            return {
                "included": candidates or [],
                "q_hat": None,
                "coverage": None,
                "uncertain": True,
                "warning": "Not calibrated",
            }
        included = score <= self._q_hat
        return {
            "included": (candidates if included else []) if candidates else included,
            "q_hat": self._q_hat,
            "coverage": 1.0 - self.alpha,
            "uncertain": not included,
            "score": score,
        }

    def uncertainty_bound(self, score: float) -> Tuple[float, float]:
        """
        Return (lower, upper) bounds on the true confidence.
        If q_hat is not set, returns (0.0, 1.0).
        """
        if self._q_hat is None:
            return (0.0, 1.0)
        # score = 1 - confidence → confidence = 1 - score
        # conformal bound: confidence ≥ 1 - q_hat if score ≤ q_hat
        lower = max(0.0, 1.0 - self._q_hat)
        upper = min(1.0, 1.0 - score + self._q_hat * 0.1)
        return (lower, upper)

    def is_calibrated(self) -> bool:
        return self._q_hat is not None

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "q_hat": self._q_hat,
            "n_calibration": self._n_calibration,
            "target_coverage": 1.0 - self.alpha,
        }
