"""
ZipfValidator — validates activation distributions follow a power-law for NSCK V5.

A healthy societal world should exhibit Zipf-like activation distributions
(α ≈ 1, R² > 0.85) analogous to natural language and neural firing rates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("nsck.zipf_validator")

_ALPHA_LOW = 0.8
_ALPHA_HIGH = 1.5
_R2_THRESHOLD = 0.85


class ZipfValidator:
    """Validates that concept activation frequencies follow a power-law.

    Parameters
    ----------
    min_concepts:
        Minimum number of concepts required for meaningful analysis (default 10).
    """

    def __init__(self, min_concepts: int = 10) -> None:
        self.min_concepts = int(min_concepts)

    # ------------------------------------------------------------------

    def get_activation_counts(self, concepts: Dict[str, Any]) -> np.ndarray:
        """Extract mean activation histories sorted descending.

        Parameters
        ----------
        concepts:
            Dict concept_id → LivingHyperVector.

        Returns
        -------
        1-D float32 array of mean activation values, sorted descending.
        Non-positive values are replaced with a small epsilon.
        """
        freqs = []
        for lhv in concepts.values():
            hist = lhv.activation_history
            if hist:
                freqs.append(float(np.mean(hist)))
            else:
                freqs.append(float(lhv.activation))
        if not freqs:
            return np.array([], dtype=np.float32)
        arr = np.array(freqs, dtype=np.float32)
        arr = np.sort(arr)[::-1]
        # Replace zeros / negatives with epsilon
        arr = np.where(arr > 0, arr, 1e-9)
        return arr

    # ------------------------------------------------------------------

    def fit_power_law(
        self, frequencies: np.ndarray
    ) -> tuple:
        """Fit f(r) = C · r^{-α} via log-log linear regression.

        Parameters
        ----------
        frequencies:
            Sorted-descending frequency array (positive values).

        Returns
        -------
        (alpha, intercept_log_C, r_squared) — all floats.
        """
        n = len(frequencies)
        if n < 2:
            return 0.0, 0.0, 0.0

        ranks = np.arange(1, n + 1, dtype=np.float64)
        log_r = np.log(ranks)
        log_f = np.log(frequencies.astype(np.float64))

        # OLS: log_f = log_C - alpha * log_r
        A = np.column_stack([np.ones(n), log_r])
        try:
            result, _, _, _ = np.linalg.lstsq(A, log_f, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0

        log_c, neg_alpha = float(result[0]), float(result[1])
        alpha = -neg_alpha  # convention: positive α

        # R²
        fitted = log_c + neg_alpha * log_r
        ss_res = float(np.sum((log_f - fitted) ** 2))
        ss_tot = float(np.sum((log_f - log_f.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return float(alpha), float(log_c), float(r2)

    # ------------------------------------------------------------------

    def validate(self, concepts: Dict[str, Any]) -> Dict[str, Any]:
        """Run full Zipf validation.

        Returns
        -------
        Dict with: is_zipf_like, alpha, r_squared, n_concepts,
        health_score (0-1), description.
        """
        n = len(concepts)
        if n < self.min_concepts:
            return {
                "is_zipf_like": False,
                "alpha": 0.0,
                "r_squared": 0.0,
                "n_concepts": n,
                "health_score": 0.0,
                "description": f"Too few concepts ({n} < {self.min_concepts}) for Zipf analysis.",
            }

        freqs = self.get_activation_counts(concepts)
        if len(freqs) == 0:
            return {
                "is_zipf_like": False,
                "alpha": 0.0,
                "r_squared": 0.0,
                "n_concepts": n,
                "health_score": 0.0,
                "description": "No activation history available.",
            }

        alpha, _, r2 = self.fit_power_law(freqs)
        entropy = self.compute_entropy(freqs)

        is_zipf = (
            _ALPHA_LOW <= alpha <= _ALPHA_HIGH
            and r2 >= _R2_THRESHOLD
        )

        # Health score: alpha closeness to 1.0 (ideal) + R² weight
        alpha_score = max(0.0, 1.0 - abs(alpha - 1.0) / 1.0)
        r2_score = float(np.clip(r2, 0.0, 1.0))
        health_score = 0.60 * r2_score + 0.40 * alpha_score

        if is_zipf:
            description = f"Zipf-like distribution: α={alpha:.3f}, R²={r2:.3f}."
        elif r2 < _R2_THRESHOLD:
            description = f"Non-Zipf: poor fit R²={r2:.3f} (need ≥{_R2_THRESHOLD})."
        else:
            description = f"Power-law fit but α={alpha:.3f} outside healthy range [{_ALPHA_LOW},{_ALPHA_HIGH}]."

        return {
            "is_zipf_like": bool(is_zipf),
            "alpha": float(alpha),
            "r_squared": float(r2),
            "entropy": float(entropy),
            "n_concepts": int(n),
            "health_score": float(health_score),
            "description": description,
        }

    # ------------------------------------------------------------------

    def compute_entropy(self, frequencies: np.ndarray) -> float:
        """Compute normalized Shannon entropy of the activation distribution."""
        if len(frequencies) == 0:
            return 0.0
        f = frequencies.astype(np.float64)
        f = f / f.sum()
        f = f[f > 0]
        entropy = -float(np.sum(f * np.log(f)))
        max_entropy = float(np.log(len(frequencies)))
        return entropy / max_entropy if max_entropy > 0 else 0.0
