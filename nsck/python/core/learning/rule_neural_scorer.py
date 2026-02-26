"""
Rule Neural Scorer
==================
Lightweight perceptron for neural re-ranking of symbolic rules.
"""
from __future__ import annotations

import sys
import os
from typing import List, Optional

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class RuleFeaturizer:
    """Extracts a feature vector from a Rule object."""

    def featurize(self, rule) -> np.ndarray:
        """Extract 6 features from rule."""
        confidence = float(getattr(rule, "confidence", 0.5))

        support_raw = float(getattr(rule, "support_count", getattr(rule, "support", 0)))
        support = min(support_raw / 100.0, 1.0)

        fire_count = float(getattr(rule, "fire_count", 0))
        max_fire = 10000.0
        fire_ratio = min(fire_count / max_fire, 1.0)

        conditions = getattr(rule, "condition", getattr(rule, "conditions", set()))
        complexity = min(len(conditions) / 10.0, 1.0)

        history = list(getattr(rule, "confidence_history", []))
        if len(history) >= 2:
            recent = history[-3:]
            if len(recent) >= 2:
                xs = np.arange(len(recent), dtype=np.float32)
                ys = np.array(recent, dtype=np.float32)
                xs -= xs.mean()
                denom = float(np.dot(xs, xs))
                trend = float(np.dot(xs, ys) / denom) if denom != 0 else 0.0
            else:
                trend = 0.0
        else:
            trend = 0.0
        trend = float(np.clip(trend, -1.0, 1.0))

        has_task = 1.0 if getattr(rule, "task_tag", None) not in (None, "global") else 0.0

        return np.array(
            [confidence, support, fire_ratio, complexity, trend, has_task],
            dtype=np.float32,
        )


class RuleNeuralScorer:
    """Single-hidden-layer perceptron for scoring rules."""

    def __init__(self, n_features: int = 6, hidden: int = 16):
        self.n_features = n_features
        self.hidden = hidden
        rng = np.random.default_rng(42)
        self._W1 = rng.standard_normal((n_features, hidden)).astype(np.float32) * 0.1
        self._b1 = np.zeros(hidden, dtype=np.float32)
        self._W2 = rng.standard_normal((hidden, 1)).astype(np.float32) * 0.1
        self._b2 = np.zeros(1, dtype=np.float32)
        self._featurizer = RuleFeaturizer()

    def _forward(self, x: np.ndarray):
        """Forward pass: returns (h, output)."""
        h = _sigmoid(x @ self._W1 + self._b1)
        out = _sigmoid(h @ self._W2 + self._b2)
        return h, float(out[0])

    def score(self, rule) -> float:
        """Score a rule: returns float in [0, 1]."""
        x = self._featurizer.featurize(rule)
        _, out = self._forward(x)
        return out

    def update(self, rule, reward: float, lr: float = 0.01):
        """Online gradient update using MSE loss."""
        x = self._featurizer.featurize(rule)
        target = float(np.clip(reward, 0.0, 1.0))

        # Forward
        h = _sigmoid(x @ self._W1 + self._b1)
        out_raw = h @ self._W2 + self._b2
        out = _sigmoid(out_raw)

        # Backward
        loss_grad = 2.0 * (out - target)  # dL/d_out
        d_out = out * (1 - out) * loss_grad  # sigmoid derivative

        dW2 = h[:, None] * d_out[None, :]
        db2 = d_out
        dh = d_out @ self._W2.T

        d_h_act = h * (1 - h) * dh
        dW1 = x[:, None] * d_h_act[None, :]
        db1 = d_h_act

        self._W2 -= lr * dW2
        self._b2 -= lr * db2
        self._W1 -= lr * dW1
        self._b1 -= lr * db1

    def batch_score(self, rules: List) -> List[float]:
        """Score multiple rules."""
        return [self.score(r) for r in rules]

    def rank_rules(self, rules: List) -> List:
        """Return rules sorted by neural score descending."""
        scores = self.batch_score(rules)
        return [r for _, r in sorted(zip(scores, rules), key=lambda x: x[0], reverse=True)]

    def save(self, path: str):
        """Save weights as numpy .npz file. Appends '.npz' if not already present."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        np.savez(
            path,
            W1=self._W1, b1=self._b1,
            W2=self._W2, b2=self._b2,
        )

    def load(self, path: str):
        """Load weights from numpy .npz file. Appends '.npz' if not already present."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        self._W1 = data["W1"]
        self._b1 = data["b1"]
        self._W2 = data["W2"]
        self._b2 = data["b2"]
