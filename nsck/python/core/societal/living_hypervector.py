"""
LivingHyperVector — a dynamic, stateful hypervector agent for NSCK V5.

Each ``LivingHyperVector`` (LHV) wraps a raw hypervector with cognitive
metadata: stability, valence, domain affinity, bond chemistry and
provenance tracking.  LHVs form the atomic units of the Societal
Knowledge World.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACTIVATION_DECAY = 0.05          # fraction lost per tick
_ACTIVATION_HISTORY_LEN = 32      # number of recent activations kept
_STABILITY_CLASS_THRESHOLDS = {   # lower-bound of stability to reach a class
    "crystallized": 0.85,
    "stable":       0.60,
    "active":       0.30,
    # below 0.30 → volatile
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hv_cosine_sim(hv_a, hv_b) -> float:
    """Compute cosine similarity between two hypervectors (any backend).

    Fast path: use the native ``similarity()`` method (Hamming-based, O(D/64)
    bitwise ops) which is available on both Rust and Python HVs and runs at
    sub-microsecond speed on the Rust backend.

    Fallback: convert to bipolar float and compute cosine via numpy dot product.
    This is only reached for exotic backends that expose a ``bits`` attribute
    but no ``similarity()`` method.
    """
    # Prefer the fast native similarity method (Rust: 0.22 µs, Python: 7 µs).
    # Note: Hamming-based similarity and bipolar cosine are monotonically related
    # for binary HVs (both measure fraction of matching bits), so the two paths
    # are consistent in ranking — using similarity() here is correct.
    try:
        return float(hv_a.similarity(hv_b))
    except AttributeError:
        pass
    try:
        a = np.asarray(hv_a.bits, dtype=np.float32) * 2.0 - 1.0
        b = np.asarray(hv_b.bits, dtype=np.float32) * 2.0 - 1.0
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LivingHyperVector:
    """A hypervector augmented with dynamic societal properties.

    Parameters
    ----------
    concept_id:
        Unique string identifier for the concept.
    hv:
        The underlying hypervector object (Python or Rust backend).
    metadata:
        Optional dict with initial property overrides.
    """

    __slots__ = (
        "concept_id", "hv",
        "age", "stability", "valence", "activation",
        "activation_history",
        "domain_affinities", "primary_domain", "neighborhood_id",
        "electronegativity", "bonds",
        "hybridization_state", "stability_class",
        "ewc_protection", "provenance", "embedding",
    )

    def __init__(
        self,
        concept_id: str,
        hv: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        md = metadata or {}
        self.concept_id: str = concept_id
        self.hv = hv
        self.age: int = 0
        self.stability: float = float(md.get("stability", 0.5))
        self.valence: float = float(md.get("valence", 0.0))
        self.activation: float = float(md.get("activation", 0.0))
        self.activation_history: List[float] = []
        self.domain_affinities: Dict[str, float] = dict(md.get("domain_affinities", {}))
        self.primary_domain: Optional[str] = md.get("primary_domain", None)
        self.neighborhood_id: Optional[str] = md.get("neighborhood_id", None)
        self.electronegativity: float = float(md.get("electronegativity", 0.0))
        self.bonds: Dict[str, float] = dict(md.get("bonds", {}))
        self.hybridization_state: str = md.get("hybridization_state", "free")
        self.stability_class: str = md.get("stability_class", "volatile")
        self.ewc_protection: float = float(md.get("ewc_protection", 0.0))
        self.provenance: Dict[str, Any] = dict(md.get("provenance", {
            "created_at": time.time(),
            "method": "direct",
        }))
        self.embedding: Optional[np.ndarray] = md.get("embedding", None)
        self.update_stability_class()

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_activation(self, value: float) -> None:
        """Record a new activation value and trim history."""
        self.activation = float(np.clip(value, 0.0, 1.0))
        self.activation_history.append(self.activation)
        if len(self.activation_history) > _ACTIVATION_HISTORY_LEN:
            self.activation_history.pop(0)

    def tick(self) -> None:
        """Advance one world tick: age++ and decay activation."""
        self.age += 1
        self.activation = max(0.0, self.activation - _ACTIVATION_DECAY)
        # Slowly increase stability with age (capped at 1.0)
        self.stability = min(1.0, self.stability + 0.001)
        self.update_stability_class()
        # Update electronegativity based on bond count and stability
        n_bonds = len(self.bonds)
        self.electronegativity = min(1.0, 0.3 * self.stability + 0.1 * n_bonds)

    def update_stability_class(self) -> None:
        """Recompute ``stability_class`` from current ``stability`` value."""
        s = self.stability
        if s >= _STABILITY_CLASS_THRESHOLDS["crystallized"]:
            self.stability_class = "crystallized"
        elif s >= _STABILITY_CLASS_THRESHOLDS["stable"]:
            self.stability_class = "stable"
        elif s >= _STABILITY_CLASS_THRESHOLDS["active"]:
            self.stability_class = "active"
        else:
            self.stability_class = "volatile"

    def add_bond(self, other_id: str, strength: float) -> None:
        """Add or update a bond to another concept."""
        self.bonds[other_id] = float(np.clip(strength, 0.0, 1.0))
        if len(self.bonds) > 1:
            self.hybridization_state = "bonded"

    def remove_bond(self, other_id: str) -> None:
        """Remove a bond if it exists."""
        self.bonds.pop(other_id, None)
        if len(self.bonds) == 0:
            self.hybridization_state = "free"

    def get_affinity(self, domain: str) -> float:
        """Return domain affinity, defaulting to 0.0."""
        return float(self.domain_affinities.get(domain, 0.0))

    def set_affinity(self, domain: str, score: float) -> None:
        """Set domain affinity and update primary_domain."""
        self.domain_affinities[domain] = float(np.clip(score, 0.0, 1.0))
        if self.domain_affinities:
            self.primary_domain = max(self.domain_affinities, key=self.domain_affinities.get)

    def hybridize(self, context_hv: Any) -> "LivingHyperVector":
        """Return a context-conditioned copy with HV bundled with context_hv."""
        try:
            new_hv = self.hv.bundle(context_hv)
        except Exception:
            new_hv = self.hv  # fallback: unchanged
        copy = deepcopy(self)
        copy.hv = new_hv
        copy.hybridization_state = "hybridized"
        return copy

    def cosine_similarity_to(self, other: "LivingHyperVector") -> float:
        """Compute cosine similarity to another LHV."""
        return _hv_cosine_sim(self.hv, other.hv)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dict (omits raw HV bits for efficiency)."""
        return {
            "concept_id": self.concept_id,
            "age": self.age,
            "stability": self.stability,
            "valence": self.valence,
            "activation": self.activation,
            "activation_history": list(self.activation_history),
            "domain_affinities": dict(self.domain_affinities),
            "primary_domain": self.primary_domain,
            "neighborhood_id": self.neighborhood_id,
            "electronegativity": self.electronegativity,
            "bonds": dict(self.bonds),
            "hybridization_state": self.hybridization_state,
            "stability_class": self.stability_class,
            "ewc_protection": self.ewc_protection,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], hv: Any) -> "LivingHyperVector":
        """Reconstruct from a serialized dict + raw HV object."""
        obj = cls.__new__(cls)
        obj.concept_id = d["concept_id"]
        obj.hv = hv
        obj.age = int(d.get("age", 0))
        obj.stability = float(d.get("stability", 0.5))
        obj.valence = float(d.get("valence", 0.0))
        obj.activation = float(d.get("activation", 0.0))
        obj.activation_history = list(d.get("activation_history", []))
        obj.domain_affinities = dict(d.get("domain_affinities", {}))
        obj.primary_domain = d.get("primary_domain", None)
        obj.neighborhood_id = d.get("neighborhood_id", None)
        obj.electronegativity = float(d.get("electronegativity", 0.0))
        obj.bonds = dict(d.get("bonds", {}))
        obj.hybridization_state = d.get("hybridization_state", "free")
        obj.stability_class = d.get("stability_class", "volatile")
        obj.ewc_protection = float(d.get("ewc_protection", 0.0))
        obj.provenance = dict(d.get("provenance", {}))
        obj.embedding = d.get("embedding", None)
        return obj

    def __repr__(self) -> str:
        return (
            f"LivingHyperVector(id={self.concept_id!r}, "
            f"age={self.age}, stability={self.stability:.2f}, "
            f"valence={self.valence:.2f}, act={self.activation:.2f}, "
            f"class={self.stability_class!r})"
        )
