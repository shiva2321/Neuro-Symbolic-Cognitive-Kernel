"""
LivingHyperVector — the atomic societal knowledge unit.

A LivingHyperVector wraps a standard binary hypervector with lifecycle
metadata and societal dynamics:

* **activation** — current salience in [0, 1].  Decays each epoch; spikes
  when the concept is queried.
* **bonds** — weighted edges to peer LivingHyperVectors.  Strength grows
  with co-activation and decays if concepts stop co-occurring.
* **role** — semantic role label (e.g. "domain_root", "hub", "leaf").
* **domain path** — hierarchical address, e.g. ``["science", "physics",
  "quantum"]``.

Performance notes
-----------------
* ``_hv_cosine_sim`` uses ``hv.similarity()`` (Hamming, O(D/64) via Rust
  packed operations) — ~0.28 µs, not ``hv.cosine_similarity()`` (~3423 µs).
* The ``.bits`` property now uses ``numpy.unpackbits`` when the Rust backend
  is active, reducing extraction time from ~1193 µs to ~16 µs.
"""
from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hv_cosine_sim(hv_a, hv_b) -> float:
    """Fast cosine-like similarity via Hamming (avoid slow .cosine_similarity).

    Uses ``hv.similarity()`` which is O(D/64) packed-bit popcount on the Rust
    backend (0.28 µs) rather than the O(D) Python loop.
    """
    try:
        return float(hv_a.similarity(hv_b))
    except Exception:
        return 0.5


def _unpack_bits(hv) -> np.ndarray:
    """Extract the bit array from any HV backend in ~16 µs.

    For the Rust backend the ``.bits`` property is installed by
    ``hypervec_shim._install_compat_methods``.  For the Python backend
    HyperVectorPy exposes ``.bits`` directly.
    """
    try:
        bits = hv.bits
        if isinstance(bits, np.ndarray):
            # Check if this is an RHC Phasor
            if bits.dtype == np.complex64 or bits.dtype == np.complex128:
                return bits
            return bits.astype(np.int8)
        # Rust: bits may already be an ndarray from the shim property
        if hasattr(bits, "dtype") and (bits.dtype == np.complex64 or bits.dtype == np.complex128):
            return bits
        return np.asarray(bits, dtype=np.int8)
    except Exception:
        # Last resort: __getstate__ path for Rust u64 words
        try:
            words = hv.__getstate__()
            arr = np.zeros(len(words) * 64, dtype=np.uint8)
            for i, w in enumerate(words):
                for b in range(64):
                    if w & (1 << b):
                        arr[i * 64 + b] = 1
            return arr.astype(np.int8)
        except Exception:
            return np.zeros(10240, dtype=np.int8)


# ---------------------------------------------------------------------------
# Bond dataclass
# ---------------------------------------------------------------------------

@dataclass
class Bond:
    """A weighted, typed edge between two LivingHyperVectors.

    Parameters
    ----------
    peer_id:
        Concept identifier of the bonded peer.
    strength:
        Current bond strength in [0, 1].  Initialized to the similarity
        score at bond-formation time.
    bond_type:
        Semantic type: ``"similarity"``, ``"causal"``, ``"hierarchical"``,
        ``"temporal"``, or ``"contrast"``.
    formed_epoch:
        Logical epoch (integer) when the bond was first created.
    last_active_epoch:
        Most recent epoch in which this bond was reinforced.
    """
    peer_id: str
    strength: float = 0.5
    bond_type: str = "similarity"
    formed_epoch: int = 0
    last_active_epoch: int = 0

    def reinforce(self, delta: float = 0.05) -> None:
        """Increase bond strength, clamped to 1.0."""
        self.strength = min(1.0, self.strength + delta)

    def decay(self, rate: float = 0.01) -> None:
        """Exponential-decay bond strength."""
        self.strength = max(0.0, self.strength * (1.0 - rate))

    def is_alive(self, threshold: float = 0.05) -> bool:
        """Return True if the bond is above the dissolution threshold."""
        return self.strength >= threshold


# ---------------------------------------------------------------------------
# LivingHyperVector
# ---------------------------------------------------------------------------

class LivingHyperVector:
    """An activated hypervector with societal lifecycle metadata.

    Parameters
    ----------
    concept_id:
        Unique string identifier for this concept.
    hv:
        The underlying binary hypervector (Rust or Python backend).
    domain_path:
        Ordered list of domain labels from root to leaf, e.g.
        ``["science", "biology", "genetics"]``.
    role:
        Semantic role within the society.  One of ``"domain_root"``,
        ``"hub"``, ``"bridge"``, ``"leaf"``.
    initial_activation:
        Starting activation level in [0, 1].
    birth_epoch:
        Logical clock value at creation.
    metadata:
        Arbitrary extra key-value pairs (source, confidence, etc.).
    is_rhc:
        Boolean indicating if this hv uses the Residue Hyperdimensional system.
    """

    __slots__ = (
        "concept_id",
        "hv",
        "domain_path",
        "role",
        "activation",
        "birth_epoch",
        "epoch_last_active",
        "_bonds",
        "metadata",
        "_cluster_id",
        "_topo_persistence",
        "is_rhc",
    )

    def __init__(
        self,
        concept_id: str,
        hv: Any,
        domain_path: Optional[List[str]] = None,
        role: str = "leaf",
        initial_activation: float = 0.5,
        birth_epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.concept_id: str = concept_id
        self.hv: Any = hv
        self.domain_path: List[str] = list(domain_path or [])
        self.role: str = role
        self.activation: float = float(np.clip(initial_activation, 0.0, 1.0))
        self.birth_epoch: int = birth_epoch
        self.epoch_last_active: int = birth_epoch
        self._bonds: Dict[str, Bond] = {}
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._cluster_id: Optional[int] = None
        self._topo_persistence: float = 0.0  # persistent-homology lifetime
        
        # Check if the HV represents continuous phasor values
        self.is_rhc: bool = False
        if hasattr(self.hv, "phases") or str(type(self.hv)).find("PhasorHyperVector") != -1:
            self.is_rhc = True

    # ------------------------------------------------------------------
    # Bond management
    # ------------------------------------------------------------------

    def form_bond(
        self,
        peer: "LivingHyperVector",
        epoch: int = 0,
        bond_type: str = "similarity",
        strength_override: Optional[float] = None,
    ) -> Bond:
        """Create or reinforce a bond to *peer*.

        If a bond to *peer* already exists, reinforce it.

        Parameters
        ----------
        peer:
            Target LivingHyperVector.
        epoch:
            Current logical epoch.
        bond_type:
            Type of bond.
        strength_override:
            If supplied, use this initial strength instead of the
            cosine similarity between the two HVs.

        Returns
        -------
        Bond
            The (new or updated) bond object.
        """
        pid = peer.concept_id
        if pid in self._bonds:
            b = self._bonds[pid]
            b.reinforce(delta=0.05)
            b.last_active_epoch = epoch
            return b

        strength = (
            strength_override
            if strength_override is not None
            else _hv_cosine_sim(self.hv, peer.hv)
        )
        bond = Bond(
            peer_id=pid,
            strength=float(strength),
            bond_type=bond_type,
            formed_epoch=epoch,
            last_active_epoch=epoch,
        )
        self._bonds[pid] = bond
        return bond

    def dissolve_bond(self, peer_id: str) -> bool:
        """Remove a bond by peer concept id.

        Returns True if the bond existed and was removed, False otherwise.
        """
        if peer_id in self._bonds:
            del self._bonds[peer_id]
            return True
        return False

    def decay_bonds(self, rate: float = 0.01, min_strength: float = 0.05) -> List[str]:
        """Decay all bonds and remove those that fall below *min_strength*.

        Returns
        -------
        List[str]
            List of peer_ids whose bonds were dissolved.
        """
        dissolved: List[str] = []
        for pid, bond in list(self._bonds.items()):
            bond.decay(rate)
            if not bond.is_alive(min_strength):
                del self._bonds[pid]
                dissolved.append(pid)
        return dissolved

    def get_bonds(self, min_strength: float = 0.0) -> List[Bond]:
        """Return all bonds above *min_strength*, sorted by descending strength."""
        return sorted(
            [b for b in self._bonds.values() if b.strength >= min_strength],
            key=lambda b: b.strength,
            reverse=True,
        )

    def bond_strength(self, peer_id: str) -> float:
        """Return strength of bond to *peer_id*, or 0.0 if no bond exists."""
        b = self._bonds.get(peer_id)
        return b.strength if b is not None else 0.0

    # ------------------------------------------------------------------
    # Activation dynamics
    # ------------------------------------------------------------------

    def activate(self, delta: float = 0.3, epoch: int = 0) -> None:
        """Spike activation by *delta*, clamped to [0, 1]."""
        self.activation = min(1.0, self.activation + float(delta))
        self.epoch_last_active = epoch

    def decay_activation(self, rate: float = 0.05) -> None:
        """Exponential activation decay."""
        self.activation = max(0.0, self.activation * (1.0 - rate))

    def spread_activation(
        self,
        peers: Dict[str, "LivingHyperVector"],
        spread_factor: float = 0.4,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Propagate activation to bonded peers.

        Each bonded peer receives ``spread_factor * bond_strength * self.activation``
        added to its current activation.

        Parameters
        ----------
        peers:
            Mapping of concept_id → LivingHyperVector for the whole society.
        spread_factor:
            Global multiplier on the spread amount.
        epoch:
            Current logical epoch (passed to peer.activate).

        Returns
        -------
        Dict[str, float]
            Mapping of peer_id → amount of activation delivered.
        """
        delivered: Dict[str, float] = {}
        for pid, bond in self._bonds.items():
            peer = peers.get(pid)
            if peer is None:
                continue
            amount = spread_factor * bond.strength * self.activation
            if amount > 1e-6:
                peer.activate(amount, epoch=epoch)
                delivered[pid] = amount
        return delivered

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    @property
    def domain(self) -> str:
        """Top-level domain label, or empty string if none."""
        return self.domain_path[0] if self.domain_path else ""

    @property
    def subdomain(self) -> str:
        """Second-level domain label, or empty string if none."""
        return self.domain_path[1] if len(self.domain_path) > 1 else ""

    @property
    def depth(self) -> int:
        """Depth in the domain hierarchy (0 = root, N = N levels deep)."""
        return len(self.domain_path)

    def is_in_domain(self, domain: str) -> bool:
        """Return True if *domain* appears anywhere in the domain path."""
        return domain in self.domain_path

    # ------------------------------------------------------------------
    # Topological health / persistent homology stub
    # ------------------------------------------------------------------

    def compute_topo_persistence(
        self,
        peers: Dict[str, "LivingHyperVector"],
        eps_range: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 20,
    ) -> float:
        """Estimate H0 persistent homology lifetime for this node.

        Uses a simplified filtration: sweep epsilon from 0 → 1 and measure
        how long this node's connected component persists before merging.

        Returns
        -------
        float
            Persistence lifetime in [0, 1].  Higher = more topologically
            significant (hub concept).  Returns 0.0 if the node has no bonds.
        """
        # A node with no bonds is never part of a multi-node component.
        reachable_bonds = [
            bond for bond in self._bonds.values()
            if bond.peer_id in peers
        ]
        if not reachable_bonds:
            self._topo_persistence = 0.0
            return 0.0

        eps_vals = np.linspace(eps_range[0], eps_range[1], n_steps)
        born_eps: Optional[float] = None
        died_eps: Optional[float] = None

        for eps in eps_vals:
            # Find neighbors with bond strength >= eps
            neighbors = [
                pid for pid, bond in self._bonds.items()
                if bond.strength >= (1.0 - float(eps)) and pid in peers
            ]
            component_size = 1 + len(neighbors)

            if born_eps is None and component_size > 1:
                born_eps = float(eps)
            if born_eps is not None and died_eps is None and component_size == 1:
                died_eps = float(eps)
                break

        if born_eps is None:
            # Node never joined a multi-node component
            lifetime = 0.0
        elif died_eps is None:
            lifetime = float(eps_range[1] - born_eps)
        else:
            lifetime = float(died_eps - born_eps)

        self._topo_persistence = max(0.0, lifetime)
        return self._topo_persistence

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict snapshot of this LHV."""
        return {
            "concept_id": self.concept_id,
            "domain_path": self.domain_path,
            "role": self.role,
            "activation": round(self.activation, 6),
            "birth_epoch": self.birth_epoch,
            "epoch_last_active": self.epoch_last_active,
            "cluster_id": self._cluster_id,
            "topo_persistence": round(self._topo_persistence, 6),
            "bond_count": len(self._bonds),
            "bonds": [
                {
                    "peer_id": b.peer_id,
                    "strength": round(b.strength, 6),
                    "bond_type": b.bond_type,
                    "formed_epoch": b.formed_epoch,
                }
                for b in self.get_bonds()
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], hv: Any) -> "LivingHyperVector":
        """Reconstruct a LivingHyperVector from a serialised dict + an HV."""
        lhv = cls(
            concept_id=data["concept_id"],
            hv=hv,
            domain_path=data.get("domain_path", []),
            role=data.get("role", "leaf"),
            initial_activation=data.get("activation", 0.5),
            birth_epoch=data.get("birth_epoch", 0),
            metadata=data.get("metadata", {}),
        )
        lhv.epoch_last_active = data.get("epoch_last_active", 0)
        lhv._cluster_id = data.get("cluster_id")
        lhv._topo_persistence = data.get("topo_persistence", 0.0)
        for bond_data in data.get("bonds", []):
            b = Bond(
                peer_id=bond_data["peer_id"],
                strength=bond_data.get("strength", 0.5),
                bond_type=bond_data.get("bond_type", "similarity"),
                formed_epoch=bond_data.get("formed_epoch", 0),
            )
            lhv._bonds[b.peer_id] = b
        return lhv

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<LivingHV id={self.concept_id!r} domain={self.domain!r} "
            f"act={self.activation:.3f} bonds={len(self._bonds)} "
            f"cluster={self._cluster_id}>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LivingHyperVector):
            return NotImplemented
        return self.concept_id == other.concept_id

    def __hash__(self) -> int:
        return hash(self.concept_id)
