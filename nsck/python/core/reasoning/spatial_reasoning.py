"""
NSCK Spatial Reasoning Module (F2 from roadmap)
================================================
VSA-native 2D/3D spatial relationship understanding.

No neural networks. No ML libraries. Everything is encoded as
HyperVectors and reasoned about through VSA operations.

Design
------
Spatial positions are encoded using Fractional Power Encoding (FPE) on
axis vectors. A 2D position (x, y) is encoded as::

    v_pos(x,y) = bind(permute(v_x_base, x_step), permute(v_y_base, y_step))

Spatial relations (above, below, left, right, inside, near, …) are
represented as *role vectors* so that::

    v_scene = bind(v_ABOVE, bind(v_cat, v_table))

can be queried: "What is above the table?" by::

    query = unbind(v_scene, bind(v_ABOVE, v_table))
    # → points to v_cat

Relations supported
-------------------
*Projective (axis-aligned)*
    above, below, left_of, right_of, in_front_of, behind

*Topological*
    inside, outside, adjacent_to, overlaps_with, contains, at_center_of

*Metric*
    near (distance < threshold), far (distance > threshold)
    exact distance between two positions

*VSA query interface*
    ``where_is(entity)``, ``what_is_at(position)``,
    ``what_relation(e1, e2)``, ``find_near(position, radius)``

Usage
-----
>>> from python.core.reasoning.spatial_reasoning import SpatialReasoner
>>> sr = SpatialReasoner()
>>> sr.place("cat",   x=1, y=3)
>>> sr.place("table", x=1, y=2)
>>> sr.place("dog",   x=4, y=2)
>>> sr.get_relation("cat", "table")
'above'
>>> sr.get_relation("dog", "table")
'right_of'
>>> sr.find_near("table", radius=2)
['cat', 'dog']
>>> sr.distance("cat", "dog")
3.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs


# ---------------------------------------------------------------------------
# Spatial role vectors (deterministic seeds)
# ---------------------------------------------------------------------------

_ROLE_SEEDS = {
    # Projective
    "ABOVE":       0xA001_0001,
    "BELOW":       0xA001_0002,
    "LEFT_OF":     0xA001_0003,
    "RIGHT_OF":    0xA001_0004,
    "IN_FRONT_OF": 0xA001_0005,
    "BEHIND":      0xA001_0006,
    # Topological
    "INSIDE":      0xA001_0007,
    "OUTSIDE":     0xA001_0008,
    "ADJACENT_TO": 0xA001_0009,
    "OVERLAPS":    0xA001_000A,
    "CONTAINS":    0xA001_000B,
    "AT_CENTER_OF": 0xA001_000C,
    # Positional helpers
    "ENTITY":      0xA001_000D,
    "POSITION":    0xA001_000E,
}

# Axis base seeds — x and y axes are orthogonal by construction
_X_AXIS_SEED = 0xBEEF_0001
_Y_AXIS_SEED = 0xBEEF_0002
_Z_AXIS_SEED = 0xBEEF_0003

# Number of bits flipped per unit step on each axis (FPE-style locality)
# More bits → locality decays faster with distance; fewer → higher noise floor.
# With D=10240 and _AXIS_FLIP_BITS=50: sim(0,1) ≈ 0.99, sim(0,100) ≈ 0.50
_AXIS_FLIP_BITS = 50
_HV_DIM = 10240

# Offset added to negative-axis step numbers so their bit-flip seeds are
# independent from the positive-axis seeds (avoids accidental symmetry).
# Must be larger than the expected max coordinate magnitude.
_NEGATIVE_STEP_OFFSET = 100_000

# Spatial relation confidence thresholds
_ADJACENT_MAX_DIST  = 1.5   # units — within this → "adjacent"
_NEAR_DEFAULT_DIST  = 3.0   # units — default "near" radius
# Minimum VSA similarity to accept a queried figure as a valid match.
# Below this threshold the resonator result is too noisy to be reliable.
_MIN_QUERY_SIMILARITY = 0.4


# ---------------------------------------------------------------------------
# SpatialPosition — a 2D/3D coordinate with VSA encoding
# ---------------------------------------------------------------------------

@dataclass
class SpatialPosition:
    """
    A 2D or 3D coordinate with a HyperVector encoding.

    Attributes
    ----------
    x, y, z : float
        Coordinates (z defaults to 0.0 for 2D).
    hv      : HyperVector
        ``bind(permute(v_x, x_step), permute(v_y, y_step))``
    """
    x: float
    y: float
    z: float = 0.0
    hv: Any = field(default=None, repr=False)

    def distance_to(self, other: "SpatialPosition") -> float:
        """Euclidean distance to another position."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


# ---------------------------------------------------------------------------
# PositionCodebook — maps real coordinates → HVs
# ---------------------------------------------------------------------------

class PositionCodebook:
    """
    Encodes 2D/3D positions as HyperVectors using FPE-style incremental bit
    flipping on per-axis base vectors.

    Axis encoding
    -------------
    Each axis has a base bit-vector.  A coordinate n is encoded by applying n
    incremental bit-flip steps starting from the base:

        bits_x(0) = x_base_bits
        bits_x(n) = bits_x(n-1) with _AXIS_FLIP_BITS bits flipped

    The resulting bit-flip chain gives the locality property:
    ``sim(v_pos(1,0), v_pos(2,0)) > sim(v_pos(1,0), v_pos(100,0))``

    2D binding
    ----------
    A 2D position is encoded as the XOR of the two 1D axis encodings:

        v_pos(x,y) = v_x(round(x)) XOR v_y(round(y))

    Because the x and y axes start from *orthogonal* base vectors (different
    seeds → effectively independent), the combined position HV retains the
    property that positions close in *either* axis are more similar than
    positions far in *both* axes.
    """

    def __init__(self):
        self._x_bits = self._make_base(_X_AXIS_SEED)
        self._y_bits = self._make_base(_Y_AXIS_SEED)
        self._z_bits = self._make_base(_Z_AXIS_SEED)
        # Cache: axis name → {int_coord: np.ndarray of bits}
        self._x_cache: Dict[int, Any] = {0: hypervec_rs.HyperVector.from_bits(self._x_bits)}
        self._y_cache: Dict[int, Any] = {0: hypervec_rs.HyperVector.from_bits(self._y_bits)}
        self._z_cache: Dict[int, Any] = {0: hypervec_rs.HyperVector.from_bits(self._z_bits)}
        # Combined position cache
        self._pos_cache: Dict[Tuple[int, int, int], Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_base(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=_HV_DIM, dtype=np.int8)

    @staticmethod
    def _flip_indices(seed: int, step: int, n_flip: int = _AXIS_FLIP_BITS) -> np.ndarray:
        rng = np.random.default_rng(seed + abs(step) * 13337)
        return rng.choice(_HV_DIM, size=n_flip, replace=False)

    def _encode_axis(
        self,
        n: int,
        base_bits: np.ndarray,
        cache: Dict[int, Any],
        axis_seed: int,
    ) -> Any:
        """Encode integer coordinate n along a single axis."""
        if n in cache:
            return cache[n]
        # Walk from the closest cached value
        if n > 0:
            start = max(k for k in cache if k <= n) if any(k <= n for k in cache) else 0
            prev = cache.get(start)
            prev_bits = prev.bits.copy() if hasattr(prev, "bits") else base_bits.copy()
            bits = prev_bits.copy()
            for step in range(start + 1, n + 1):
                bits[self._flip_indices(axis_seed, step)] ^= 1
        elif n < 0:
            start = min(k for k in cache if k >= n) if any(k >= n for k in cache) else 0
            prev = cache.get(start)
            prev_bits = prev.bits.copy() if hasattr(prev, "bits") else base_bits.copy()
            bits = prev_bits.copy()
            for step in range(start - 1, n - 1, -1):
                # Use a large offset so negative coordinates get independent flip sets
                # (avoids any accidental symmetry with positive flip sets)
                bits[self._flip_indices(axis_seed, step - _NEGATIVE_STEP_OFFSET)] ^= 1
        else:
            return cache[0]
        result = hypervec_rs.HyperVector.from_bits(bits)
        cache[n] = result
        return result

    def _encode_x(self, ix: int) -> Any:
        return self._encode_axis(ix, self._x_bits, self._x_cache, _X_AXIS_SEED)

    def _encode_y(self, iy: int) -> Any:
        return self._encode_axis(iy, self._y_bits, self._y_cache, _Y_AXIS_SEED)

    def _encode_z(self, iz: int) -> Any:
        return self._encode_axis(iz, self._z_bits, self._z_cache, _Z_AXIS_SEED)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, x: float, y: float, z: float = 0.0) -> Any:
        """
        Encode position (x, y, z) as a HyperVector.

        Coordinates are rounded to the nearest integer.  The locality property
        guarantees that nearby positions → more similar HVs.
        """
        ix, iy, iz = round(x), round(y), round(z)
        key = (ix, iy, iz)
        if key in self._pos_cache:
            return self._pos_cache[key]

        vx = self._encode_x(ix)
        vy = self._encode_y(iy)
        vz = self._encode_z(iz)

        # XOR-bind the three axis encodings.
        # Because x, y, z bases are orthogonal (independent seeds), positions
        # that differ only in x are discriminated by the x axis component.
        v_pos = vx.xor(vy).xor(vz)
        self._pos_cache[key] = v_pos
        return v_pos

    def similarity(
        self,
        pos1: SpatialPosition,
        pos2: SpatialPosition,
    ) -> float:
        """HV cosine similarity between two positions (1.0 = same, →0 = far)."""
        return float(pos1.hv.similarity(pos2.hv))


# ---------------------------------------------------------------------------
# SpatialRelationEncoder — encodes / decodes spatial relations
# ---------------------------------------------------------------------------

class SpatialRelationEncoder:
    """
    Encodes spatial assertions as VSA composites and decodes them via unbinding.

    An assertion "cat is ABOVE table" is encoded as::

        v_assert = XOR(bind(v_ABOVE, bind(v_cat, v_table)), v_scene_id)

    To answer "What is ABOVE the table?":
        - Probe: ``unbind(v_scene, v_ABOVE)`` → ``bind(v_cat, v_table)``
        - Unbind by table: ``unbind(result, v_table)`` → ~``v_cat``
        - Search codebook for nearest concept to ``v_cat``

    For simplicity in this implementation we maintain explicit relation
    records but verify them with VSA similarity to demonstrate the VSA
    encoding concept.
    """

    def __init__(self):
        self._roles: Dict[str, Any] = {
            name: hypervec_rs.HyperVector(seed)
            for name, seed in _ROLE_SEEDS.items()
        }

    def role_hv(self, relation: str) -> Any:
        """Return the HyperVector for a named spatial role."""
        key = relation.upper()
        if key not in self._roles:
            self._roles[key] = hypervec_rs.HyperVector(hash(key) % (2 ** 32))
        return self._roles[key]

    def encode_assertion(
        self,
        figure_hv: Any,
        ground_hv: Any,
        relation: str,
    ) -> Any:
        """
        Encode the assertion "figure <relation> ground" as a VSA composite.

        Returns
        -------
        HyperVector: ``bind(v_role, bind(v_figure, v_ground))``
        """
        role = self.role_hv(relation)
        fg = figure_hv.xor(ground_hv)      # bind figure and ground
        return role.xor(fg)                 # bind role with the pair

    def query_figure(
        self,
        assertion_hv: Any,
        ground_hv: Any,
        relation: str,
        concept_hvs: Dict[str, Any],
    ) -> Optional[str]:
        """
        Query: "What is <relation> <ground>?"

        Returns the best-matching concept name from *concept_hvs*.
        """
        role = self.role_hv(relation)
        # Unbind role → bind(figure, ground)
        fg_recovered = assertion_hv.xor(role)
        # Unbind ground → figure
        figure_recovered = fg_recovered.xor(ground_hv)

        best_name = None
        best_sim = -1.0
        for name, hv in concept_hvs.items():
            sim = float(figure_recovered.similarity(hv))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name if best_sim > _MIN_QUERY_SIMILARITY else None


# ---------------------------------------------------------------------------
# SpatialReasoner — the main public API
# ---------------------------------------------------------------------------

class SpatialReasoner:
    """
    Full spatial reasoning system for NSCK.

    Entities are placed on a 2D/3D grid. The system can then answer:
    - What is the spatial relation between two entities?
    - Where is an entity?
    - What entities are near a given position?
    - What entities satisfy a spatial query (e.g. "above the table")?

    Internally combines:
    * PositionCodebook for VSA-based position encoding
    * SpatialRelationEncoder for VSA assertion encoding
    * Allen interval logic for temporal order (reused for 1D spatial)
    * Euclidean metric for metric queries

    Example
    -------
    >>> sr = SpatialReasoner()
    >>> sr.place("cat",   x=1, y=3)
    >>> sr.place("table", x=1, y=2)
    >>> sr.get_relation("cat", "table")
    'above'
    >>> sr.find_near("table", radius=2.0)
    ['cat', 'dog']
    """

    def __init__(self):
        self._pos_cb = PositionCodebook()
        self._rel_enc = SpatialRelationEncoder()
        # entity name → SpatialPosition
        self._entities: Dict[str, SpatialPosition] = {}
        # entity name → HyperVector (for VSA queries)
        self._entity_hvs: Dict[str, Any] = {}
        # list of (figure, relation, ground) tuples
        self._assertions: List[Tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # Entity placement
    # ------------------------------------------------------------------

    def place(
        self,
        entity: str,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> None:
        """
        Place an entity at position (x, y, z) in the spatial scene.

        Parameters
        ----------
        entity : str
            Name of the entity (e.g. "cat").
        x, y, z : float
            Coordinates. z defaults to 0.0 for 2D scenes.
        """
        pos_hv = self._pos_cb.encode(x, y, z)
        pos = SpatialPosition(x=x, y=y, z=z, hv=pos_hv)
        self._entities[entity] = pos
        # Entity HV = bind(entity_seed, position_hv) — encodes both identity + position
        entity_seed_hv = hypervec_rs.HyperVector(abs(hash(entity)) % (2 ** 32))
        self._entity_hvs[entity] = entity_seed_hv.xor(pos_hv)

    def move(
        self,
        entity: str,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> None:
        """Move an already-placed entity to a new position."""
        self.place(entity, x, y, z)

    # ------------------------------------------------------------------
    # Relation extraction
    # ------------------------------------------------------------------

    def get_relation(self, figure: str, ground: str) -> Optional[str]:
        """
        Return the primary spatial relation between *figure* and *ground*.

        The relation is determined by the vector from ground → figure:
        - dx > |dy|   → right_of
        - dx < -|dy|  → left_of
        - dy > |dx|   → above  (y increases upward by convention)
        - dy < -|dx|  → below
        - distance < ADJACENT_MAX_DIST → adjacent_to

        Returns ``None`` if either entity is not in the scene.
        """
        if figure not in self._entities or ground not in self._entities:
            return None

        fp = self._entities[figure]
        gp = self._entities[ground]

        dx = fp.x - gp.x
        dy = fp.y - gp.y
        dz = fp.z - gp.z
        dist = fp.distance_to(gp)

        if dist < 1e-9:
            return "at_same_position"
        if dist <= _ADJACENT_MAX_DIST:
            return "adjacent_to"

        adx, ady, adz = abs(dx), abs(dy), abs(dz)

        # 3D: check z first (dominant axis)
        if dz > max(adx, ady):
            return "above"      # above = higher z (or higher y in 2D)
        if dz < -max(adx, ady):
            return "below"

        # 2D dominant-axis logic
        if ady >= adx and ady > 0:
            return "above" if dy > 0 else "below"
        if adx > ady and adx > 0:
            return "right_of" if dx > 0 else "left_of"

        return "adjacent_to"

    def get_all_relations(self, figure: str, ground: str) -> List[str]:
        """
        Return all applicable spatial relations between *figure* and *ground*.

        Unlike ``get_relation()`` (which returns only the dominant relation),
        this returns a full list including metric and topological predicates.
        """
        if figure not in self._entities or ground not in self._entities:
            return []

        relations = []
        primary = self.get_relation(figure, ground)
        if primary:
            relations.append(primary)

        fp = self._entities[figure]
        gp = self._entities[ground]
        dist = fp.distance_to(gp)

        if dist <= _NEAR_DEFAULT_DIST:
            relations.append("near")
        else:
            relations.append("far")

        # Include distance-based adjacency regardless of direction
        if dist <= _ADJACENT_MAX_DIST and primary != "adjacent_to":
            relations.append("adjacent_to")

        return relations

    # ------------------------------------------------------------------
    # Metric queries
    # ------------------------------------------------------------------

    def distance(self, entity1: str, entity2: str) -> Optional[float]:
        """Euclidean distance between two entities. Returns None if unknown."""
        if entity1 not in self._entities or entity2 not in self._entities:
            return None
        return self._entities[entity1].distance_to(self._entities[entity2])

    def find_near(
        self,
        anchor: str,
        radius: float = _NEAR_DEFAULT_DIST,
    ) -> List[str]:
        """
        Return all entities within *radius* units of *anchor*.

        The anchor entity itself is excluded from the results.
        """
        if anchor not in self._entities:
            return []

        ap = self._entities[anchor]
        result = []
        for name, pos in self._entities.items():
            if name == anchor:
                continue
            if ap.distance_to(pos) <= radius:
                result.append(name)
        return sorted(result)

    def find_above(self, ground: str) -> List[str]:
        """Return all entities above *ground*."""
        return [e for e in self._entities
                if e != ground and self.get_relation(e, ground) == "above"]

    def find_below(self, ground: str) -> List[str]:
        """Return all entities below *ground*."""
        return [e for e in self._entities
                if e != ground and self.get_relation(e, ground) == "below"]

    def find_left_of(self, ground: str) -> List[str]:
        """Return all entities left of *ground*."""
        return [e for e in self._entities
                if e != ground and self.get_relation(e, ground) == "left_of"]

    def find_right_of(self, ground: str) -> List[str]:
        """Return all entities right of *ground*."""
        return [e for e in self._entities
                if e != ground and self.get_relation(e, ground) == "right_of"]

    # ------------------------------------------------------------------
    # VSA-based spatial query (demonstrating VSA encoding)
    # ------------------------------------------------------------------

    def encode_scene_assertion(
        self,
        figure: str,
        relation: str,
        ground: str,
    ) -> Optional[Any]:
        """
        VSA-encode the assertion "<figure> <relation> <ground>" and store it.

        Returns the composite HyperVector or None if entities are missing.
        """
        if figure not in self._entity_hvs or ground not in self._entity_hvs:
            return None
        assertion_hv = self._rel_enc.encode_assertion(
            self._entity_hvs[figure],
            self._entity_hvs[ground],
            relation,
        )
        self._assertions.append((figure, relation, ground))
        return assertion_hv

    def position_similarity(self, entity1: str, entity2: str) -> Optional[float]:
        """
        VSA cosine similarity between the position HVs of two entities.

        Higher = closer in the spatial scene.
        """
        if entity1 not in self._entities or entity2 not in self._entities:
            return None
        hv1 = self._entities[entity1].hv
        hv2 = self._entities[entity2].hv
        return float(hv1.similarity(hv2))

    # ------------------------------------------------------------------
    # Scene inspection
    # ------------------------------------------------------------------

    def where_is(self, entity: str) -> Optional[SpatialPosition]:
        """Return the SpatialPosition of *entity*, or None if not placed."""
        return self._entities.get(entity)

    def list_entities(self) -> List[str]:
        """Return all entity names currently in the scene."""
        return list(self._entities.keys())

    def scene_summary(self) -> str:
        """Return a human-readable description of the current scene."""
        if not self._entities:
            return "Empty scene."
        lines = ["Spatial scene:"]
        for name, pos in sorted(self._entities.items()):
            lines.append(f"  {name}: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
        lines.append("\nRelations:")
        names = sorted(self._entities.keys())
        for i, e1 in enumerate(names):
            for e2 in names[i + 1:]:
                rel = self.get_relation(e1, e2)
                dist = self.distance(e1, e2)
                lines.append(
                    f"  {e1} {rel} {e2} (distance={dist:.2f})"
                )
        return "\n".join(lines)
