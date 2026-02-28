"""
ProceduralMemory — skill caching and accelerated decision paths.

Stores (context_hv, action, reward) triples. When a familiar context is
encountered, returns the cached best action without full deliberation.

V4: Lowered familiarity threshold to 0.72 and added LSH-bucket fast recall.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

import python.core.vsa.hypervec_shim as hv_mod


def _compute_lsh(bits: np.ndarray, n_bits: int = 16, seed: int = 0xDEAD) -> int:
    """Compute a locality-sensitive hash key for a binary hypervector.

    Uses fixed random projections (seeded deterministically) to map the
    high-dimensional binary vector into a compact n_bits integer bucket.
    This gives O(1) candidate lookup instead of O(N) linear scan.
    """
    rng = np.random.default_rng(seed)
    projections = rng.integers(0, len(bits), size=n_bits)
    return int(sum((int(bits[i]) & 1) << j for j, i in enumerate(projections)))


@dataclass
class Skill:
    """A cached (context → action) skill."""
    context_hv: hv_mod.HyperVector
    action: str
    reward: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    label: Optional[str] = None


class ProceduralMemory:
    """
    Caches skills (context HV → action mappings) for fast retrieval.

    When a query context matches a stored skill above `familiarity_threshold`,
    the cached action is returned, bypassing full deliberation.

    V4: Threshold lowered to 0.72 (from 0.85) and LSH bucket index added for
    O(1) candidate lookup instead of O(N) linear scan.
    """

    def __init__(
        self,
        familiarity_threshold: float = 0.72,
        max_skills: int = 500,
        lsh_bits: int = 16,
    ) -> None:
        self.familiarity_threshold = familiarity_threshold
        self.max_skills = max_skills
        self._skills: List[Skill] = []
        self._hit_count: int = 0
        self._miss_count: int = 0
        # V4: LSH bucket index for fast candidate retrieval
        self._lsh_bits: int = lsh_bits
        self._lsh_buckets: Dict[int, List[int]] = {}

    def _get_lsh_key(self, hv: hv_mod.HyperVector) -> Optional[int]:
        """Compute LSH bucket key for a HyperVector. Returns None on error."""
        try:
            bits = np.asarray(hv.bits, dtype=np.float32)
            return _compute_lsh(bits, self._lsh_bits)
        except Exception:
            return None

    @property
    def lsh_bits(self) -> int:
        return self._lsh_bits

    def cache_skill(
        self,
        context_hv: hv_mod.HyperVector,
        action: str,
        reward: float,
        label: Optional[str] = None,
    ) -> Skill:
        """Store a new skill or update an existing one."""
        # Check if we already have a very similar context
        for idx, skill in enumerate(self._skills):
            if skill.context_hv.similarity(context_hv) > 0.95:
                # Update in-place if this reward is better
                if reward > skill.reward:
                    skill.action = action
                    skill.reward = reward
                skill.access_count += 1
                skill.last_accessed = time.time()
                return skill

        new_idx = len(self._skills)
        skill = Skill(
            context_hv=context_hv,
            action=action,
            reward=reward,
            label=label,
        )
        self._skills.append(skill)

        # V4: Register in LSH bucket
        lsh_key = self._get_lsh_key(context_hv)
        if lsh_key is not None:
            self._lsh_buckets.setdefault(lsh_key, []).append(new_idx)

        if len(self._skills) > self.max_skills:
            # Evict least recently used
            self._skills.sort(key=lambda s: s.last_accessed)
            self._skills = self._skills[-self.max_skills:]
            # Rebuild LSH index after eviction
            self._rebuild_lsh_index()
        return skill

    def _rebuild_lsh_index(self) -> None:
        """Rebuild the LSH bucket index from scratch (called after eviction)."""
        self._lsh_buckets = {}
        for idx, skill in enumerate(self._skills):
            lsh_key = self._get_lsh_key(skill.context_hv)
            if lsh_key is not None:
                self._lsh_buckets.setdefault(lsh_key, []).append(idx)

    def recall_action(
        self,
        query_hv: hv_mod.HyperVector,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Recall a cached action for a familiar context.
        Returns (action, similarity, reward) or None if no familiar match.

        V4: First checks the LSH bucket for O(1) candidate lookup,
        falls back to full O(N) scan if bucket is empty.
        """
        best_sim = 0.0
        best_skill: Optional[Skill] = None

        # V4: LSH-bucket fast path
        candidates_checked: Optional[List[int]] = None
        lsh_key = self._get_lsh_key(query_hv)
        if lsh_key is not None and lsh_key in self._lsh_buckets:
            candidates_checked = self._lsh_buckets[lsh_key]

        if candidates_checked:
            # Only check LSH bucket members
            for idx in candidates_checked:
                if idx < len(self._skills):
                    skill = self._skills[idx]
                    sim = float(skill.context_hv.similarity(query_hv))
                    if sim > best_sim:
                        best_sim = sim
                        best_skill = skill
        else:
            # Fall back to full linear scan when bucket is empty or LSH failed
            for skill in self._skills:
                sim = float(skill.context_hv.similarity(query_hv))
                if sim > best_sim:
                    best_sim = sim
                    best_skill = skill

        if best_skill is not None and best_sim >= self.familiarity_threshold:
            best_skill.access_count += 1
            best_skill.last_accessed = time.time()
            self._hit_count += 1
            return (best_skill.action, best_sim, best_skill.reward)

        self._miss_count += 1
        return None

    def get_statistics(self) -> Dict[str, Any]:
        total = self._hit_count + self._miss_count
        return {
            "total_skills": len(self._skills),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / max(1, total),
            "lsh_buckets": len(self._lsh_buckets),
        }
