"""
ProceduralMemory — skill caching and accelerated decision paths.

Stores (context_hv, action, reward) triples. When a familiar context is
encountered, returns the cached best action without full deliberation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

import python.core.vsa.hypervec_shim as hv_mod


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
    """

    def __init__(
        self,
        familiarity_threshold: float = 0.85,
        max_skills: int = 500,
    ) -> None:
        self.familiarity_threshold = familiarity_threshold
        self.max_skills = max_skills
        self._skills: List[Skill] = []
        self._hit_count: int = 0
        self._miss_count: int = 0

    def cache_skill(
        self,
        context_hv: hv_mod.HyperVector,
        action: str,
        reward: float,
        label: Optional[str] = None,
    ) -> Skill:
        """Store a new skill or update an existing one."""
        # Check if we already have a very similar context
        for skill in self._skills:
            if skill.context_hv.similarity(context_hv) > 0.95:
                # Update in-place if this reward is better
                if reward > skill.reward:
                    skill.action = action
                    skill.reward = reward
                skill.access_count += 1
                skill.last_accessed = time.time()
                return skill

        skill = Skill(
            context_hv=context_hv,
            action=action,
            reward=reward,
            label=label,
        )
        self._skills.append(skill)
        if len(self._skills) > self.max_skills:
            # Evict least recently used
            self._skills.sort(key=lambda s: s.last_accessed)
            self._skills = self._skills[-self.max_skills:]
        return skill

    def recall_action(
        self,
        query_hv: hv_mod.HyperVector,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Recall a cached action for a familiar context.
        Returns (action, similarity, reward) or None if no familiar match.
        """
        best_sim = 0.0
        best_skill: Optional[Skill] = None
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
        }
