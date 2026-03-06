"""
SocietalSnapshot — observational snapshot of the SocietyManager state (V30).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class SocietalSnapshot:
    """Full observational snapshot of the societal hypervector system at a single epoch.

    Created by ``SocietyManager.snapshot()``.
    """

    epoch: int
    n_concepts: int
    n_bonds: int
    avg_bond_strength: float
    avg_activation: float
    n_communities: int
    percolation_threshold: float
    giant_component_size: int
    top_activated: List[Tuple[str, float]] = field(default_factory=list)
    top_bonded: List[Tuple[str, int]] = field(default_factory=list)
    domain_tree_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "n_concepts": self.n_concepts,
            "n_bonds": self.n_bonds,
            "avg_bond_strength": round(self.avg_bond_strength, 4),
            "avg_activation": round(self.avg_activation, 4),
            "n_communities": self.n_communities,
            "percolation_threshold": round(self.percolation_threshold, 4),
            "giant_component_size": self.giant_component_size,
            "top_activated": [(c, round(a, 4)) for c, a in self.top_activated],
            "top_bonded": list(self.top_bonded),
            "domain_tree_summary": self.domain_tree_summary,
        }
