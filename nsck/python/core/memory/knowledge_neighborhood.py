from typing import Dict, List, Optional, Set
import time
from dataclasses import dataclass, field
import numpy as np

from python.core.vsa.hypervec_shim import HyperVector
from python.core.vsa.living_hv import LivingHyperVector

@dataclass
class KnowledgeNeighborhood:
    """
    A tightly clustered region of LivingHyperVectors.
    Analogous to a town or city district.
    """
    neighborhood_id: str
    domain_name: str
    anchor_hv: HyperVector
    
    members: Set[str] = field(default_factory=set)
    parent_id: Optional[str] = None # Area or Section ID
    creation_epoch: int = field(default_factory=lambda: int(time.time()))
    stability_score: float = 0.5
    
    # Social Meta-Properties
    social_energy: float = 1.0
    collective_valence: float = 0.0
    
    # Structural properties
    betti_0: int = 1
    betti_1: int = 0
    average_valence: float = 0.0
    
    def add_member(self, concept_id: str):
        self.members.add(concept_id)
        
    def remove_member(self, concept_id: str):
        if concept_id in self.members:
            self.members.remove(concept_id)
            
    def compute_centroid(self, registry: Dict[str, LivingHyperVector]) -> Optional[HyperVector]:
        """Recalculate the anchor HV based on current members."""
        if not self.members:
            return None
            
        hvs = [registry[cid].hv for cid in self.members if cid in registry]
        if not hvs:
            return None
            
        # Bundle all member HVs to find the new centroid
        centroid = hvs[0]
        for idx in range(1, len(hvs)):
            centroid = centroid.bundle(hvs[idx])
            
        return centroid

@dataclass
class KnowledgeDomain:
    """
    A large-scale knowledge structure containing multiple neighborhoods.
    Analogous to a State or Country (e.g., 'Physics', 'Visual', 'Procedural').
    """
    domain_name: str
    hierarchy_level: str = "State" # town, city, state, country, continent
    parent_domain_id: Optional[str] = None
    
    neighborhoods: Dict[str, KnowledgeNeighborhood] = field(default_factory=dict)
    sub_domains: Dict[str, 'KnowledgeDomain'] = field(default_factory=dict)
    global_centroid: Optional[HyperVector] = None
    
    # Monitoring properties
    percolated: bool = False
    emergence_epoch: Optional[int] = None
    
    tda_health_history: List[Dict[str, float]] = field(default_factory=list)
    graph_entropy: float = 1.0
    
    def add_neighborhood(self, n: KnowledgeNeighborhood):
        self.neighborhoods[n.neighborhood_id] = n
        
    def update_global_centroid(self):
        """Bundle all neighborhood anchors to represent the entire domain."""
        anchors = [n.anchor_hv for n in self.neighborhoods.values()]
        if not anchors:
            return
            
        centroid = anchors[0]
        for idx in range(1, len(anchors)):
            centroid = centroid.bundle(anchors[idx])
        self.global_centroid = centroid
        
    def measure_health(self) -> float:
        """
        Aggregate stability and TDA metrics to get domain health.
        1.0 is perfectly consistent, 0.0 is scattered noise.
        """
        if not self.neighborhoods:
            return 0.0
            
        avg_stability = float(np.mean([n.stability_score for n in self.neighborhoods.values()]))
        
        # Penalize if too many holes (betti_1) compared to components (betti_0)
        total_b0 = sum(n.betti_0 for n in self.neighborhoods.values())
        total_b1 = sum(n.betti_1 for n in self.neighborhoods.values())
        
        topology_penalty = 1.0
        if total_b0 > 0:
            hole_ratio = total_b1 / total_b0
            topology_penalty = max(0.1, 1.0 - (hole_ratio * 0.1))
            
        return avg_stability * topology_penalty
