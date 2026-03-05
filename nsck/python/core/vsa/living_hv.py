import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Any

from python.core.vsa.hypervec_shim import HyperVector

# Try optional fast-path (parallel DashMap backend)
try:
    from python.core.vsa.hypervec_shim import LivingHVStore
except ImportError:
    LivingHVStore = None

@dataclass
class LivingHyperVector:
    """A 'person' in the semantic world, wrapping a 10,240-bit HyperVector."""
    hv: HyperVector
    concept_id: str
    
    # --- DOMAIN AFFILIATION (the "address") ---
    domain_affinities: Dict[str, float] = field(default_factory=dict)
    primary_domain: Optional[str] = field(default=None)
    neighborhood_id: Optional[str] = field(default=None)
    
    # --- TEMPORAL PROPERTIES (the "age & history") ---
    birth_epoch: int = field(default_factory=lambda: int(time.time()))
    last_activated_epoch: int = field(default_factory=lambda: int(time.time()))
    activation_count: int = field(default=0)
    activation_history: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    
    # --- VALENCE (the "relationship potential") ---
    valence: int = field(default=4) # 1-8 
    current_bonds: Dict[str, float] = field(default_factory=dict)
    electronegativity: float = field(default=0.5)
    hybridization_state: str = field(default="unhybridized")
    
    # --- STABILITY (the "zoning class") ---
    stability_class: str = field(default="gas") # diamond, molecular, gas
    ewc_protection: float = field(default=0.0)
    schema_membership: List[str] = field(default_factory=list)
    
    # --- SOCIAL PROPERTIES (the "reputation") ---
    hub_score: float = field(default=0.0)
    bridge_score: float = field(default=0.0)
    novelty_score: float = field(default=1.0)
    
    # --- SOURCE PROVENANCE ---
    source: str = field(default="native")
    transplant_domain: Optional[str] = field(default=None)
    # Raw (unnormalised) scores — kept so that each call to update_domain_affinity
    # normalises from the original values rather than from already-normalised ones.
    _raw_domain_scores: Dict[str, float] = field(default_factory=dict)

    def activate(self, epoch: int):
        self.last_activated_epoch = epoch
        self.activation_count += 1
        self.activation_history.append(epoch)

    def update_domain_affinity(self, domain_name: str, score: float):
        # Store the raw score first, then re-normalise all domains from raw values.
        # This prevents double-normalisation when multiple domains are updated
        # sequentially (e.g. physics=0.8 then chemistry=0.2 should give 0.8/0.8).
        self._raw_domain_scores[domain_name] = score
        total = sum(self._raw_domain_scores.values())
        if total > 0:
            for k, v in self._raw_domain_scores.items():
                self.domain_affinities[k] = v / total
        else:
            self.domain_affinities[domain_name] = score

        # update primary domain
        if self.domain_affinities:
            self.primary_domain = max(self.domain_affinities.items(), key=lambda x: x[1])[0]

class ValenceEngine:
    """Manages semantic bonding between LivingHyperVectors modeled after Chemical Valence."""
    
    def __init__(self, min_sim: float = 0.60, min_co_activations: int = 3, decay_epochs: int = 1000):
        self.min_sim = min_sim
        self.min_co_activations = min_co_activations
        self.decay_epochs = decay_epochs
        self.co_activation_counts: Dict[tuple[str, str], int] = {}
        
    def record_co_activation(self, id_a: str, id_b: str):
        key = tuple(sorted([id_a, id_b]))
        self.co_activation_counts[key] = self.co_activation_counts.get(key, 0) + 1
        
    def try_bind(self, l_a: LivingHyperVector, l_b: LivingHyperVector) -> bool:
        """Attempt to form a covalent knowledge bond."""
        if len(l_a.current_bonds) >= l_a.valence or len(l_b.current_bonds) >= l_b.valence:
            return False # Subshells full
            
        key = tuple(sorted([l_a.concept_id, l_b.concept_id]))
        co_acts = self.co_activation_counts.get(key, 0)
        
        if co_acts < self.min_co_activations:
            return False
            
        # Noise-robust similarity 
        sim = l_a.hv.similarity_robust(l_b.hv, method='cosine')
        if sim < self.min_sim:
            return False
            
        stability_factor = 1.0
        if l_a.stability_class == "diamond": stability_factor *= 1.2
        if l_b.stability_class == "diamond": stability_factor *= 1.2
            
        bond_strength = sim * (1.0 + 0.1 * co_acts) * stability_factor
        
        l_a.current_bonds[l_b.concept_id] = bond_strength
        l_b.current_bonds[l_a.concept_id] = bond_strength
        
        self.update_hybridization(l_a)
        self.update_hybridization(l_b)
        
        return True

    def remove_stale_bonds(self, lx: LivingHyperVector, current_epoch: int):
        if lx.stability_class == "diamond":
            return # Diamond covalent bonds don't easily decay
            
        if (current_epoch - lx.last_activated_epoch) > self.decay_epochs:
            # Gas/Molecular bonds weaken
            to_remove = []
            for neighbor_id, strength in list(lx.current_bonds.items()):
                new_strength = strength * 0.9 # 10% decay
                if new_strength < 0.35:
                    to_remove.append(neighbor_id)
                else:
                    lx.current_bonds[neighbor_id] = new_strength
                    
            for nv in to_remove:
                del lx.current_bonds[nv]
                
            if to_remove:
                self.update_hybridization(lx)

    def update_hybridization(self, l: LivingHyperVector):
        n = len(l.current_bonds)
        if n >= 4:
            l.hybridization_state = "sp3"
        elif n == 3:
            l.hybridization_state = "sp2"
        elif n == 2:
            l.hybridization_state = "sp"
        else:
            l.hybridization_state = "unhybridized"
