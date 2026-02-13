"""
NSCK Perception Engine (Phase 2.2)
==================================
Handles Sensor Fusion and Cross-Modal Retrieval.

Components:
- CleanupMemory: Associative memory for restoring noisy vectors.
- FusionEngine: Binds Audio + Visual into a single Percept.

Math:
- Fusion = Majority( (Audio XOR Role_Audio) + (Visual XOR Role_Visual) )
- Retrieval = Cleanup( Fusion XOR Role_Audio )
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

# Constants
HV_DIM = 10000

class CleanupMemory:
    """
    Associative Memory (Nearest Neighbor).
    Stores <Label, Hypervector> pairs.
    """
    def __init__(self):
        self.memory: Dict[str, np.ndarray] = {}
        
    def learn(self, label: str, hv: np.ndarray):
        """Store a canonical vector."""
        self.memory[label] = hv
        
    def query(self, noisy_hv: np.ndarray) -> Tuple[str, float]:
        """
        Find closest match.
        Returns (Label, Similarity).
        """
        best_sim = -1.0
        best_label = "None"
        
        for label, stored_hv in self.memory.items():
            # Hamming Similarity
            diffs = np.logical_xor(noisy_hv, stored_hv)
            sim = 1.0 - np.mean(diffs)
            
            if sim > best_sim:
                best_sim = sim
                best_label = label
                
        return best_label, best_sim

class FusionEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        # Generate Role Vectors
        self.role_audio = self._gen_hv()
        self.role_visual = self._gen_hv()
        
    def _gen_hv(self) -> np.ndarray:
        return self.rng.randint(0, 2, size=HV_DIM).astype(bool)
        
    def fuse(self, audio: np.ndarray, visual: np.ndarray) -> np.ndarray:
        """
        Fuse Audio and Visual modalities.
        F = Majority( (A ^ Ra), (V ^ Rv) )
        """
        # Bind with roles
        bound_a = np.logical_xor(audio, self.role_audio)
        bound_v = np.logical_xor(visual, self.role_visual)
        
        # Bundle (Majority Limit of 2 items -> Random Tie Break)
        # We can implement 2-item majority by:
        # bit = a if rand() < 0.5 else v
        # Effectively 50% mix.
        
        mask = self.rng.rand(HV_DIM) < 0.5
        fused = np.where(mask, bound_a, bound_v)
        
        return fused
        
    def retrieve(self, fused: np.ndarray, modality: str, memory: CleanupMemory) -> Tuple[str, float]:
        """
        Query the fused vector for a specific modality.
        modality: "audio" or "visual"
        """
        if modality == "audio":
            # Q = F ^ Ra
            #   = (0.5(A^Ra) + 0.5(V^Rv)) ^ Ra
            #   = 0.5(A^Ra^Ra) + 0.5(V^Rv^Ra)
            #   = 0.5(A) + 0.5(Noise)
            query_vec = np.logical_xor(fused, self.role_audio)
            
        elif modality == "visual":
            query_vec = np.logical_xor(fused, self.role_visual)
        else:
            return "Error", 0.0
            
        return memory.query(query_vec)

class SpatialAnalyzer:
    """
    Analyzes spatial relationships between objects based on quadrants.
    Quadrants: 0=TL, 1=TR, 2=BL, 3=BR
    """
    @staticmethod
    def get_predicates(obj_a_quads: np.ndarray, obj_b_quads: np.ndarray) -> List[str]:
        """
        Compare two objects (represented by their 4-quadrant intensities).
        Returns list of predicates: ["LeftOf", "Above", etc.]
        """
        # quads is [TL_mean, TL_std, TR_mean, TR_std, ...]
        # Grab means
        a_tl, a_tr, a_bl, a_br = obj_a_quads[0], obj_a_quads[2], obj_a_quads[4], obj_a_quads[6]
        b_tl, b_tr, b_bl, b_br = obj_b_quads[0], obj_b_quads[2], obj_b_quads[4], obj_b_quads[6]
        
        # Weighted center of mass (rough estimate)
        a_x = (a_tr + a_br) - (a_tl + a_bl)
        a_y = (a_bl + a_br) - (a_tl + a_tr)
        
        b_x = (b_tr + b_br) - (b_tl + b_bl)
        b_y = (b_bl + b_br) - (b_tl + b_tr)
        
        predicates = []
        if a_x < b_x: predicates.append("LeftOf")
        if a_x > b_x: predicates.append("RightOf")
        if a_y < b_y: predicates.append("Above")
        if a_y > b_y: predicates.append("Below")
        
        return predicates

