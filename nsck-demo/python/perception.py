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

