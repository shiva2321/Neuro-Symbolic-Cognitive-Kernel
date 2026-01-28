import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from ncgn.memory.vsa_core import VSAEngine

class AssociativeMemory:
    """
    Associative Memory (The "Codebook").
    
    Acts as the bridge between raw vectors and symbolic labels.
    It stores "Clean Concepts" and provides a probabilistic lookup mechanism
    based on the research in R5 (Dynamic Thresholding).
    """
    
    def __init__(self, engine: VSAEngine):
        self.engine = engine
        self.codebook: Dict[str, torch.Tensor] = {}
        self.labels: List[str] = []
        self.memory_tensor: Optional[torch.Tensor] = None # Shape (N, D) - Cached for fast batch query
        
    def add_concept(self, name: str, vector: Optional[torch.Tensor] = None):
        """
        Add a concept to the codebook.
        
        Args:
            name (str): Symbolic label (e.g., "Digit_5", "Apple")
            vector (Tensor, optional): The vector to store. If None, generates a new random one.
        """
        if vector is None:
            vector = self.engine.create_vector()
            
        self.codebook[name] = vector
        self.labels.append(name)
        self._refresh_memory_tensor()
        
    def _refresh_memory_tensor(self):
        """
        Rebuilds the efficient tensor lookup block from the dictionary.
        """
        if not self.codebook:
            self.memory_tensor = None
            return
            
        # Ensure consistent ordering based on self.labels
        vectors = [self.codebook[label] for label in self.labels]
        self.memory_tensor = torch.stack(vectors).to(self.engine.device)
        
    def query(self, vector: torch.Tensor) -> Tuple[str, float, float]:
        """
        Query the memory with a probe vector.
        
        Returns:
            (BestMatchLabel, SimilarityScore, ConfidenceZScore)
            
        Logic:
            1. Calculate similarities to all known concepts.
            2. Identify the best match.
            3. Calculate Z-Score against the distribution of NON-matches (the noise floor).
        """
        if self.memory_tensor is None:
            return ("None", 0.0, 0.0)
            
        # 1. Batch Similarity: Shape (N,)
        similarities = self.engine.similarity_batch(vector, self.memory_tensor)
        
        if len(self.labels) == 1:
            # Degenerate case: only 1 concept known
            score = similarities[0].item()
            return (self.labels[0], score, 0.0) # Z-score undefined for N=1
            
        # 2. Find Best Match
        best_val, best_idx = torch.max(similarities, dim=0)
        best_label = self.labels[best_idx.item()]
        
        # 3. Calculate Z-Score (Research R5)
        # We need statistics of the "Noise" (all other concepts)
        # Create a mask to exclude the best match
        mask = torch.ones(len(self.labels), dtype=torch.bool, device=self.engine.device)
        mask[best_idx] = False
        
        noise_dist = similarities[mask]
        
        mu_noise = torch.mean(noise_dist).item()
        sigma_noise = torch.std(noise_dist).item()
        
        if sigma_noise < 1e-9:
            z_score = 100.0 if best_val > mu_noise else 0.0
        else:
            z_score = (best_val.item() - mu_noise) / sigma_noise
        
        # DEBUG
        print(f"DEBUG Z: Best={best_val.item():.4f}, Mu={mu_noise:.4f}, Sigma={sigma_noise:.4f} -> Z={z_score:.2f}")
            
        return (best_label, best_val.item(), z_score)

    def get_vector(self, name: str) -> Optional[torch.Tensor]:
        return self.codebook.get(name)
