import torch
import torchhd
from typing import Optional, List, Union

class VSAEngine:
    """
    Core Vector Symbolic Architecture (VSA) Engine.
    
    This engine handles the creation and manipulation of hypervectors in a 
    high-dimensional space (D=10,000). It acts as the mathematical substrate 
    for the "Memory" component of the NCGN kernel.
    
    Operations supported (MAP Architecture):
    - Binding (*): XOR/Multiplication (Role-Filler association)
    - Bundling (+): Superposition/Majority Vote (Set aggregation)
    - Permutation (Π): Cyclic shift (Sequence encoding)
    """

    def __init__(self, dimensions: int = 10000, model: str = "MAP", device: Optional[str] = None):
        """
        Initialize the VSA Engine.

        Args:
            dimensions (int): The dimensionality of the vectors. Research suggests D >= 10,000 for robust orthogonality.
            model (str): VSA model type. Defaults to 'MAP' (Multiply-Add-Permute).
            device (str): Computation device ('cpu' or 'cuda'). Auto-detected if None.
        """
        self.d = dimensions
        self.model = model
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[VSAEngine] Initialized with D={self.d} on {self.device}")

    def create_vector(self, name: Optional[str] = None) -> torchhd.VSATensor:
        """
        Create a new random orthogonal hypervector.
        
        Args:
            name (str, optional): A label for debugging purposes.
            
        Returns:
            torchhd.VSATensor: A random hypervector of dimension D.
        """
        # torchhd.random creates a set of vectors. We ask for 1, then select index 0.
        return torchhd.random(1, self.d, vsa=self.model, device=self.device)[0]

    def create_batch(self, count: int) -> torchhd.VSATensor:
        """
        Create a batch of k random orthogonal hypervectors.
        
        Args:
            count (int): Number of vectors to generate.
            
        Returns:
            torchhd.VSATensor: A tensor of shape (count, D).
        """
        return torchhd.random(count, self.d, vsa=self.model, device=self.device)

    def bind(self, v1: torchhd.VSATensor, v2: torchhd.VSATensor) -> torchhd.VSATensor:
        """
        Bind two hypervectors. 
        Operation: v1 * v2 (Conservation of dimensionality).
        The result is orthogonal to both inputs.
        
        Usage: Associate a Role with a Filler (e.g., Position * (1,1)).
        """
        return v1.bind(v2)

    def bundle(self, vectors: List[torchhd.VSATensor]) -> torchhd.VSATensor:
        """
        Bundle (superimpose) a list of hypervectors.
        Operation: Majority Vote or Normalization sum.
        The result is similar (high cosine score) to all inputs.
        
        Usage: Create a Set or store multiple items in one memory trace.
        """
        if not vectors:
            raise ValueError("Cannot bundle an empty list of vectors.")
        
        # Stack to shape (N, D) then bundle
        batch = torch.stack(vectors)
        return torchhd.multiset(batch)

    def permute(self, v: torchhd.VSATensor, shifts: int = 1) -> torchhd.VSATensor:
        """
        Permute (shift) a hypervector.
        Operation: Cyclic shift.
        The result is orthogonal to the input.
        
        Usage: Encode time or sequence order.
        """
        return v.permute(shifts)

    def similarity(self, v1: torchhd.VSATensor, v2: torchhd.VSATensor) -> float:
        """
        Calculate cosine similarity between two vectors.
        Returns a float between -1.0 and 1.0.
        """
        return torchhd.cosine_similarity(v1, v2).item()

    def similarity_batch(self, query: torchhd.VSATensor, memory: torchhd.VSATensor) -> torch.Tensor:
        """
        Calculate cosine similarity between a query vector and a memory bank.
        
        Args:
            query: Shape (D,)
            memory: Shape (N, D)
            
        Returns:
            Tensor of shape (N,) containing similarity scores.
        """
        return torchhd.cosine_similarity(query, memory)
