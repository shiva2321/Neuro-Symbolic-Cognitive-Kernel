import torch
import torch.nn as nn
import torchhd

class VSABridge(nn.Module):
    """
    The Neural-Symbolic Bridge.
    
    Transforms the rate-coded output of the SNN (Neural) into a 
    Hypervector (Symbolic/Holographic).
    
    Mechanism:
    Hypervector = Sign(ProjectionMatrix * RateVector)
    
    This acts as a Locality Sensitive Hashing (LSH) function that preserves
    cosine similarity: similar images -> similar firing rates -> similar hypervectors.
    """
    
    def __init__(self, input_dim: int, vsa_dim: int = 10000, device: str = "cpu"):
        """
        Args:
            input_dim (int): Dimension of the SNN output (e.g., 128).
            vsa_dim (int): Dimension of the Hypervector (e.g., 10,000).
            device (str): 'cpu' or 'cuda'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.vsa_dim = vsa_dim
        self.device = device
        
        # Random Projection Matrix (Fixed, not learned via Backprop usually, 
        # though it can be fine-tuned if desired).
        # We sample from a Normal distribution N(0, 1/input_dim)
        self.projection = nn.Linear(input_dim, vsa_dim, bias=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=input_dim**-0.5)
        
        # Freeze the projection weights to ensure stable symbol grounding
        for param in self.projection.parameters():
            param.requires_grad = False
            
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project and Binarize.
        
        Args:
            x (Tensor): Rate vectors. Shape: (Batch, input_dim)
            
        Returns:
            Tensor: MAP Hypervectors (-1, +1). Shape: (Batch, vsa_dim)
        """
        # Linear Projection
        projected = self.projection(x)
        
        # Binarize/Sign to map to MAP architecture {-1, 1}
        # In torchhd MAP, vectors are usually normalized or sign-flipped.
        # torch.sign returns {-1, 0, 1}. We want to avoid 0s effectively.
        hypervector = torch.sign(projected)
        
        # Handle zeros (rare, but mathematically possible): map to 1
        hypervector[hypervector == 0] = 1.0
        
        return hypervector
