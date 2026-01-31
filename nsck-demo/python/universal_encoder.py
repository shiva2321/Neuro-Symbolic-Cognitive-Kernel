import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalEncoder(nn.Module):
    """
    The 'Pre-Frontal Cortex' that unifies all senses.
    
    Accepts:
    1. Visual (4D): [Batch, Channel, Height, Width] -> CNN
    2. Audio/Temporal (3D): [Batch, Channel, Time] -> 1D Conv
    3. Conceptual/Text (2D): [Batch, Dim] -> Linear
    
    Output:
    - Fixed Latent Vector: [Batch, 256]
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # --- PATH A: VISUAL (Spatial) ---
        # "The Eyes"
        # Input: [B, C, H, W] -> Output: [B, 256]
        self.visual_conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1) # Assume 4 channels (RGBA/Stacked)
        self.visual_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.visual_pool = nn.AdaptiveAvgPool2d((4, 4)) # FORCE everything to 4x4
        self.visual_fc = nn.Linear(32 * 4 * 4, latent_dim)

        # --- PATH B: TEMPORAL (Audio/Sensors) ---
        # "The Ears"
        # Input: [B, C, T] -> Output: [B, 256]
        self.temp_conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1) # Assume mono/single sensor
        self.temp_pool = nn.AdaptiveAvgPool1d(16) # FORCE time to 16 distinct "moments"
        self.temp_fc = nn.Linear(16 * 16, latent_dim)

        # --- PATH C: CONCEPTUAL (Text/Vectors) ---
        # "The Wernicke's Area"
        # Input: [B, Dim] -> Output: [B, 256]
        # We use a lazy linear projection (instantiated on first forward)
        # or a generic MLP if dim is known. For now, we assume simple projection.
        self.concept_fc = nn.LazyLinear(latent_dim) 

    def forward(self, x, modality_hint=None):
        """
        Auto-routes input based on shape.
        """
        dims = x.ndim
        
        # --- 4D: Visual (B, C, H, W) ---
        if dims == 4:
            # Check channel count, project if not 4
            if x.shape[1] != 4:
                # Dynamic Channel Adaptation
                if not hasattr(self, 'adapt_conv'):
                    # Create a 1x1 conv to map [Available Channels] -> 4
                    # We send to device same as input
                    self.adapt_conv = nn.Conv2d(x.shape[1], 4, kernel_size=1).to(x.device)
                x = self.adapt_conv(x)
            
            h = F.relu(self.visual_conv1(x))
            h = F.relu(self.visual_conv2(h))
            h = self.visual_pool(h) # [B, 32, 4, 4]
            h = h.flatten(1)        # [B, 512]
            return F.relu(self.visual_fc(h))

        # --- 3D: Temporal (B, C, T) ---
        elif dims == 3:
            # Handle variable input channels
            h = F.relu(self.temp_conv1(x))
            h = self.temp_pool(h)   # [B, 16, 16]
            h = h.flatten(1)
            return F.relu(self.temp_fc(h))

        # --- 2D: Conceptual (B, Dim) ---
        elif dims == 2:
            return F.tanh(self.concept_fc(x))
            
        else:
            raise ValueError(f"UniversalEncoder: Unsupported input shape {x.shape}")
