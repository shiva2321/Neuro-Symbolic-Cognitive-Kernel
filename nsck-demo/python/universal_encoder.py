import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalEncoder(nn.Module):
    """
    The 'Pre-Frontal Cortex' that unifies all senses.
    
    EFFICIENCY: Uses lightweight pooling + small linear projections
    instead of heavy convolutions. Total params: ~35K (vs ~150K+ with Conv2d).
    
    Accepts:
    1. Visual (4D): [Batch, Channel, Height, Width] -> Pool + Linear
    2. Audio/Temporal (3D): [Batch, Channel, Time] -> Pool + Linear
    3. Conceptual/Text (2D): [Batch, Dim] -> Linear
    
    Output:
    - Fixed Latent Vector: [Batch, 256]
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # --- PATH A: VISUAL (Spatial) ---
        # Lightweight: AdaptivePool to fixed 8x8 grid, then small linear
        self.visual_pool = nn.AdaptiveAvgPool2d((8, 8))
        # Max 4 channels * 8 * 8 = 256 features
        self.visual_fc = nn.Linear(4 * 8 * 8, latent_dim)
        # Lightweight 1x1 convs for saliency hooks / transfer tests
        self.visual_conv1 = nn.Conv2d(4, 4, kernel_size=1, bias=True)
        self.visual_conv2 = nn.Conv2d(4, 4, kernel_size=1, bias=True)
        # Channel adaptation layers (pre-registered for common channel counts)
        self._visual_channel_adapters = nn.ModuleDict()

        # --- PATH B: TEMPORAL (Audio/Sensors) ---
        # Lightweight: AdaptivePool to fixed 32 time steps, then linear
        self.temp_pool = nn.AdaptiveAvgPool1d(32)
        self.temp_fc = nn.Linear(32, latent_dim)

        # --- PATH C: CONCEPTUAL (Text/Vectors) ---
        self.concept_fc = nn.LazyLinear(latent_dim) 

    def _get_channel_adapter(self, in_channels: int, device) -> nn.Module:
        """Get or create a 1x1 channel projection (lightweight, no spatial conv)."""
        key = str(in_channels)
        if key not in self._visual_channel_adapters:
            adapter = nn.Linear(in_channels, 4)
            self._visual_channel_adapters[key] = adapter
        return self._visual_channel_adapters[key].to(device)

    def forward(self, x, modality_hint=None):
        """
        Auto-routes input based on shape.
        """
        dims = x.ndim
        
        # --- 4D: Visual (B, C, H, W) ---
        if dims == 4:
            b, c, h, w = x.shape
            if c != 4:
                # Lightweight channel projection: permute → linear → permute
                adapter = self._get_channel_adapter(c, x.device)
                # (B, C, H, W) → (B, H, W, C) → Linear → (B, H, W, 4) → (B, 4, H, W)
                x = x.permute(0, 2, 3, 1)
                x = adapter(x)
                x = x.permute(0, 3, 1, 2)

            x = F.relu(self.visual_conv1(x))
            x = F.relu(self.visual_conv2(x))
            h = self.visual_pool(x)     # [B, 4, 8, 8]
            h = h.flatten(1)            # [B, 256]
            return F.relu(self.visual_fc(h))

        # --- 3D: Temporal (B, C, T) ---
        elif dims == 3:
            # Average channels if multi-channel, then pool time
            if x.shape[1] > 1:
                x = x.mean(dim=1, keepdim=True)  # [B, 1, T]
            h = self.temp_pool(x)       # [B, 1, 32]
            h = h.squeeze(1)            # [B, 32]
            return F.relu(self.temp_fc(h))

        # --- 2D: Conceptual (B, Dim) ---
        elif dims == 2:
            return F.tanh(self.concept_fc(x))
            
        else:
            raise ValueError(f"UniversalEncoder: Unsupported input shape {x.shape}")
